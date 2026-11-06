
"""Translation-synchronised MPV playback.

mpv listens on a local TCP port (paused); ingest's ffmpeg streams video+audio into it.
VOD playback starts as soon as translation is 5 seconds ahead, then pauses
whenever translated coverage falls to that lead. No deliberate video prebuffer.
For live streams, delay_s controls how far behind live playback starts.

A scheduler polls mpv's time-pos over JSON IPC (~10x/s) and draws subtitles
using osd-overlay.
"""
import ctypes
import json
import logging
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MpvConfig
from ..ingest import Ingest, fmt_hms
from ..types import Subtitle
from ..timeline import record_session_anchor

log = logging.getLogger(__name__)

IPC_PATH = (
    r"\\.\pipe\auto-translate-mpv"
    if sys.platform == "win32"
    else "/tmp/auto-translate-mpv.sock"
)
WINDOWS_DEFAULT = Path(r"C:\Program Files\MPV Player\mpv.exe")
OVERLAY_ID = 1
INPUT_CONF = Path(__file__).with_name("mpv_input.conf")
SEEKBAR_SCRIPT = Path(__file__).with_name("mpv_seekbar.lua")

JUMP_MESSAGE = "auto-translate-jump"


def _create_kill_job(proc: subprocess.Popen) -> Any:
    """Automatically terminate MPV when the Python process exits."""
    if sys.platform != "win32":
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.CreateJobObjectW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)

    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000

    if not kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)

    if not kernel32.AssignProcessToJobObject(
        job,
        wintypes.HANDLE(int(proc._handle)),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)

    return job


def _close_job(job: Any) -> None:
    """Close the Windows Job Object handle."""
    if job is None or sys.platform != "win32":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(job)


class MpvError(Exception):
    pass


class MpvIpc:
    """mpv's JSON IPC: one JSON object per line."""

    def __init__(self, path: str, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s

        while True:
            try:
                if sys.platform == "win32":
                    self._f = open(path, "r+b", buffering=0)
                else:
                    sock = socket.socket(socket.AF_UNIX)
                    sock.connect(path)
                    self._f = sock.makefile("rwb", buffering=0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

        self._lock = threading.Lock()
        self._next_id = 0
        self.events: queue.Queue[dict] = queue.Queue()

    def command(self, *args: Any, **named: Any) -> Any:
        """Send one command and wait for its response."""
        with self._lock:
            self._next_id += 1
            rid = self._next_id

            cmd = {"name": args[0], **named} if named else list(args)

            self._f.write(
                (
                    json.dumps(
                        {"command": cmd, "request_id": rid}
                    ) + "\n"
                ).encode()
            )

            while True:
                line = self._f.readline()

                if not line:
                    raise ConnectionError("mpv closed the IPC connection")

                msg = json.loads(line)

                if msg.get("request_id") == rid:
                    if msg.get("error") != "success":
                        raise MpvError(msg.get("error"))
                    return msg.get("data")

                if "event" in msg:
                    self.events.put(msg)

    def get(self, prop: str) -> Any:
        """Return a property's value, or None if unavailable."""
        try:
            return self.command("get_property", prop)
        except MpvError:
            return None

    def close(self) -> None:
        self._f.close()


@dataclass
class _Line:
    sub: Subtitle
    show_from: float
    show_until: float


def find_mpv(configured: str) -> str:
    if configured:
        return configured

    if found := shutil.which("mpv"):
        return found

    if sys.platform == "win32" and WINDOWS_DEFAULT.exists():
        return str(WINDOWS_DEFAULT)

    raise SystemExit(
        "mpv not found: install it (winget install shinchiro.mpv) "
        "or set [mpv] path"
    )


class MpvOutput:
    # Keep playback this far behind completed translation.
    PAUSE_LEAD_SECONDS = 5.0

    def __init__(
        self,
        cfg: MpvConfig,
        ingest: Ingest,
        stop: threading.Event,
    ) -> None:
        self.cfg = cfg
        self.ingest = ingest
        self.stop = stop

        self._lines: list[_Line] = []
        self._lock = threading.Lock()
        self._now: float | None = None
        self._shown = ""
        self._lags: list[float] = []
        self._session_start = ingest.source.start_s
        self._buffer_condition = threading.Condition()
        self._translated_until = ingest.source.start_s
        self._source_finished = False
        self._job = None
        self._closed = False
        self._video_retries = 0
        self._pending_seek_offset: float | None = None
        self._clock_sample_logged = False
        self._clock_last_sample_at = 0.0
        self._clock_last_sample_pos: float | None = None
        self._mpv_anchor: float | None = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "initial"
        self._logged_subtitle_generations: set[int] = set()
        self._logged_displayed_subtitles: set[tuple[int, float, float]] = set()
        self._audio_clock_diag_until = 0.0
        self._audio_clock_diag_last_at = 0.0
        self._audio_clock_diag_done = False

        self._proc = subprocess.Popen([
            find_mpv(cfg.path),

            "--pause",
            f"--input-ipc-server={IPC_PATH}",

            "--force-window=yes",
            "--title=auto-translate",
            "--keep-open=yes",
            "--idle=yes",
            "--rebase-start-time=no",

            # Keep a modest forward/backward demux cache while reading the
            # progressive local fMP4 HLS playlist.
            "--cache=yes",
            "--demuxer-readahead-secs=12",
            "--demuxer-max-bytes=128MiB",
            "--demuxer-max-back-bytes=32MiB",
            # Match FFmpeg's live HLS reader: begin at the oldest retained segment.
            "--demuxer-lavf-o=live_start_index=0",
            "--input-default-bindings=no",

            # Disable unsafe on-screen controls.
            "--osc=no",
            "--osd-on-seek=no",
            "--osd-bar=no",
            "--no-input-default-bindings",
            f"--input-conf={INPUT_CONF}",
            f"--script={SEEKBAR_SCRIPT}",

            # Hardware video decoding.
            f"--hwdec={cfg.hwdec}",

            "--msg-level=all=warn",
        ])

        if sys.platform == "win32":
            try:
                self._job = _create_kill_job(self._proc)
                log.info("mpv automatic process cleanup enabled")
            except OSError:
                log.exception(
                    "Could not enable automatic mpv process cleanup"
                )

        try:
            self.ipc = MpvIpc(IPC_PATH)

            # Wait for the idle MPV IPC endpoint to become responsive.
            deadline = time.monotonic() + 10

            while (
                self.ipc.get("idle-active") is None
                and time.monotonic() < deadline
            ):
                if self._proc.poll() is not None:
                    raise RuntimeError("mpv exited during startup")
                time.sleep(0.1)

            time.sleep(0.5)

        except Exception:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()

            _close_job(self._job)
            self._job = None
            raise

        log.info(
            "mpv ready for progressive fMP4 HLS playback "
            "(live delay %.1fs, VOD translation lead %.1fs)",
            cfg.delay_s,
            self.PAUSE_LEAD_SECONDS,
        )

        self._thread = threading.Thread(
            target=self._run,
            name="mpv",
            daemon=True,
        )
        self._thread.start()

    # ---- Output interface ----

    def show(self, sub: Subtitle) -> None:
        cfg = self.cfg

        if sub.end_s < self._session_start:
            return

        # Use the first speech-positive VAD frame, not the earlier padded
        # audio start used to preserve context for ASR.
        speech_start = (
            sub.speech_start_s
            if sub.speech_start_s is not None
            else sub.start_s
        )
        show_from = speech_start
        log.debug(
            "subtitle timing: audio_start=%.3f vad_onset=%.3f "
            "mpv_show=%.3f preroll=%.3fs",
            sub.start_s,
            speech_start,
            show_from,
            max(0.0, speech_start - sub.start_s),
        )
        until = (
            max(sub.end_s, show_from + cfg.min_display_s)
            + cfg.hold_s
        )

        line = _Line(sub, show_from, until)
        now = self._now

        if sub.generation not in self._logged_subtitle_generations:
            self._logged_subtitle_generations.add(sub.generation)
            log.info(
                "first subtitle received: generation=%d speech=%.3f-%.3f "
                "mpv_clock=%.3f delta=%+.3fs",
                sub.generation,
                sub.start_s,
                sub.end_s,
                now if now is not None else float("nan"),
                (now - sub.start_s) if now is not None else float("nan"),
            )

        if now is not None and now > until:
            log.warning(
                "dropping expired subtitle instead of shifting it late: "
                "generation=%d speech_end=%.3f mpv_clock=%.3f late=%.3fs",
                sub.generation,
                sub.end_s,
                now,
                now - sub.end_s,
            )
            return

        with self._lock:
            self._lines.append(line)

    def advance(self, progress_s: float) -> None:
        """Record the latest audio position safely processed for translation."""
        with self._buffer_condition:
            self._translated_until = max(
                self._translated_until,
                progress_s,
            )
            self._buffer_condition.notify_all()

    def reset_buffer(self, progress_s: float) -> None:
        """Reset translated coverage after a VOD seek."""
        with self._buffer_condition:
            self._translated_until = progress_s
            self._source_finished = False
            self._buffer_condition.notify_all()

    def finish(self) -> None:
        """Signal that no more translated audio will arrive."""
        with self._buffer_condition:
            self._source_finished = True
            self._buffer_condition.notify_all()

    def drain(self) -> None:
        """Keep MPV open on the final frame until the user closes it."""
        while not self.stop.is_set():
            if self._proc.poll() is not None:
                return

            if self.ipc.get("eof-reached"):
                log.info(
                    "VOD playback finished; MPV will remain open until closed"
                )
                while not self.stop.is_set() and self._proc.poll() is None:
                    time.sleep(0.25)
                return

            time.sleep(0.25)

    def close(self) -> None:
        """Cleanly stop MPV and release its automatic cleanup protection."""
        if self._closed:
            return

        self._closed = True
        self.stop.set()

        with self._buffer_condition:
            self._buffer_condition.notify_all()

        if self._lags:
            on_time = sum(abs(x) <= 0.5 for x in self._lags)

            log.info(
                "mpv sync: %d lines shown, %d within 0.5s of speech, "
                "avg %+.2fs, worst %+.2fs",
                len(self._lags),
                on_time,
                sum(self._lags) / len(self._lags),
                max(self._lags, key=abs),
            )

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2)

        try:
            if self._proc.poll() is None:
                self.ipc.command("quit")
        except (ConnectionError, OSError, MpvError):
            pass

        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                log.error("Could not stop mpv process")

        finally:
            try:
                self.ipc.close()
            except (AttributeError, OSError):
                pass

            _close_job(self._job)
            self._job = None

    # ---- Scheduler ----

    def _run(self) -> None:
        try:
            first = True

            while not self.stop.is_set():
                self._play_session(show_hint=first)
                first = False

        except (
            ConnectionError,
            OSError,
            json.JSONDecodeError,
            MpvError,
        ):
            if not self.stop.is_set():
                log.exception("mpv playback/IPC error; stopping")
                self.stop.set()

    def _play_session(self, show_hint: bool) -> None:
        """Play only while translated subtitles remain sufficiently ahead."""

        lead = self.PAUSE_LEAD_SECONDS

        while not self.ingest.first_pts_ready.wait(0.2):
            if self.stop.is_set():
                return

        assert self.ingest.first_pts is not None

        # The cache writer has already produced its initialization segment and
        # first media fragment. MPV reads the same growing event playlist as
        # the ASR audio extractor.
        playlist = self.ingest.video_playlist_url
        self.ipc.command("loadfile", playlist, "replace")
        if not self._wait_for_event("start-file", timeout_s=10.0):
            log.error("MPV did not start loading the fMP4 HLS playlist")
            self.stop.set()
            return
        if not self._wait_for_event("file-loaded", timeout_s=15.0):
            log.error("MPV could not load the initial fMP4 HLS fragments")
            self.stop.set()
            return

        self._session_start = self.ingest.source.start_s

        # Audio extraction and MPV read the same fMP4 playlist, so both
        # use its media timeline. Measure MPV's local time-pos origin rather
        # than estimating it from the upstream YouTube audio PTS.
        self._mpv_anchor = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "seek" if self.ingest.generation > 0 else "initial"
        mapping = "measured MPV session anchor against shared fMP4 source timeline"

        if self._pending_seek_offset is not None:
            seek_offset = self._pending_seek_offset
            self.ipc.command("seek", seek_offset, "absolute+exact")
            deadline = time.monotonic() + 5.0
            measured_pos = None
            while time.monotonic() < deadline:
                pos = self.ipc.get("time-pos")
                if pos is not None:
                    measured_pos = float(pos)
                    if abs(measured_pos - seek_offset) <= 0.75:
                        break
                time.sleep(0.05)
            # The HLS playlist's time-pos is relative to its first segment;
            # keep the requested fragment offset as the source-clock anchor.
            self._mpv_anchor = seek_offset
            record_session_anchor(
                self.ingest.source.timeline,
                self._clock_source_anchor,
                self._mpv_anchor,
                self._anchor_reason,
            )
            log.info(
                "reused cached fMP4 fragments: playlist_offset=%.3fs observed_mpv_pos=%s",
                seek_offset,
                f"{measured_pos:.3f}" if measured_pos is not None else "unavailable",
            )
            self._pending_seek_offset = None

        self._clock_sample_logged = False
        self._audio_clock_diag_until = 0.0
        self._audio_clock_diag_last_at = 0.0
        self._audio_clock_diag_done = False
        log.info(
            "MPV clock mapping: source_start=%.3f; waiting for measured "
            "MPV session anchor; %s",
            self._clock_source_anchor,
            mapping,
        )

        if self.ingest.source.live:
            unpause_at = (
                self.ingest.first_pts_at
                + self.cfg.delay_s
            )
            log.info(
                "live playback delay: %.1fs",
                self.cfg.delay_s,
            )
        else:
            # Start the stream immediately, but keep MPV paused until the
            # translator has built a small lead. This is translation waiting,
            # not video prebuffering.
            self._now = self._session_start
            unpause_at = None

            log.info(
                "waiting for translation to reach %s (+%.0fs)",
                fmt_hms(self._session_start + lead),
                lead,
            )

        if self.stop.is_set():
            return

        paused = True

        while not self.stop.is_set():

            if self.ingest.video_failed.is_set():
                if self._recover_video_stream():
                    return
                return

            if (
                not self.ingest.source.live
                and not paused
                and self._now is not None
            ):
                with self._buffer_condition:
                    translated_lead = self._translated_until - self._now
                    should_pause = (
                        not self._source_finished
                        and translated_lead <= lead
                    )

                if should_pause:
                    self.ipc.command(
                        "set_property",
                        "pause",
                        True,
                    )
                    paused = True
                    unpause_at = None

                    log.info(
                        "playback paused: translation lead %.1fs (target %.1fs)",
                        translated_lead,
                        lead,
                    )

            should_unpause = (
                unpause_at is not None
                and time.monotonic() >= unpause_at
            )

            if paused and unpause_at is None:
                with self._buffer_condition:
                    should_unpause = (
                        self._source_finished
                        or (
                            self._now is not None
                            and self._translated_until >= self._now + lead
                        )
                    )

            if paused and should_unpause:

                self.ipc.command(
                    "set_property",
                    "pause",
                    False,
                )

                paused = False
                unpause_at = None

                if (
                    self._anchor_reason == "seek"
                    and not self._audio_clock_diag_done
                ):
                    self._audio_clock_diag_until = time.monotonic() + 12.0
                    self._audio_clock_diag_done = True
                    log.info("MPV audio-clock diagnostics started after seek")

                log.info(
                    "mpv playing from %s",
                    fmt_hms(self._session_start),
                )

            if (pos := self.ipc.get("time-pos")) is not None:
                if self._mpv_anchor is None:
                    self._mpv_anchor = float(pos)
                    record_session_anchor(
                        self.ingest.source.timeline,
                        self._clock_source_anchor,
                        self._mpv_anchor,
                        self._anchor_reason,
                    )
                    log.info(
                        "MPV timeline anchor: source=%.3f mpv=%.3f "
                        "reason=%s duration=%s",
                        self._clock_source_anchor,
                        self._mpv_anchor,
                        self._anchor_reason,
                        fmt_hms(self.ingest.source.duration)
                        if self.ingest.source.duration is not None else "?",
                    )

                self._now = (
                    self._clock_source_anchor
                    + float(pos)
                    - self._mpv_anchor
                )
                if self.ingest.source.duration is not None:
                    self._now = min(self._now, self.ingest.source.duration)
                    self.ipc.command(
                        "script-message",
                        "auto-translate-seekbar-update",
                        f"{self._now:.3f}",
                        f"{self.ingest.source.duration:.3f}",
                    )

                sample_now = time.monotonic()
                if (
                    not self._clock_sample_logged
                    or sample_now - self._clock_last_sample_at >= 10.0
                ):
                    raw_delta = (
                        float(pos) - self._clock_last_sample_pos
                        if self._clock_last_sample_pos is not None
                        else 0.0
                    )
                    log.info(
                        "MPV clock sample: raw=%.3f mapped=%.3f "
                        "session_elapsed=%+.3fs raw_step=%+.3fs",
                        pos,
                        self._now,
                        self._now - self._session_start,
                        raw_delta,
                    )
                    self._clock_sample_logged = True
                    self._clock_last_sample_at = sample_now
                    self._clock_last_sample_pos = float(pos)

                self._render(self._now)

                sample_at = time.monotonic()
                if (
                    sample_at < self._audio_clock_diag_until
                    and sample_at - self._audio_clock_diag_last_at >= 0.5
                ):
                    audio_pts = self.ipc.get("audio-pts")
                    avsync = self.ipc.get("avsync")
                    mpv_paused = self.ipc.get("pause")
                    log.info(
                        "MPV audio clock: time_pos=%s audio_pts=%s "
                        "avsync=%s pause=%s",
                        pos,
                        audio_pts,
                        avsync,
                        mpv_paused,
                    )
                    self._audio_clock_diag_last_at = sample_at

            if self._replay_requested():
                if self._now is not None and self.ipc.get("eof-reached"):
                    log.info("replay requested after VOD end; restarting from beginning")
                    if self._jump(self.ingest.source.start_s - self._now):
                        return

            if (delta := self._jump_request()) and self._jump(delta):
                return

            time.sleep(0.1)

    def _recover_video_stream(self) -> bool:
        """Restart the shared fMP4 cache and audio reader at the current source position."""
        if self._video_retries >= 3:
            log.error("fMP4 cache recovery stopped after 3 attempts")
            self.stop.set()
            return False

        restart_at = self._now
        if restart_at is None:
            restart_at = self.ingest.source.start_s

        self._video_retries += 1
        log.warning(
            "restarting progressive fMP4 session (attempt %d/3) at %.2fs",
            self._video_retries,
            restart_at,
        )

        self.ipc.command("set_property", "pause", True)
        self._now = None
        self._shown = ""
        with self._lock:
            self._lines.clear()

        self._pending_seek_offset = None
        self.ingest.request_restart(restart_at)
        self.ingest.go()
        self._clock_source_anchor = restart_at
        self._mpv_anchor = None
        self._anchor_reason = "video-recovery"
        self._clock_sample_logged = False
        return True

    def _replay_requested(self) -> bool:
        """Return whether the user pressed Play/Space."""
        requested = False
        while not self.ipc.events.empty():
            ev = self.ipc.events.get()
            args = ev.get("args") or []
            if (
                ev.get("event") == "client-message"
                and args[:1] == ["auto-translate-replay"]
            ):
                requested = True
            elif (
                ev.get("event") == "client-message"
                and args[:1] == [JUMP_MESSAGE]
            ):
                # Preserve jump requests for the normal jump handler.
                self.ipc.events.put(ev)
                break
        return requested

    def _jump_request(self) -> float:
        """Collect jump-key requests from mpv."""
        delta = 0.0

        while not self.ipc.events.empty():
            ev = self.ipc.events.get()
            args = ev.get("args") or []

            if (
                ev.get("event") == "client-message"
                and args[:1] == [JUMP_MESSAGE]
            ):
                delta += float(args[1])

        return delta

    def _jump(self, delta: float) -> bool:
        """Restart ingest and mpv at the requested VOD position."""
        src = self.ingest.source

        if src.live:
            self.ipc.command(
                "show-text",
                "Jumping only works for VODs",
                2000,
            )
            return False

        if self._now is None:
            return False

        target = max(0.0, self._now + delta)

        if src.duration:
            target = min(
                target,
                max(0.0, src.duration - 5),
            )

        log.info(
            "jump %+.0fs → %s",
            delta,
            fmt_hms(target),
        )

        self.ipc.command("set_property", "pause", True)

        self._now = None
        self._shown = ""

        self.ipc.command(
            "osd-overlay",
            id=OVERLAY_ID,
            format="none",
            data="",
        )

        with self._lock:
            self._lines.clear()

        self._video_retries = 0
        cache_start = self.ingest.cache_source_start_s
        cache_end = self.ingest.cache_end_s
        can_reuse_cache = (
            cache_start <= target <= cache_end - 1.0
        )
        preserved = self.ingest.request_restart(
            target,
            preserve_video_cache=can_reuse_cache,
        )
        self._pending_seek_offset = (
            target - cache_start if preserved else None
        )
        if preserved:
            log.info(
                "seek will reuse local fMP4 fragments: %.2fs-%.2fs",
                cache_start,
                cache_end,
            )
        else:
            log.info(
                "seek requires a new progressive fMP4 fetch at %.2fs",
                target,
            )

        # The next playback session waits for the new fMP4 playlist and then
        # loads it. This avoids reconnecting MPV to a transient TCP stream.
        self.ingest.go()
        return True

    def _wait_for_event(
        self,
        name: str,
        timeout_s: float = 5.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            self.ipc.get("pause")

            while not self.ipc.events.empty():
                if self.ipc.events.get().get("event") == name:
                    return True

            time.sleep(0.05)

        return False

    def _render(self, now: float) -> None:
        with self._lock:
            self._lines = [
                ln
                for ln in self._lines
                if ln.show_until > now - 60
            ]

            active = [
                ln
                for ln in self._lines
                if ln.show_from <= now < ln.show_until and ln.sub.en
            ]

        active.sort(key=lambda ln: ln.show_from)

        selected: list[_Line] = []
        used_rows = 0

        for ln in reversed(active):
            # The limit counts subtitle entries, not individual language rows.
            # Each entry may contain Japanese above its English translation.
            # Each visible utterance occupies one English-only row.
            row_count = int(bool(ln.sub.en))

            if (
                selected
                and used_rows + row_count
                > max(1, self.cfg.max_subtitle_rows)
            ):
                break

            selected.append(ln)
            used_rows += row_count

        active = list(reversed(selected))

        text = (
            _ass_many(
                [ln.sub for ln in active],
                self.cfg,
            )
            if active
            else ""
        )

        # Log each subtitle only on its first visible scheduler tick.
        # Overlay changes can be caused by an older overlapping line, so
        # logging active[0] on every text change does not measure new lines.
        for ln in active:
            key = (
                ln.sub.generation,
                ln.sub.start_s,
                ln.sub.end_s,
            )
            if key in self._logged_displayed_subtitles:
                continue

            self._logged_displayed_subtitles.add(key)
            delta = now - ln.show_from
            self._lags.append(delta)
            log.info(
                "subtitle first visible: generation=%d speech=%.3f-%.3f "
                "mpv_clock=%.3f start_delta=%+.3fs active=%d",
                ln.sub.generation,
                ln.sub.start_s,
                ln.sub.end_s,
                now,
                delta,
                len(active),
            )

        if text != self._shown:
            self._shown = text

            self.ipc.command(
                "osd-overlay",
                id=OVERLAY_ID,
                format="ass-events" if text else "none",
                data=text,
                res_x=0,
                res_y=720,
                z=10,
                hidden=False,
            )


def _ass_many(
    subtitles: list[Subtitle],
    cfg: MpvConfig,
) -> str:
    """Render several active utterances as stacked mpv ASS overlay rows."""

    def esc(value: str) -> str:
        return (
            value
            .replace("\\", "＼")
            .replace("{", "｛")
            .replace("}", "｝")
            .replace("\n", " ")
        )

    blocks: list[str] = []

    for sub in subtitles:
        head = r"{\an2\bord2.5\3c&H000000&\4c&H80000000&}"
        rows: list[str] = []

        if sub.en:
            rows.append(
                rf"{{\fs{cfg.en_size}\1c&HFFFFFF&}}"
                + esc(sub.en)
            )

        if not rows:
            continue

        blocks.append(
            head + r"\N".join(rows)
        )

    return r"\N".join(blocks)


def _ass(sub: Subtitle, cfg: MpvConfig) -> str:
    """Backward-compatible single-subtitle renderer."""
    return _ass_many([sub], cfg)
