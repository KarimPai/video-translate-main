
"""Translation-synchronised MPV playback.

MPV plays the original media while FFmpeg reads audio independently.
VOD playback starts after a five-second translation lead, including after
seeks. Once started, playback is uninterrupted even if translation falls behind.
No deliberate video prebuffer.
For live streams, delay_s controls how far behind live playback starts.

A scheduler polls mpv's time-pos over JSON IPC (~10x/s) and draws subtitles
using osd-overlay.
"""
import ctypes
import json
import logging
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextvars import copy_context
from ctypes import wintypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import MpvConfig
from ..gui_control import EXIT_FULLSCREEN_MESSAGE, FULLSCREEN_MESSAGE, emit_event
from ..ingest import Ingest, fmt_hms
from ..output.coverage import CoverageTracker
from ..runtime_diagnostics import (
    increment_counter,
    record_event,
    rotate_log,
    update_metrics,
)
from ..session_cache import SessionSubtitleCache, same_source_cue
from ..subtitle_clean import sanitize_english_output
from ..subtitle_layout import chunk_at, display_until, split_cue, wrap_lines
from ..timeline import record_session_anchor
from ..types import Subtitle
from ..windows_subprocess import hidden_console_flags, owned_popen
from ..ytdlp import executable_path as ytdlp_executable_path

log = logging.getLogger(__name__)

IPC_PREFIX = (
    r"\\.\pipe\auto-translate-mpv"
    if sys.platform == "win32"
    else "/tmp/auto-translate-mpv"
)
WINDOWS_DEFAULT = Path(r"C:\Program Files\MPV Player\mpv.exe")
OVERLAY_ID = 1
INPUT_CONF = Path(__file__).with_name("mpv_input.conf")
GUI_INPUT_CONF = Path(__file__).with_name("mpv_gui_input.conf")

JUMP_MESSAGE = "auto-translate-jump"


def _create_kill_job(proc: subprocess.Popen) -> Any:
    """Automatically terminate MPV when the Python process exits."""
    if sys.platform != "win32":
        return None

    handle = getattr(proc, "_handle", None)
    if handle is None:
        return None
    if hasattr(handle, "_mock_name") or hasattr(handle, "mock_calls"):
        return None
    try:
        raw_handle = int(handle)
    except (TypeError, ValueError):
        return None
    if raw_handle <= 4:
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
        wintypes.HANDLE(raw_handle),
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


class MpvCommandTimeout(MpvError):
    """The command may have executed before the reply deadline."""


class _WindowsPipe:
    """Raw newline IPC over a duplex, overlapped Windows pipe.

    A blocking CRT file read can serialize writes on the same descriptor.
    Keep the reader independent so it can wait while commands are sent.
    """

    def __init__(self, path: str) -> None:
        import _winapi

        self._api = _winapi
        self._handle = _winapi.CreateFile(
            path,
            _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
            0, 0, _winapi.OPEN_EXISTING,
            _winapi.FILE_FLAG_OVERLAPPED, 0,
        )
        self._buffer = bytearray()

    def readline(self) -> bytes:
        api = self._api
        while b"\n" not in self._buffer:
            try:
                ov, _ = api.ReadFile(self._handle, 65536, overlapped=True)
                _, error = ov.GetOverlappedResult(True)
            except OSError as exc:
                if exc.winerror == api.ERROR_BROKEN_PIPE:
                    return b""
                raise
            if error:
                raise ctypes.WinError(error)
            data = ov.getbuffer()
            if not data:
                return b""
            self._buffer.extend(data)
        end = self._buffer.index(b"\n") + 1
        line = bytes(self._buffer[:end])
        del self._buffer[:end]
        return line

    def write(self, data: bytes) -> None:
        ov, _ = self._api.WriteFile(self._handle, data, overlapped=True)
        try:
            # Bound writes as well as response waits if MPV stops reading.
            if self._api.WaitForSingleObject(ov.event, 5000) != 0:
                raise TimeoutError("mpv IPC pipe write timed out")
        except BaseException:
            try:
                ov.cancel()
                ov.GetOverlappedResult(True)
            except OSError:
                pass
            raise
        written, error = ov.GetOverlappedResult(True)
        if error:
            raise ctypes.WinError(error)
        if written != len(data):
            raise ConnectionError("incomplete mpv IPC pipe write")

    def close(self) -> None:
        # Cancel a pending reader before releasing the handle.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.CancelIoEx.restype = wintypes.BOOL
        kernel32.CancelIoEx(self._handle, None)
        self._api.CloseHandle(self._handle)


class MpvIpc:
    """mpv JSON IPC with a dedicated reader and bounded command waits."""

    COMMAND_TIMEOUT_S = 5.0

    def __init__(self, path: str, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s

        while True:
            try:
                if sys.platform == "win32":
                    self._f = _WindowsPipe(path)
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
        self._properties = {}
        self._reader_failure = None
        self.events: queue.Queue[dict] = queue.Queue()
        self._responses: queue.Queue[dict] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_messages,
            name="mpv-ipc-reader",
            daemon=True,
        )
        self._reader.start()

    def _read_messages(self) -> None:
        """Read IPC continuously so command waits can have real timeouts."""
        try:
            while True:
                line = self._f.readline()
                if not line:
                    raise ConnectionError("mpv closed the IPC connection")

                msg = json.loads(line)
                if not isinstance(msg, dict):
                    continue
                if msg.get("event") == "property-change":
                    name = msg.get("name")
                    if isinstance(name, str) and name:
                        self._properties[name] = msg.get("data")
                elif "event" in msg:
                    if msg["event"] == "start-file":
                        # A new local playlist can reuse the old PTS origin.
                        # Do not expose the preceding file's cached position
                        # while waiting for its first fresh observation.
                        self._properties["time-pos"] = None
                        self._properties["eof-reached"] = None
                    self.events.put(msg)
                elif "request_id" in msg:
                    self._responses.put(msg)
        except Exception as exc:  # noqa: BLE001
            self._reader_failure = exc
            self._responses.put({"_reader_error": str(exc)})

    def command(self, *args: Any, **named: Any) -> Any:
        """Send one command and wait no longer than COMMAND_TIMEOUT_S."""
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            cmd = {"name": args[0], **named} if named else list(args)

            self._f.write(
                (
                    json.dumps({"command": cmd, "request_id": rid}) + "\n"
                ).encode()
            )

            deadline = time.monotonic() + self.COMMAND_TIMEOUT_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MpvCommandTimeout(
                        f"mpv IPC command timed out after "
                        f"{self.COMMAND_TIMEOUT_S:.1f}s: {cmd!r}"
                    )

                try:
                    msg = self._responses.get(timeout=remaining)
                except queue.Empty:
                    raise MpvCommandTimeout(
                        f"mpv IPC command timed out after "
                        f"{self.COMMAND_TIMEOUT_S:.1f}s: {cmd!r}"
                    ) from None

                if "_reader_error" in msg:
                    raise ConnectionError(
                        f"mpv IPC reader stopped: {msg['_reader_error']}"
                    )
                if msg.get("request_id") != rid:
                    continue
                if msg.get("error") != "success":
                    raise MpvError(f"{cmd!r}: {msg.get('error')}")
                return msg.get("data")

    def get(self, prop: str) -> Any:
        """Return a property's value, or None if unavailable."""
        try:
            if self._reader_failure is not None:
                raise ConnectionError("mpv IPC reader stopped") from self._reader_failure
            if prop in self._properties:
                return self._properties[prop]
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

    if sys.platform == "win32":
        marker = Path(__file__).resolve().parents[3] / ".cache" / "mpv-path.txt"
        try:
            installed = Path(marker.read_text(encoding="utf-8-sig").strip())
            if installed.is_absolute() and installed.name.lower() == "mpv.exe" and installed.is_file():
                return str(installed)
        except OSError:
            pass

    if found := shutil.which("mpv"):
        return found

    if sys.platform == "win32" and WINDOWS_DEFAULT.exists():
        return str(WINDOWS_DEFAULT)

    raise SystemExit(
        "mpv not found: install it (winget install shinchiro.mpv) "
        "or set [mpv] path"
    )


def _youtube_hook_options(cfg: MpvConfig, location: str) -> list[str]:
    if Path(location).exists():
        return []
    return [
        f"--script-opts=ytdl_hook-ytdl_path={ytdlp_executable_path()}",
        f"--ytdl-format={cfg.ytdlp_format}",
    ]


def _mpv_media_url(ingest: Ingest) -> str:
    """Share media segments in live mode; preserve original VOD playback."""
    if ingest.source.live and ingest.video_out:
        return ingest.video_playlist_url
    return ingest.source.location


@dataclass
class _LivePlaybackHealth:
    last_position: float | None = None
    progress_at: float | None = None
    healthy_since: float | None = None

    def sample(self, position, paused, now):
        if paused:
            self.progress_at = self.healthy_since = None
            self.last_position = None
            return None
        if self.progress_at is None:
            self.progress_at = now
        if position is not None and position != self.last_position:
            if (self.last_position is None or self.healthy_since is None
                    or now - self.progress_at > 5 or position < self.last_position
                    or position - self.last_position > now - self.progress_at + 5):
                self.healthy_since = now
            self.last_position = position
            self.progress_at = now
            if now - self.healthy_since >= 60:
                return "healthy"
        if now - self.progress_at >= 60:
            return "stalled"
        return None


class MpvOutput:
    # Minimum translated lead before starting/resuming after a seek.
    PAUSE_LEAD_SECONDS = 5.0
    LIVE_LATE_WINDOW_S = 7.0  # don't stack very old lines after a pipeline stall
    LIVE_LATE_DISPLAY_S = 2.0
    _seek_lead_target: float | None = None

    def __init__(
        self,
        cfg: MpvConfig,
        ingest: Ingest,
        stop: threading.Event,
        subtitle_cache: SessionSubtitleCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.ingest = ingest
        self.stop = stop
        self._subtitle_cache = subtitle_cache

        self._lines: list[_Line] = []
        self._subtitle_ready = threading.Event()
        self._lock = threading.Lock()
        self._now: float | None = None
        self._shown = ""
        self._session_start = ingest.source.start_s
        self._buffer_condition = threading.Condition()
        # Coverage tracker manages the contiguous translation watermark.
        self._coverage = CoverageTracker(ingest.source.start_s)
        self._source_finished = False
        # Map from speech interval ID to pending entry for reliable identity
        self._speech_id_map: dict[int, tuple[float, float]] = {}
        # Generation tracking: authoritative generation for callbacks.
        # Reset only through reset_buffer(). Stale callbacks are rejected.
        self._pending_generation = 0
        self._job = None
        self._closed = False
        self._gui_ready_emitted = False
        self._gui_commands: queue.Queue[tuple[str, float | None]] = queue.Queue(maxsize=16)
        self._last_gui_status_at = 0.0
        self._last_diagnostics_at = 0.0
        self._video_retries = 0
        self._mpv_anchor: float | None = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "initial"
        self._direct_video_loaded = False
        # The original VOD's mpv time-pos is the source timestamp. Do not
        # derive a new offset from the restarted FFmpeg audio reader on seek.
        # Local fMP4 fallback retains its separate anchored timeline.
        self._uses_original_timeline = False
        self._pending_seek: tuple[float, float, bool] | None = None
        self._native_seek_at: float | None = None
        self._last_live_mpv_pos: float | None = None
        self._last_live_sample_at: float | None = None
        self._clock_generation: int | None = None
        self._live_clock_discontinuity = False
        # Pause ownership: track who initiated the pause so auto-pause doesn't
        # become a permanent user-pause that blocks auto-resume.
        # "none" = playing, "user" = manual pause, "auto" = VOD buffer pause
        self._pause_owner: str = "none"
        self._manually_paused = False
        self._vod_auto_paused = False  # True when MPV is paused by the VOD sync gate
        # Seek-buffering: True while waiting for translation coverage after a seek.
        # Auto-resumes when sufficient contiguous coverage is available.
        self._seek_buffering = False
        # Recovery cooldown: after the pipeline watchdog resumes from auto-pause,
        # suppress auto-pause for a short window to prevent immediate re-trigger
        # when translation coverage has not yet caught up.
        self._recovery_cooldown_until: float = 0.0
        if not ingest.source.live:
            ingest.enable_vod_restart()
            ingest.enable_vod_playback_pacing(
                getattr(ingest.cfg, "vod_max_audio_lead_s", 75.0)
            )
            # Let Ingest relax its read-ahead cap when MPV is auto-paused
            # (not manually paused) so audio continues flowing for translation.
            set_cb = getattr(ingest, "set_auto_paused_callback", None)
            if callable(set_cb):
                set_cb(lambda: self._vod_auto_paused)

        mpv_log = (Path.cwd() / "logs" / "mpv.log")
        mpv_log.parent.mkdir(parents=True, exist_ok=True)
        rotate_log(mpv_log)

        ipc_path = f"{IPC_PREFIX}-{os.getpid()}-{uuid4().hex}"
        self._proc = owned_popen([
            find_mpv(cfg.path),

            "--pause",
            # MPV is a console-subsystem application on some Windows builds;
            # opening it under pythonw can still flash a separate CMD window.
            # Embedded GUI playback uses only MPV IPC and file logging.
            *([] if not cfg.window_id else ["--no-terminal"]),
            f"--input-ipc-server={ipc_path}",

            "--force-window=yes",
            *([f"--wid={cfg.window_id}"] if cfg.window_id else []),
            "--title=auto-translate",
            "--keep-open=yes",
            "--idle=yes",
            "--rebase-start-time=no",

            # Keep a modest forward/backward demux cache for the direct
            # YouTube video stream.
            "--cache=yes",
            "--demuxer-readahead-secs=120",
            f"--demuxer-max-bytes={cfg.live_forward_cache_mib}MiB" if ingest.source.live else "--demuxer-max-bytes=128MiB",
            f"--demuxer-max-back-bytes={cfg.live_back_cache_mib}MiB" if ingest.source.live else "--demuxer-max-back-bytes=32MiB",
            # Start near the HLS live edge, matching FFmpeg's normal -3.
            # Index 0 started MPV at an older segment than the ASR audio.
            *(
                [
                    "--demuxer-lavf-o=live_start_index=0"
                    if ingest.video_out else
                    "--demuxer-lavf-o=live_start_index=-3"
                ]
                if ingest.source.live else []
            ),
            "--input-default-bindings=no",

            # Let MPV's yt-dlp hook open the original YouTube page URL.
            # This preserves the full VOD timeline and lets MPV select both
            # video and audio tracks with the required HTTP headers.
            *_youtube_hook_options(cfg, ingest.source.location),

            # The embedded player already has a Qt timeline.
            "--osc=no" if cfg.window_id else "--osc=yes",
            "--osd-on-seek=no",
            *(["--osd-border-style=background-box"] if cfg.background_opacity > 0 else []),
            f"--input-conf={GUI_INPUT_CONF if cfg.window_id else INPUT_CONF}",

            # Hardware video decoding.
            f"--hwdec={cfg.hwdec}",

            f"--log-file={mpv_log}",
            "--msg-level=all=warn",
        ],
            # Suppress MPV console creation in GUI sessions, preserving the
            # embedded --wid render window and original CLI behavior.
            creationflags=hidden_console_flags(),
            stdin=subprocess.DEVNULL if cfg.window_id else None,
            stdout=subprocess.DEVNULL if cfg.window_id else None,
            stderr=subprocess.DEVNULL if cfg.window_id else None,
        )

        if sys.platform == "win32":
            try:
                self._job = _create_kill_job(self._proc)
            except OSError:
                log.exception(
                    "Could not enable automatic mpv process cleanup"
                )

        try:
            self.ipc = MpvIpc(ipc_path)

            for observer, prop in enumerate(("time-pos", "pause", "paused-for-cache", "eof-reached",
                                              "video-params", "video-codec", "hwdec-current"), start=1):
                self.ipc.command("observe_property", observer, prop)
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
                    self._proc.wait(timeout=2)

            _close_job(self._job)
            self._job = None
            raise

        log.info(
            "mpv ready for direct source video playback "
            "(live delay %.1fs, VOD translation lead %.1fs)",
            cfg.delay_s,
            self.PAUSE_LEAD_SECONDS,
        )

        self._thread = threading.Thread(
            target=copy_context().run,
            args=(self._run,),
            name="mpv",
            daemon=True,
        )
        self._thread.start()

    # ---- Output interface ----

    def is_manually_paused(self) -> bool:
        """Expose manual VOD pause to the pipeline watchdog."""
        return self._manually_paused

    def resume_from_auto_pause(self) -> None:
        """Break VOD auto-pause deadlock by resuming MPV playback.

        Called by the pipeline recovery watchdog when no subtitle has been
        published for 15+ seconds with pending work and auto-pause active.

        Sets a cooldown window (10s) to prevent immediate re-trigger of
        auto-pause when translation coverage has not yet caught up.
        """
        if self._vod_auto_paused and self._player_command("set_property", "pause", False):
            self._vod_auto_paused = False
            self._pause_owner = "none"
            # Cooldown prevents the scheduler loop from immediately re-pausing
            # when translation lead is still insufficient after recovery resume.
            self._recovery_cooldown_until = time.monotonic() + 10.0
            log.info("Resumed from auto-pause via recovery request (cooldown 10s)")

    def show(self, sub: Subtitle) -> None:
        if sub.end_s < self._session_start:
            return

        cleaned = sanitize_english_output(sub.en)
        if cleaned != sub.en:
            sub = replace(sub, en=cleaned)

        if not sub.en.strip():
            log.error(
                "subtitle not queued because English translation is empty "
                "(generation=%d, %.2f-%.2fs, Japanese=%r)",
                sub.generation,
                sub.start_s,
                sub.end_s,
                sub.ja,
            )
            return

        # Use the first speech-positive VAD frame, not the earlier padded
        # audio start used to preserve context for ASR.
        speech_start = (
            sub.speech_start_s
            if sub.speech_start_s is not None
            else sub.start_s
        )
        show_from = speech_start
        until = sub.end_s

        now = self._now
        if now is not None and now > until:
            late_by = now - until
            allowed_late = max(0.0, getattr(self.cfg, "live_late_window_s", self.LIVE_LATE_WINDOW_S))
            if self.ingest.source.live and late_by <= allowed_late:
                log.warning(
                    "displaying late live subtitle now: "
                    "generation=%d speech_end=%.3f mpv_clock=%.3f late=%.3fs",
                    sub.generation, sub.end_s, now, late_by,
                )
                show_from = now
                until = now + self.LIVE_LATE_DISPLAY_S
                # Record drift between actual speech time and display position.
                display_lag_s = now - (sub.speech_start_s if sub.speech_start_s is not None else sub.start_s)
                record_event("subtitle_display_lag", generation=sub.generation,
                             speech_start_s=sub.speech_start_s, pcm_start_s=sub.start_s,
                             display_at_s=now, lag_s=round(display_lag_s, 3), live=True)
            else:
                log.warning(
                    "dropping expired subtitle: generation=%d "
                    "speech_end=%.3f mpv_clock=%.3f late=%.3fs live=%s",
                    sub.generation, sub.end_s, now, late_by,
                    self.ingest.source.live,
                )
                increment_counter("expired")
                return

        line = _Line(sub, show_from, self._display_until(sub, show_from, until))
        with self._lock:
            if sub.generation != getattr(self.ingest, "generation", sub.generation):
                increment_counter("dropped")
                return
            self._lines = [
                existing for existing in self._lines
                if not (existing.sub.generation == sub.generation
                        and same_source_cue(
                            existing.sub, sub, SessionSubtitleCache.REPLACEMENT_TOLERANCE_S,
                        ))
            ]
            self._lines.append(line)
            pending = sum(ln.show_from > (self._now if self._now is not None else -1) for ln in self._lines)
        update_metrics(counters={"pending_subtitles": pending})
        record_event("subtitle_enqueued", generation=sub.generation,
                     speech_start_s=speech_start, speech_end_s=sub.end_s,
                     pcm_start_s=sub.start_s, asr_completed_at=sub.timings.asr_end,
                     translation_completed_at=sub.timings.mt_end,
                     show_from_s=show_from, show_until_s=line.show_until)
        self._subtitle_ready.set()

    def advance(self, progress_s: float) -> None:
        """Record the latest VAD progress position.

        This advances VAD progress in the coverage tracker. The translation
        watermark advances only when subtitles are published or when silence
        is contiguous with the current watermark position.
        """
        with self._buffer_condition:
            self._coverage.advance_vad(progress_s)
            translated_until = self._coverage.translated_until
            if self._seek_lead_target is not None and translated_until >= self._seek_lead_target:
                log.info("translation lead ready at %.3fs", self._seek_lead_target)
                self._seek_lead_target = None
                self._subtitle_ready.set()
            self._buffer_condition.notify_all()

    def record_speech(self, start_s: float, end_s: float, generation: int) -> int | None:
        """Record a speech region that VAD has processed but not yet translated.

        Returns the speech interval ID, or None if the callback is stale.
        Stale callbacks (wrong generation) are rejected without modifying state.
        The generation is stored with the interval so stale-generation speech
        can be safely abandoned after a seek.
        """
        with self._buffer_condition:
            if generation != self._pending_generation:
                # Stale callback — reject without modifying state
                return None
            sid = self._coverage.record_speech(start_s, end_s, generation=generation)
            self._speech_id_map[sid] = (start_s, end_s)
            return sid

    def advance_to_subtitle(self, subtitle_end_s: float, generation: int, speech_id: int | None = None,
                             subtitle_start_s: float | None = None) -> None:
        """Advance the watermark to cover a published subtitle.

        When a subtitle is published, the watermark can advance to its end_s,
        including any contiguous VAD-confirmed silence. Will NOT skip over
        other pending speech.

        If `speech_id` is provided, uses it for reliable identity matching.
        Otherwise falls back to interval matching by timestamps.

        Stale callbacks (wrong generation) are rejected without modifying state.
        """
        with self._buffer_condition:
            if generation != self._pending_generation:
                # Stale callback — reject without modifying state
                return
            # Complete the speech interval using reliable identity
            if speech_id is not None:
                completed = self._coverage.complete_speech(speech_id)
                self._speech_id_map.pop(speech_id, None)
            elif subtitle_start_s is not None:
                # Fallback: match by actual subtitle interval timestamps.
                # Using the real start/end is much safer than guessing
                # `subtitle_end_s - 2.0`.
                completed = self._coverage.complete_by_interval(
                    subtitle_start_s, subtitle_end_s, tolerance=0.5
                )
            else:
                # No speech_id and no start_s available — do NOT guess.
                # Completing the wrong pending entry would corrupt coverage.
                log.debug(
                    "advance_to_subtitle: no speech_id or start_s, skipping completion at end_s=%.3f",
                    subtitle_end_s,
                )
                completed = False
            if not completed:
                log.debug(
                    "advance_to_subtitle: no matching pending speech at end_s=%.3f",
                    subtitle_end_s,
                )
            if self._seek_lead_target is not None and self._coverage.translated_until >= self._seek_lead_target:
                log.info("translation lead ready at %.3fs", self._seek_lead_target)
                self._seek_lead_target = None
                self._subtitle_ready.set()
            self._buffer_condition.notify_all()

    def complete_speech(self, start_s: float, end_s: float, generation: int, speech_id: int | None = None) -> None:
        """Clear pending speech and advance the watermark for a suppressed translation.

        When a translation is suppressed (empty output, prompt echo, untranslated
        Japanese), no subtitle is published, but the pending speech entry must
        still be cleared so the watermark can advance.

        If `speech_id` is provided, uses it for reliable identity matching.
        Otherwise falls back to interval matching.

        Stale callbacks (wrong generation) are rejected without modifying state.
        """
        with self._buffer_condition:
            if generation != self._pending_generation:
                # Stale callback — reject without modifying state
                return
            if speech_id is not None:
                completed = self._coverage.complete_speech(speech_id)
                self._speech_id_map.pop(speech_id, None)
            else:
                completed = self._coverage.complete_by_interval(start_s, end_s, tolerance=0.5)
            if not completed:
                log.debug(
                    "complete_speech: no matching pending speech (%.3f-%.3f)",
                    start_s, end_s,
                )
            if self._seek_lead_target is not None and self._coverage.translated_until >= self._seek_lead_target:
                log.info("translation lead ready at %.3fs (after suppressed translation)", self._seek_lead_target)
                self._seek_lead_target = None
                self._subtitle_ready.set()
            self._buffer_condition.notify_all()

    def playback_position(self):
        """Last sampled position in the subtitle source timeline."""
        return self._now

    def _translation_ready(self) -> bool:
        """Start after a 5s lead from new work or previously cached audio."""
        with self._buffer_condition:
            translated_until = self._coverage.translated_until
            ready = self._source_finished or (
                self._now is not None
                and translated_until >= self._now + self.PAUSE_LEAD_SECONDS
            )
        if ready:
            return True
        cache = getattr(self, "_subtitle_cache", None)
        now = self._now
        return (
            cache is not None and now is not None
            and cache.covered_until(now) >= now + self.PAUSE_LEAD_SECONDS
        )

    def _vod_should_pause(self) -> bool:
        """Check if MPV should pause due to low translation lead (VOD only).

        Returns True when the translation lead falls below the pause threshold
        and the user has not manually paused.
        """
        if self.ingest.source.live:
            return False
        if self._source_finished:
            return False
        now = self._now
        if now is None:
            return False
        translated_until = self._coverage.translated_until
        lead = translated_until - now
        pause_threshold = getattr(self.cfg, "vod_pause_threshold_s", 2.0)
        return lead < pause_threshold

    def _vod_should_resume(self) -> bool:
        """Check if MPV should resume after auto-pause (VOD only).

        Returns True when the translation lead exceeds the resume threshold
        (hysteresis prevents oscillation).
        """
        if self.ingest.source.live:
            return False
        now = self._now
        if now is None:
            return False
        translated_until = self._coverage.translated_until
        lead = translated_until - now
        resume_threshold = getattr(self.cfg, "vod_resume_threshold_s", 8.0)
        return lead >= resume_threshold

    def reset_buffer(self, progress_s: float) -> None:
        """Reset translated coverage after a VOD seek.

        Clears all pending speech, resets the watermark, and prepares for
        fresh translation coverage at the new seek position. The seek-buffering
        state is managed by _restart_audio_at() which calls this method.

        Synchronizes _pending_generation with the current ingest.generation
        so that valid callbacks are not rejected after seeking.
        Also abandons any pending speech from the previous generation that
        survived the reset (should be none, but defensive).

        Idempotent per generation: _restart_audio_at() calls reset_buffer()
        immediately, then ingest.py emits a reset AudioProgress marker that
        causes pipeline.py to call reset_buffer() again. The second call must
        NOT wipe valid pending speech registered between the two resets.
        We track the generation we already reset for and skip the destructive
        portion on a duplicate call for the same generation.
        """
        with self._buffer_condition:
            new_gen = self.ingest.generation
            # Idempotency guard: if we already reset for this generation,
            # only update the seek lead target (in case progress_s advanced)
            # and skip the destructive reset that would wipe valid pending speech.
            if new_gen == self._pending_generation:
                # Already reset for this generation — update lead target only
                if self._seek_lead_target is None or progress_s + self.PAUSE_LEAD_SECONDS > self._seek_lead_target:
                    self._seek_lead_target = progress_s + self.PAUSE_LEAD_SECONDS
                log.debug("reset_buffer: already reset for generation=%d, skipping duplicate", new_gen)
                self._buffer_condition.notify_all()
                return
            self._seek_lead_target = progress_s + self.PAUSE_LEAD_SECONDS
            self._coverage.reset(progress_s)
            self._source_finished = False
            self._speech_id_map.clear()
            # Synchronize generation with the current ingest generation.
            # request_restart() increments ingest._generation before calling
            # reset_buffer(), so the new generation is already available.
            self._pending_generation = new_gen
            self._coverage.set_generation(new_gen)
            # Defensive: abandon any stale-generation pending speech
            stale = self._coverage.abandon_stale_generation()
            if stale > 0:
                log.info("Abandoned %d stale-generation pending speech intervals on seek", stale)
            # Reset pause state after seek — _restart_audio_at() will set
            # the appropriate state (seek-buffering or manual pause preserved).
            self._vod_auto_paused = False
            self._buffer_condition.notify_all()

    def abandon_stale_pending_speech(self, cutoff_s: float) -> int:
        """Abandon pending speech intervals that are too far behind playback.

        Called periodically by the pipeline watchdog to prevent unbounded
        pending speech accumulation from blocking the watermark.
        Returns the number of intervals abandoned.
        """
        with self._buffer_condition:
            abandoned = self._coverage.abandon_pending_before(cutoff_s)
            # Also clean up the speech_id_map for abandoned intervals
            if abandoned > 0:
                # Remove IDs whose intervals were before the cutoff
                self._speech_id_map = {
                    sid: interval for sid, interval in self._speech_id_map.items()
                    if interval[0] >= cutoff_s
                }
                self._buffer_condition.notify_all()
            return abandoned

    def finish(self, generation: int | None = None) -> None:
        with self._buffer_condition:
            if generation is not None and generation != self.ingest.generation:
                return
            self._source_finished = True
            self._coverage.mark_source_finished()
            self._buffer_condition.notify_all()

    def drain(self) -> None:
        """Keep MPV open until closed, tolerating a player IPC disconnect."""
        while not self.stop.is_set():
            if self._proc.poll() is not None:
                return
            try:
                reached_eof = self.ipc.get("eof-reached")
            except (ConnectionError, OSError, MpvError) as exc:
                # The MPV process can exit between poll() and the IPC write.
                # BrokenPipeError [WinError 232] is expected during shutdown,
                # not an audio/transcription failure.
                log.warning("MPV IPC closed during drain; exiting cleanly: %s", exc)
                return
            if reached_eof:
                log.info(
                    "VOD playback finished; MPV will remain open until closed"
                )
                while not self.stop.is_set() and self._proc.poll() is None:
                    time.sleep(0.25)
                return

            time.sleep(0.25)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Cleanly stop MPV and release its automatic cleanup protection."""
        if self._closed:
            return

        self._closed = True
        self.stop.set()

        with self._buffer_condition:
            self._buffer_condition.notify_all()

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

    def _player_command(self, *args, **kwargs) -> bool:
        """A rejected command or late reply does not close the IPC connection."""
        try:
            self.ipc.command(*args, **kwargs)
            return True
        except MpvError as exc:
            log.warning("MPV %s was not confirmed: %s", args[0], exc)
            return False

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
        ) as exc:
            if not self.stop.is_set():
                # WinError 232 means MPV's IPC pipe closed. Distinguish a
                # normal window close from an unexpected player exit rather
                # than reporting the pipe failure itself as the root cause.
                returncode = self._proc.poll()
                if returncode is None:
                    # Windows can close the named pipe a moment before the
                    # process handle reports exit code 0. Give normal quits a
                    # brief chance to finish before treating them as failures.
                    try:
                        returncode = self._proc.wait(timeout=0.75)
                    except subprocess.TimeoutExpired:
                        pass
                if returncode == 0:
                    log.info("MPV window closed normally; stopping translator")
                elif returncode is not None:
                    log.error(
                        "MPV exited with code %s; IPC closed (%s). "
                        "Check logs/mpv.log for the playback error.",
                        returncode, exc,
                    )
                else:
                    log.warning(
                        "MPV IPC disconnected (%s); stopping translator "
                        "and checking final process state", exc
                    )
                self.stop.set()

        except Exception as exc:
            if not self.stop.is_set():
                self.scheduler_failure = exc
                log.exception("MPV scheduler crashed")
                self.stop.set()

    def _play_session(self, show_hint: bool) -> None:
        """Buffer before start/seeks; never stutter-pause during playback."""

        lead = self.PAUSE_LEAD_SECONDS

        # Load the video immediately; audio timestamp readiness must not
        # prevent MPV from opening the source. The audio pipeline can fail or
        # be delayed independently of video playback.
        if not self._direct_video_loaded:
            self._session_start = self.ingest.source.start_s
            shared_live_hls = bool(
                self.ingest.source.live and self.ingest.video_out
            )
            if shared_live_hls:
                # The ingest worker starts the shared fMP4 writer. MPV must
                # open its existing playlist, not a separately resolved YouTube
                # live rendition that may begin at another HLS segment.
                if not self._wait_live_ready(self.ingest.video_ready, "playlist", 155.0):
                    return
                log.info("MPV live: loading same local HLS playlist as ASR")
            video_url = _mpv_media_url(self.ingest)
            if not shared_live_hls:
                log.info(
                    "loading original source in MPV through its yt-dlp hook"
                    if not Path(video_url).exists()
                    else "loading local source in MPV"
                )
            remote_vod = not self.ingest.source.live and video_url.startswith(("https://", "http://"))
            attempts = 2 if remote_vod else 1
            loaded = False
            for attempt in range(attempts):
                try:
                    self.ipc.command("loadfile", video_url, "replace")
                    loaded = (
                        self._wait_for_event("start-file", timeout_s=10.0)
                        and self._wait_for_event("file-loaded", timeout_s=30.0)
                    )
                    if loaded and remote_vod:
                        tracks = self.ipc.get("track-list")
                        if isinstance(tracks, list) and not any(
                            track.get("type") == "video" and not track.get("albumart")
                            for track in tracks
                        ):
                            log.warning("YouTube VOD loaded without a video track")
                            loaded = False
                except (MpvError, ConnectionError, OSError):
                    log.exception("MPV rejected the original source")
                if loaded or self.stop.is_set():
                    break
                if attempt + 1 < attempts:
                    log.warning("Retrying YouTube VOD load with fresh media URLs")

            if not loaded and shared_live_hls:
                log.error(
                    "MPV could not load shared live HLS; refusing an "
                    "unsynchronized original-source fallback"
                )
                self.stop.set()
                return

            if not loaded and not self.ingest.video_out:
                log.error("MPV could not load the source; see logs/mpv.log")
                self.stop.set()
                return

            if not loaded:
                log.warning(
                    "MPV could not load the original source; "
                    "falling back to the translator's local fMP4 video stream"
                )
                try:
                    cache_url = self.ingest.video_playlist_url
                    self.ipc.command("loadfile", cache_url, "replace")
                    cache_started = self._wait_for_event(
                        "start-file",
                        timeout_s=10.0,
                    )
                    cache_loaded = (
                        cache_started
                        and self._wait_for_event("file-loaded", timeout_s=30.0)
                    )
                except (MpvError, ConnectionError, OSError, RuntimeError):
                    log.exception("MPV could not start the local fMP4 video fallback")
                    self.stop.set()
                    return

                if not cache_loaded:
                    log.error("MPV did not finish loading the local fMP4 video fallback")
                    self.stop.set()
                    return

                log.info(
                    "MPV loaded local fMP4 fallback; seek range follows "
                    "the currently cached video"
                )

            self._uses_original_timeline = loaded and not self.ingest.source.live
            self._direct_video_loaded = True
            # The GUI must not hide URL controls until MPV has actually
            # loaded a video and the running pipeline owns its output.
            if self.cfg.window_id and not self._gui_ready_emitted:
                from ..gui_runner import READY_MARKER
                print(READY_MARKER, flush=True)
                emit_event("ready", live=bool(self.ingest.source.live),
                           duration=self.ingest.source.duration)
                self._gui_ready_emitted = True
            if self._session_start > 0 and not self.ingest.source.live:
                self._player_command("seek", self._session_start, "absolute+exact")
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    pos = self.ipc.get("time-pos")
                    if pos is not None and abs(float(pos) - self._session_start) <= 1.0:
                        break
                    time.sleep(0.05)
                else:
                    if pos is None:
                        raise RuntimeError("MPV could not confirm the initial playback position")
                    self._restart_audio_at(float(pos))
                    return

        if self.ingest.source.live:
            if not self._wait_live_ready(self.ingest.first_pts_ready, "decoded PCM", 60.0):
                return
            if not self._wait_live_ready(self.ingest.audio_media_pts_ready, "media timestamps", 15.0):
                return
            self._session_start = self.ingest.source.start_s

        # MPV plays the original VOD timeline; FFmpeg independently extracts
        # audio for ASR. Anchor each audio generation to MPV's current position.
        self._mpv_anchor = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "seek" if self.ingest.generation > 0 else "initial"
        self._last_live_mpv_pos = None
        self._last_live_sample_at = None
        self._clock_generation = None
        self._live_clock_discontinuity = False
        current_pos = self.ipc.get("time-pos")
        # An HLS live clock may initially report 0 (or a placeholder) and
        # jump to the playlist's multi-hour media PTS after file-loaded.
        # Do not anchor live captions to that unstable startup sample.
        if current_pos is not None and not self.ingest.source.live:
            self._mpv_anchor = float(current_pos)
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
        live_health = _LivePlaybackHealth()
        # Allow sufficient time for model loading and first-utterance translation.
        # A cold-start model may take 10-30s to load; 3x the lead provides margin
        # without blocking the user indefinitely (they can manually unpause).
        vod_wait_until = time.monotonic() + lead * 3

        while not self.stop.is_set():

            self._subtitle_ready.clear()
            if self.ingest.video_failed.is_set():
                recovered = self._recover_video_stream()
                if not recovered:
                    # Recovery gave up (max retries) — stop the scheduler
                    return
                # Recovery succeeded: return from _play_session() so _run()
                # calls it again, which re-enters the video loading block
                # (guarded by _direct_video_loaded) and loads the new fMP4
                # playlist for livestreams or re-opens the source for VOD.
                return

            should_unpause = (
                unpause_at is not None
                and time.monotonic() >= unpause_at
            )

            if paused and unpause_at is None:
                should_unpause = self._translation_ready()
                if not should_unpause and time.monotonic() >= vod_wait_until:
                    log.info("VOD translation wait reached %.1fs; resuming video", lead)
                    should_unpause = True

            # User input overrides the optional translation prebuffer. Never
            # force MPV back to pause immediately after its Play button is used.
            # Once manually playing, do not auto-pause for low translation lead.
            mpv_paused = self.ipc.get("pause")
            if paused and mpv_paused is False:
                paused = False
                unpause_at = None
                log.info(
                    "MPV manually started; continuing playback without "
                    "waiting for the translation buffer"
                )

            if (paused and should_unpause and self._pause_owner != "user"
                    and self._player_command("set_property", "pause", False)):

                paused = False
                mpv_paused = False
                unpause_at = None

                log.info(
                    "mpv playing from %s",
                    fmt_hms(self._session_start),
                )

            # Update manual pause tracking: only reflect user-initiated pauses.
            # Never infer user pause from MPV's raw pause property — that would
            # incorrectly treat auto-pauses as user pauses and block auto-resume.
            if self._pause_owner == "user":
                self._manually_paused = mpv_paused is True
            elif self._pause_owner == "auto":
                self._manually_paused = False
            else:
                self._manually_paused = False

            # Seek-buffering auto-resume: after a seek, resume when sufficient
            # contiguous translated coverage is available. Use a lower threshold
            # (5s) for seek-resume vs steady-state recovery (8s).
            if self._seek_buffering and self._now is not None:
                tu = self._coverage.translated_until
                seek_lead = tu - self._now
                seek_resume_threshold = getattr(self.cfg, "vod_seek_resume_threshold_s", self.PAUSE_LEAD_SECONDS)
                if seek_lead >= seek_resume_threshold and self._player_command("set_property", "pause", False):
                    self._seek_buffering = False
                    self._pause_owner = "none"
                    self._vod_auto_paused = False
                    log.info(
                        "Seek-buffering resume: translation lead %.1fs >= %.1fs threshold "
                        "(now=%.1fs translated_until=%.1fs)",
                        seek_lead, seek_resume_threshold,
                        self._now, tu,
                    )

            # Adaptive VOD sync: pause when translation lead is too low,
            # resume when it recovers (hysteresis prevents oscillation).
            # User manual pause/play controls remain authoritative.
            # After recovery resume, suppress auto-pause for a cooldown window
            # to prevent immediate re-trigger when translation has not caught up.
            in_recovery_cooldown = time.monotonic() < self._recovery_cooldown_until
            if not paused and not self.ingest.source.live and self._pause_owner != "user" \
                    and not self._seek_buffering and not in_recovery_cooldown:
                if self._vod_should_pause() and mpv_paused is not True:
                    # Translation falling behind — auto-pause MPV
                    if self._player_command("set_property", "pause", True):
                        self._vod_auto_paused = True
                        self._pause_owner = "auto"
                        tu = self._coverage.translated_until
                        lead = tu - self._now if self._now else 0
                        pending = self._coverage.pending_count()
                        log.info(
                            "VOD auto-pause: translation lead %.1fs < %.1fs threshold "
                            "(now=%.1fs translated_until=%.1fs pending_speech=%d)",
                            lead, getattr(self.cfg, "vod_pause_threshold_s", 2.0),
                            self._now if self._now else -1,
                            tu, pending,
                        )
                elif self._vod_auto_paused and self._vod_should_resume() and self._player_command("set_property", "pause", False):
                    # Translation caught up — resume MPV
                    self._vod_auto_paused = False
                    self._pause_owner = "none"
                    tu = self._coverage.translated_until
                    lead = tu - self._now if self._now else 0
                    log.info(
                        "VOD auto-resume: translation lead %.1fs >= %.1fs threshold "
                        "(now=%.1fs translated_until=%.1fs)",
                        lead, getattr(self.cfg, "vod_resume_threshold_s", 8.0),
                        self._now if self._now else -1,
                        tu,
                    )

            replay, delta = self._read_player_events()
            if replay and self._now is not None and self.ipc.get("eof-reached"):
                self._jump(-self._now)
            if delta:
                self._jump(delta)
            pos = self.ipc.get("time-pos")
            if not self.ingest.source.live and self._settle_seek(pos):
                # Seek confirmed and audio restarted — reset the translation
                # wait timeout so the video buffers briefly before resuming.
                vod_wait_until = time.monotonic() + lead
            if self.ingest.source.live and self.ingest.video_out:
                health = live_health.sample(pos, paused or mpv_paused is True, time.monotonic())
                if health == "stalled":
                    log.error("Live MPV clock stalled for 60s; restarting shared session")
                    recovered = self._recover_video_stream()
                    if not recovered:
                        return
                    # Return so _run() restarts _play_session() and reloads
                    # the new fMP4 playlist with fresh clock anchors.
                    return
                if health == "healthy" and self._video_retries:
                    log.info("Live playback healthy for 60s; resetting recovery attempts")
                    self._video_retries = 0
            if pos is not None:
                if self.ingest.source.live:
                    aligned = self._sample_live_clock(float(pos), time.monotonic())
                    if self._live_clock_discontinuity:
                        recovered = self._recover_video_stream()
                        if not recovered:
                            return
                        # Return so _run() restarts _play_session() and reloads
                        # the new fMP4 playlist with fresh clock anchors.
                        return
                else:
                    aligned = True
                if self._mpv_anchor is None and not self.ingest.source.live:
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

                if aligned:
                    self._now = self._source_time_from_mpv(float(pos))
                    if not self.ingest.source.live:
                        self.ingest.set_playback_position(self._now)
                    if self.ingest.source.duration is not None:
                        self._now = min(self._now, self.ingest.source.duration)

                    # A paused startup sample is not audible speech. Preserve
                    # an existing overlay on manual pause without drawing a
                    # newly completed cue until playback resumes.
                    cache_paused = getattr(self.ipc, "_properties", {}).get("paused-for-cache") is True
                    if not self.ingest.source.live or not (paused or mpv_paused is True or cache_paused):
                        self._render(self._now)

            if getattr(self.cfg, "window_id", None):
                self._emit_gui_status(mpv_paused is True)
                self._handle_gui_commands()

            # Wake for newly translated text or the next queued cue boundary.
            wait_s = 0.1
            if not paused and self._now is not None:
                with self._lock:
                    for line in self._lines:
                        for boundary in (line.show_from, line.show_until):
                            if self._now < boundary < self._now + wait_s:
                                wait_s = boundary - self._now
            self._subtitle_ready.wait(max(0.005, wait_s))

    def _wait_live_ready(self, event, name, timeout):
        deadline = time.monotonic() + timeout
        while not self.stop.is_set():
            if event.wait(0.2):
                return True
            if self.ingest.video_failed.is_set() or time.monotonic() >= deadline:
                log.warning("Shared live %s unavailable, requesting recovery", name)
                self._recover_video_stream()
                break
        return False

    def _recover_video_stream(self) -> bool:
        """Restart the shared fMP4 cache and audio reader at the current source position."""
        if self._video_retries >= 3:
            self.scheduler_failure = RuntimeError("fMP4 cache recovery stopped after 3 attempts")
            log.error("%s", self.scheduler_failure)
            self.stop.set()
            return False

        restart_at = self._now
        if restart_at is None:
            restart_at = self.ingest.source.start_s

        self._video_retries += 1
        record_event("playback_recovery", attempt=self._video_retries, source_s=restart_at)
        log.warning(
            "restarting progressive fMP4 session (attempt %d/3) at %.2fs",
            self._video_retries,
            restart_at,
        )

        self._player_command("set_property", "pause", True)
        self._player_command("osd-overlay", id=OVERLAY_ID, format="none", data="")
        self._now = None
        self._shown = ""
        with self._lock:
            self._lines.clear()

        self.ingest.request_restart(restart_at)
        if self.ingest.source.live and self.ingest.video_out:
            # Reopen the fresh shared playlist, never continue playing the
            # disconnected HLS session while audio restarts independently.
            self._direct_video_loaded = False
        self.reset_buffer(restart_at)
        self.ingest.go()
        self._session_start = restart_at
        self._clock_source_anchor = restart_at
        self._mpv_anchor = None
        self._anchor_reason = "video-recovery"
        update_metrics(playback={"live_clock_aligned": False,
                                 "timeline_generation": self.ingest.generation,
                                 "source_to_mpv_offset_s": None})
        return True

    def queue_gui_command(self, action: str, value: float | None) -> None:
        """Accept a parsed GUI command without touching MPV from the stdin thread."""
        if not self.cfg.window_id:
            return
        if action == "quit":
            log.info("Authenticated GUI shutdown requested")
            self.stop.set()
            self._subtitle_ready.set()
            return
        log.info("GUI player command queued: %s value=%s", action, value)
        try:
            self._gui_commands.put_nowait((action, value))
        except queue.Full:
            # Keep the latest seek position if the user drags repeatedly.
            try:
                self._gui_commands.get_nowait()
            except queue.Empty:
                pass
            self._gui_commands.put_nowait((action, value))
        self._subtitle_ready.set()

    def _emit_gui_status(self, paused: bool) -> None:
        """Publish the source timeline, not raw live-HLS MPV timestamps."""
        now = time.monotonic()
        if now - self._last_gui_status_at < 0.5:
            return
        self._last_gui_status_at = now
        if now - getattr(self, "_last_diagnostics_at", 0.) >= 2.:
            self._last_diagnostics_at = now
            # These are observer-fed cache reads; monitoring adds no blocking
            # MPV requests or NVIDIA calls to the scheduling loop.
            properties = getattr(self.ipc, "_properties", {})
            params = properties.get("video-params")
            params = params if isinstance(params, dict) else {}
            width, height = params.get("w"), params.get("h")
            hwdec = properties.get("hwdec-current")
            update_metrics(playback={
                "resolution": f"{width}x{height}" if width and height else None,
                "decoder": properties.get("video-codec"), "hwdec": hwdec,
                "hardware_acceleration": None if hwdec is None else hwdec not in ("no", ""),
            })
        live = bool(self.ingest.source.live)
        duration = None if live else self.ingest.source.duration
        if duration is None and not live:
            duration = self.ipc.get("duration")
        emit_event(
            "status", live=live,
            position=round(self._now, 2) if self._now is not None else None,
            duration=duration,
            paused=paused,
        )

    def _handle_gui_commands(self) -> None:
        """Apply GUI commands on MPV's scheduler, preserving ASR seek resets.

        An earlier implementation silently dropped seeks while the MPV
        position was momentarily unavailable during buffering or a restart.
        Retain the latest request until the source clock becomes available.
        """
        while not self._gui_commands.empty():
            try:
                action, value = self._gui_commands.get_nowait()
            except queue.Empty:
                break
            if action == "pause":
                self._toggle_user_pause()
                continue
            if self.ingest.source.live:
                log.info("GUI seek rejected: livestream")
                emit_event("seek_ack", accepted=False, target=value)
                continue
            if value is None:
                emit_event("seek_ack", accepted=False, target=None)
                continue
            base = self._seek_base()
            if base is None:
                # Queue again and wait for the video clock to become available.
                # Return now to avoid spinning through the same request forever.
                self._gui_commands.put_nowait((action, value))
                return
            delta = value - base if action == "seek" else value
            target = max(0.0, base + delta)
            if self.ingest.source.duration:
                target = min(target, max(0.0, self.ingest.source.duration - 5))
            if target == base and self._pending_seek is None:
                emit_event("seek_ack", accepted=True, target=target)
                continue
            log.info("GUI seek command: %s %+.2fs -> %.2fs", action, delta, target)
            self._jump(delta, gui=True)

    def _toggle_user_pause(self):
        """Toggle user-initiated pause/play via GUI or keyboard.
        
        Sets _pause_owner to "user" when pausing, "none" when playing.
        This distinguishes user intent from automatic VOD buffer pauses.
        """
        paused = self.ipc.get("pause")
        if isinstance(paused, bool):
            new_paused = not paused
            if self._player_command("set_property", "pause", new_paused):
                if new_paused:
                    self._pause_owner = "user"
                    self._vod_auto_paused = False
                else:
                    self._pause_owner = "none"
                    self._vod_auto_paused = False
                    self._seek_buffering = False
            log.debug("User pause owner=%s", self._pause_owner)

    def _read_player_events(self) -> tuple[bool, float]:
        """Consume each IPC event once, including events arriving during a seek."""
        requested = False
        delta = 0.0
        while True:
            try:
                ev = self.ipc.events.get_nowait()
            except queue.Empty:
                break
            args = ev.get("args") or []
            if ev.get("event") == "end-file" and ev.get("reason") == "error":
                record_event("video_decoder_error", error=ev.get("file_error"))
            if ev.get("event") == "seek" and not self.ingest.source.live:
                self._native_seek_at = time.monotonic()
            if (
                ev.get("event") == "client-message"
                and args[:1] == ["auto-translate-replay"]
            ):
                requested = True
            elif ev.get("event") == "client-message" and args[:1] == ["auto-translate-gui-pause"]:
                self._toggle_user_pause()
                requested = True
            elif getattr(self.cfg, "window_id", None) and ev.get("event") == "client-message" and args[:1] == [FULLSCREEN_MESSAGE]:
                emit_event("toggle_fullscreen")
            elif getattr(self.cfg, "window_id", None) and ev.get("event") == "client-message" and args[:1] == [EXIT_FULLSCREEN_MESSAGE]:
                emit_event("exit_fullscreen")
            elif (
                ev.get("event") == "client-message"
                and args[:1] == [JUMP_MESSAGE]
            ):
                if len(args) < 2:
                    continue
                try:
                    jump = float(args[1])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(delta + jump):
                    delta += jump
        return requested, delta

    def _jump(self, delta: float, *, gui: bool = False) -> None:
        """Submit a video seek. Audio follows the position MPV actually reaches."""
        src = self.ingest.source

        if src.live:
            self._player_command(
                "show-text",
                "Jumping only works for VODs",
                2000,
            )
            return

        base = self._seek_base()
        if base is None:
            return

        target = max(0.0, base + delta)

        if src.duration:
            target = min(target, max(0.0, src.duration - 5))

        log.info("jump %+.0fs -> %s", delta, fmt_hms(target))
        mpv_target = self._mpv_time_from_source(target)
        try:
            self.ipc.command("seek", mpv_target, "absolute+exact")
        except (MpvCommandTimeout, TimeoutError):
            log.warning("MPV seek reply timed out; checking playback position")
        except MpvError as exc:
            log.warning("MPV seek to %.3fs failed: %s", target, exc)
            if gui:
                emit_event("seek_ack", accepted=False, target=target)
            return
        pending_gui = self._pending_seek[2] if self._pending_seek is not None else False
        self._pending_seek = (mpv_target, time.monotonic() + 5.0, gui or pending_gui)
        self._native_seek_at = None

    def _seek_base(self) -> float | None:
        """Use the latest submitted target while a seek has not settled."""
        pending = self._pending_seek
        if pending is not None:
            target = self._source_time_from_mpv(pending[0])
            if target is not None:
                return target
        return self._now

    def _settle_seek(self, pos: float | None) -> bool:
        """Confirm once per tick so controls stay responsive during remote seeks."""
        now = time.monotonic()
        pending = self._pending_seek
        if pending is None:
            if self._native_seek_at is None or now - self._native_seek_at < 0.3:
                return False
            if pos is None or self.ipc.get("seeking") is not False:
                return False
        else:
            target, deadline, gui = pending
            landed = (pos is not None
                      and abs(pos - target) <= 0.75
                      and self.ipc.get("seeking") is False)
            if not landed and now < deadline:
                return False
            self._pending_seek = None
            if gui:
                emit_event("seek_ack", accepted=landed, target=self._source_time_from_mpv(target))
            if not landed:
                log.warning("MPV seek position not confirmed at %.3fs", target)
                # Preserve any native event so a late landing still resyncs audio.
                return False

        self._native_seek_at = None
        source_target = self._source_time_from_mpv(float(pos))
        self._last_seek_position = source_target
        if self._cached_seek_ready(source_target):
            return False
        self._restart_audio_at(source_target)
        return True

    def _sample_live_clock(self, mpv_pos: float, sampled_at: float) -> bool:
        """Map shared media PTS to PCM samples; never infer origin from arrival.

        Both readers consume the same fMP4 media. FFmpeg preserves PTS until
        its output filter reports sample zero, while MPV disables start-time
        rebasing. Buffering/delay therefore change position, never the mapping.
        A discontinuity invalidates this generation rather than carrying old
        subtitle positions into unrelated media.
        """
        anchor = self.ingest.live_clock_anchor()
        if anchor is None or not math.isfinite(mpv_pos):
            self._now = None
            if anchor is None and self._clock_generation is not None:
                self._mpv_anchor = None
                self._live_clock_discontinuity = True
            update_metrics(playback={"live_clock_aligned": False})
            return False
        generation, source_origin, media_origin = anchor
        if self._clock_generation is not None and generation != self._clock_generation:
            self._now = None
            self._live_clock_discontinuity = True
            self._mpv_anchor = None
            update_metrics(playback={"live_clock_aligned": False,
                                     "source_to_mpv_offset_s": None})
            return False
        if mpv_pos < media_origin - 0.25 and self._last_live_mpv_pos is None:
            # MPV's placeholder zero is not sample zero when the audio starts
            # at a large HLS PTS. Wait for a position in the measured timeline.
            self._now = None
            update_metrics(playback={"live_clock_aligned": False})
            return False
        if self._clock_generation is None:
            self._clock_generation = generation
            self._clock_source_anchor = source_origin
            self._mpv_anchor = media_origin
            record_event("live_clock_anchor", generation=generation,
                         pcm_origin_s=source_origin, media_pts_s=media_origin,
                         mpv_position_s=mpv_pos, offset_s=source_origin - media_origin)
            log.info("Live shared media anchor: generation=%d PCM=%.6f media PTS=%.6f MPV=%.6f",
                     generation, source_origin, media_origin, mpv_pos)
        previous_pos = self._last_live_mpv_pos
        previous_at = self._last_live_sample_at
        if previous_pos is not None and previous_at is not None:
            elapsed = max(0.0, sampled_at - previous_at)
            raw_step = mpv_pos - previous_pos
            if raw_step < -0.5 or raw_step > elapsed + 0.5:
                self._now = None
                self._mpv_anchor = None
                self._live_clock_discontinuity = True
                update_metrics(playback={"live_clock_aligned": False,
                                         "source_to_mpv_offset_s": None})
                record_event("mpv_clock_discontinuity", generation=generation,
                             previous_pts_s=previous_pos, media_pts_s=mpv_pos,
                             elapsed_s=elapsed)
                log.warning(
                    "MPV live clock discontinuity: %.3f -> %.3f "
                    "(elapsed=%.3fs); invalidating shared session",
                    previous_pos, mpv_pos, elapsed,
                )
                return False
        self._last_live_mpv_pos = mpv_pos
        self._last_live_sample_at = sampled_at
        properties = getattr(getattr(self, "ipc", None), "_properties", {})
        update_metrics(playback={"timeline_generation": generation,
                                 "source_to_mpv_offset_s": media_origin - source_origin,
                                 "live_clock_aligned": True,
                                 "live_playback_delay_s": self.cfg.delay_s,
                                 "live_buffering": properties.get("paused-for-cache"),
                                 "live_paused": properties.get("pause")})
        return True

    def _source_time_from_mpv(self, mpv_pos: float) -> float | None:
        """Map MPV clock to absolute audio/subtitle timestamp.

        Direct YouTube/local VOD playback already exposes the real video
        timeline; anchoring again on each audio restart accumulates drift
        after repeated seeks. Only the fMP4 fallback needs a relative clock.
        """
        if self.ingest.source.live:
            if self._mpv_anchor is None:
                return None
            return self._clock_source_anchor + mpv_pos - self._mpv_anchor
        if getattr(self, "_uses_original_timeline", False):
            return mpv_pos
        if self._mpv_anchor is None:
            return mpv_pos
        return self._clock_source_anchor + mpv_pos - self._mpv_anchor

    def _mpv_time_from_source(self, source_pos: float) -> float:
        """Inverse of _source_time_from_mpv, including for repeated seeks."""
        if getattr(self, "_uses_original_timeline", False):
            return source_pos
        if self._mpv_anchor is None:
            return source_pos
        return self._mpv_anchor + source_pos - self._clock_source_anchor

    def _cached_seek_ready(self, target: float) -> bool:
        """Only skip an audio restart when the next 5s is fully translated."""
        cache = getattr(self, "_subtitle_cache", None)
        if cache is None or not self._direct_video_loaded:
            return False
        end = cache.covered_until(target)
        # An isolated historical range must restart at its end; the current
        # audio reader could otherwise keep processing a different part of the VOD.
        return (end >= target + self.PAUSE_LEAD_SECONDS
                and self.ingest.source.start_s <= end
                and target <= self._coverage.translated_until <= end + 0.02)

    def _restart_audio_at(self, target: float) -> None:
        """Buffer translation at the confirmed video position.
        
        Enters seek-buffering state so playback auto-resumes when sufficient
        contiguous translated coverage is available. Preserves manual pause
        intent: if the user deliberately paused before seeking, playback
        remains paused after the seek.
        """
        # Preserve manual pause intent across seeks. If the user paused
        # before seeking, keep them paused after the seek completes.
        was_manually_paused = self._pause_owner == "user"
        
        self._player_command("set_property", "pause", True)
        self._now = None
        cleared = self._player_command(
            "osd-overlay",
            id=OVERLAY_ID,
            format="none",
            data="",
        )
        self._shown = "" if cleared else None
        with self._lock:
            self._lines.clear()

        self._session_start = target
        self._clock_source_anchor = target
        self._mpv_anchor = None
        self._anchor_reason = "seek"
        cache = getattr(self, "_subtitle_cache", None)
        resume = cache.covered_until(target) if cache is not None and self._uses_original_timeline else target
        if resume > target:
            log.info("Seek reused processed audio %.3f–%.3fs; resuming ASR at boundary", target, resume)
        self.ingest.request_restart(resume)
        # The queued reset marker may arrive after the MPV thread restarts.
        self.reset_buffer(target)
        self.ingest.go()

        # Enter seek-buffering state for auto-resume, unless user was manually paused
        if was_manually_paused:
            self._pause_owner = "user"
            self._seek_buffering = False
            log.info("Seek completed while manually paused; remaining paused")
        else:
            self._seek_buffering = True
            self._pause_owner = "none"
            self._vod_auto_paused = False
            log.info("Seek completed; entering seek-buffering for auto-resume")

        # Previously translated captions should be visible immediately, even
        # while FFmpeg restarts audio extraction for uncached sections.
        if getattr(self, "_subtitle_cache", None) is not None:
            self._now = target
            self._render(target)

    def _wait_for_event(
        self,
        name: str,
        timeout_s: float = 5.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        deferred: list[dict] = []
        found = False

        try:
            while time.monotonic() < deadline:
                try:
                    ev = self.ipc.events.get(
                        timeout=min(0.05, max(0.001, deadline - time.monotonic()))
                    )
                except queue.Empty:
                    continue

                if (name == "file-loaded" and ev.get("event") == "end-file"
                        and ev.get("reason") == "error"):
                    log.warning("MPV media load failed: %s", ev.get("file_error", "unknown error"))
                    break
                if ev.get("event") == name:
                    found = True
                    break

                # Keep unrelated events (including a quickly-following
                # file-loaded event) available to the next event consumer.
                deferred.append(ev)
        finally:
            for ev in deferred:
                self.ipc.events.put(ev)

        return found

    def _display_until(self, sub: Subtitle, onset: float, end: float) -> float:
        cfg = self.cfg
        return display_until(sub.en, onset, end, cfg.min_subtitle_duration_s,
                             cfg.reading_cps, cfg.max_subtitle_duration_s)

    def _render(self, now: float) -> None:
        live = self.ingest.source.live
        generation = getattr(self.ingest, "generation", None)
        with self._lock:
            self._lines = [
                ln for ln in self._lines if ln.show_until > now - 60
                and (not live or generation is None or ln.sub.generation == generation)
            ]
            active = [
                ln for ln in self._lines
                if ln.show_from <= now < ln.show_until and ln.sub.en
            ]
            pending = sum(ln.show_from > now for ln in self._lines)
        update_metrics(counters={"pending_subtitles": pending})

        # The per-video text cache survives seek resets and the rolling
        # in-memory display list. Ask it for cues at the current timestamp.
        cache = getattr(self, "_subtitle_cache", None)
        if cache is not None and not live:
            cfg = self.cfg
            active = [
                _Line(sub, onset, self._display_until(sub, onset, sub.end_s))
                for sub in cache.active_at(
                    now, max(cfg.min_subtitle_duration_s, cfg.max_subtitle_duration_s),
                    display_until=lambda sub, onset: self._display_until(sub, onset, sub.end_s))
                for onset in (sub.speech_start_s if sub.speech_start_s is not None else sub.start_s,)
                if sub.end_s > getattr(self, "_last_seek_position", -1.0)
            ]

        active.sort(key=lambda ln: ln.show_from)

        # Long cues are split into chunks that follow the speech; one chunk is shown.
        max_chars = self.cfg.max_line_chars
        shown_lines = []
        for ln in active:
            chunks = split_cue(ln.sub.en, max_chars, self.cfg.max_cue_lines)
            if len(chunks) > 1:
                speech_end = ln.sub.end_s if ln.sub.end_s > ln.show_from + 0.5 else ln.show_until
                ln = _Line(replace(ln.sub, en=chunk_at(chunks, ln.show_from, speech_end, now)),
                           ln.show_from, ln.show_until)
            shown_lines.append(ln)
        active = shown_lines

        selected: list[_Line] = []
        used_rows = 0

        for ln in reversed(active):
            # The limit counts wrapped English screen rows; the newest cue always shows.
            row_count = len(wrap_lines(ln.sub.en, max_chars))

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

        if text != self._shown:
            if not self._player_command(
                "osd-overlay",
                id=OVERLAY_ID,
                format="ass-events" if text else "none",
                data=text,
                res_x=0,
                res_y=720,
                z=10,
                hidden=False,
            ):
                return
            if text and not self._shown:
                log.debug("MPV subtitle overlay accepted at %.3fs", now)
            if text:
                anchor = getattr(self, "_mpv_anchor", None)
                for line in active:
                    onset = line.sub.speech_start_s
                    if onset is None:
                        onset = line.sub.start_s
                    record_event("subtitle_displayed", generation=line.sub.generation,
                                 speech_start_s=onset, speech_end_s=line.sub.end_s,
                                 source_position_s=now, onset_lateness_s=now - onset,
                                 mpv_position_s=getattr(self, "_last_live_mpv_pos", None),
                                 source_to_mpv_offset_s=(anchor - self._clock_source_anchor)
                                 if anchor is not None else None)
            self._shown = text


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
    # ASS alpha is transparency: 00 opaque, FF invisible.
    box_alpha = round((1 - cfg.background_opacity) * 255)
    head = rf"{{\an2\bord2.5\3c&H000000&\4c&H000000&\4a&H{box_alpha:02X}&}}"

    for sub in subtitles:
        rows: list[str] = []

        english = sanitize_english_output(sub.en)
        if english:
            rows.append(
                rf"{{\fs{cfg.en_size}\1c&HFFFFFF&}}"
                + r"\N".join(esc(line) for line in wrap_lines(english, cfg.max_line_chars))
            )

        if not rows:
            continue

        blocks.append(
            head + r"\N".join(rows)
        )

    return r"\N".join(blocks)
