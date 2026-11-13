
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
from .rtx_vsr import (
    VSR_FILTER, VSR_LABEL, VsrPerformanceGuard,
    filter_present, launch_options, source_eligible,
)
from ..ingest import Ingest, fmt_hms
from ..types import Subtitle
from ..session_cache import SessionSubtitleCache
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
            ov.cancel()
            ov.GetOverlappedResult(True)
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
                if "event" in msg:
                    self.events.put(msg)
                elif "request_id" in msg:
                    self._responses.put(msg)
        except (OSError, ValueError, ConnectionError) as exc:
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
                    raise MpvError(
                        f"mpv IPC command timed out after "
                        f"{self.COMMAND_TIMEOUT_S:.1f}s: {cmd!r}"
                    )

                try:
                    msg = self._responses.get(timeout=remaining)
                except queue.Empty:
                    raise MpvError(
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


def _mpv_media_url(ingest: Ingest) -> str:
    """Share media segments in live mode; preserve original VOD playback."""
    if ingest.source.live and ingest.video_out:
        return ingest.video_playlist_url
    return ingest.source.location


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
        self._translated_until = ingest.source.start_s
        self._source_finished = False
        self._job = None
        self._closed = False
        self._video_retries = 0
        self._last_video_recovery_at: float | None = None
        self._mpv_anchor: float | None = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "initial"
        self._direct_video_loaded = False
        # The original VOD's mpv time-pos is the source timestamp. Do not
        # derive a new offset from the restarted FFmpeg audio reader on seek.
        # Local fMP4 fallback retains its separate anchored timeline.
        self._uses_original_timeline = False
        self._last_direct_pos: float | None = None
        self._last_direct_sample_at = 0.0
        self._last_live_mpv_pos: float | None = None
        self._last_live_sample_at: float | None = None
        self._manually_paused = False
        self._vsr_launch_args = launch_options(cfg.rtx_vsr, cfg.hwdec, sys.platform)
        self._vsr_requested = bool(self._vsr_launch_args)
        self._vsr_active = False
        self._vsr_guard = VsrPerformanceGuard()
        self._vsr_next_health_at = 0.0
        if cfg.rtx_vsr and not self._vsr_requested:
            log.warning(
                "RTX VSR skipped: requires Windows and --hwdec=auto-safe; "
                "keeping standard MPV playback"
            )
        if not ingest.source.live:
            ingest.enable_vod_playback_pacing(
                getattr(ingest.cfg, "vod_max_audio_lead_s", 75.0)
            )

        mpv_log = (Path.cwd() / "logs" / "mpv.log")
        mpv_log.parent.mkdir(parents=True, exist_ok=True)

        self._proc = subprocess.Popen([
            find_mpv(cfg.path),

            "--pause",
            f"--input-ipc-server={IPC_PATH}",

            "--force-window=yes",
            "--title=auto-translate",
            "--keep-open=yes",
            "--idle=yes",
            "--rebase-start-time=no",

            # Keep a modest forward/backward demux cache for the direct
            # YouTube video stream.
            "--cache=yes",
            "--demuxer-readahead-secs=120",
            "--demuxer-max-bytes=1GiB",
            "--demuxer-max-back-bytes=1GiB",
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
            *(
                [f"--ytdl-format={cfg.ytdlp_format}"]
                if not Path(ingest.source.location).exists()
                else []
            ),

            # Use MPV's built-in native on-screen controller.
            "--osc=yes",
            "--osd-on-seek=no",
            "--no-input-default-bindings",
            f"--input-conf={INPUT_CONF}",

            # Hardware video decoding.
            f"--hwdec={cfg.hwdec}",
            *self._vsr_launch_args,

            f"--log-file={mpv_log}",
            "--msg-level=all=warn",
        ])

        if sys.platform == "win32":
            try:
                self._job = _create_kill_job(self._proc)
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
            "mpv ready for direct source video playback "
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

    def _enable_vsr_if_eligible(self) -> None:
        """Add NVIDIA scaling only after MPV confirms a <=480p video source."""
        video_params = self.ipc.get("video-params")
        if not source_eligible(video_params):
            log.warning(
                "RTX VSR skipped: source height not confirmed <=480p (%r); "
                "standard MPV scaling remains active", video_params,
            )
            self._vsr_requested = False
            return

        try:
            self.ipc.command("vf", "add", VSR_FILTER)
            if not filter_present(self.ipc.get("vf")):
                raise MpvError("MPV did not retain the NVIDIA video filter")
        except (MpvError, ConnectionError, OSError) as exc:
            self._disable_vsr(f"filter unavailable: {exc}")
            return

        self._vsr_active = True
        self._vsr_next_health_at = time.monotonic() + 30.0
        log.info(
            "RTX VSR filter requested for %sp source: 3x NVIDIA D3D11 scaling. "
            "NVIDIA App must enable RTX VSR and set High; MPV filter acceptance "
            "alone does not confirm AI enhancement.", video_params["h"],
        )

    def _disable_vsr(self, reason: str) -> None:
        """Remove only the enhancement filter, preserving playback and translation."""
        self._vsr_active = False
        self._vsr_requested = False
        try:
            self.ipc.command("vf", "remove", VSR_LABEL)
        except (MpvError, ConnectionError, OSError) as exc:
            log.warning("RTX VSR filter removal not confirmed: %s", exc)
        log.warning(
            "RTX VSR disabled (%s); returning to ordinary MPV scaling. "
            "Haas, PLaMo, subtitles, and seeking remain unchanged.", reason,
        )

    def _check_vsr_health(self) -> None:
        """Disable the optional filter if 30-second playback checks fail."""
        if not self._vsr_active or not self._direct_video_loaded:
            return
        now = time.monotonic()
        if now < self._vsr_next_health_at:
            return
        self._vsr_next_health_at = now + 30.0

        if not filter_present(self.ipc.get("vf")):
            self._disable_vsr("MPV no longer reports the NVIDIA filter")
            return

        hwdec = self.ipc.get("hwdec-current")
        dropped = self.ipc.get("frame-drop-count")
        delayed = self.ipc.get("vo-delayed-frame-count")
        paused = bool(self.ipc.get("pause"))
        caching = bool(self.ipc.get("paused-for-cache"))
        reason = self._vsr_guard.sample(
            now=now, frame_drops=dropped, delayed_frames=delayed,
            hwdec=hwdec, paused=paused, caching=caching,
        )
        log.info(
            "RTX VSR playback check: hwdec=%s dropped=%s delayed=%s "
            "manual_or_buffer_pause=%s cache_pause=%s",
            hwdec, dropped, delayed, paused, caching,
        )
        if reason:
            self._disable_vsr(reason)

    # ---- Output interface ----

    def is_manually_paused(self) -> bool:
        """Expose manual VOD pause to the pipeline watchdog."""
        return self._manually_paused

    def show(self, sub: Subtitle) -> None:
        if sub.end_s < self._session_start:
            return

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
            allowed_late = max(0.0, getattr(getattr(self, "cfg", None), "live_late_window_s", self.LIVE_LATE_WINDOW_S))
            if self.ingest.source.live and late_by <= allowed_late:
                log.warning(
                    "displaying late live subtitle now: "
                    "generation=%d speech_end=%.3f mpv_clock=%.3f late=%.3fs",
                    sub.generation, sub.end_s, now, late_by,
                )
                show_from = now
                until = now + self.LIVE_LATE_DISPLAY_S
            else:
                log.warning(
                    "dropping expired subtitle: generation=%d "
                    "speech_end=%.3f mpv_clock=%.3f late=%.3fs live=%s",
                    sub.generation, sub.end_s, now, late_by,
                    self.ingest.source.live,
                )
                return

        line = _Line(sub, show_from, until)
        with self._lock:
            self._lines.append(line)
        self._subtitle_ready.set()

    def advance(self, progress_s: float) -> None:
        """Record the latest audio position safely processed for translation."""
        with self._buffer_condition:
            self._translated_until = max(
                self._translated_until,
                progress_s,
            )
            if self._seek_lead_target is not None and self._translated_until >= self._seek_lead_target:
                log.info("translation lead ready at %.3fs", self._seek_lead_target)
                self._seek_lead_target = None
                self._subtitle_ready.set()
            self._buffer_condition.notify_all()

    def playback_position(self):
        """Last sampled position in the subtitle source timeline."""
        return self._now

    def _translation_ready(self) -> bool:
        """Start after a 5s lead from new work or previously cached audio."""
        with self._buffer_condition:
            ready = self._source_finished or (
                self._now is not None
                and self._translated_until >= self._now + self.PAUSE_LEAD_SECONDS
            )
        if ready:
            return True
        cache = getattr(self, "_subtitle_cache", None)
        now = self._now
        return (
            cache is not None and now is not None
            and cache.covered_until(now) >= now + self.PAUSE_LEAD_SECONDS
        )

    def reset_buffer(self, progress_s: float) -> None:
        """Reset translated coverage after a VOD seek."""
        with self._buffer_condition:
            self._seek_lead_target = progress_s + self.PAUSE_LEAD_SECONDS
            self._translated_until = progress_s
            self._source_finished = False
            self._buffer_condition.notify_all()

    def finish(self) -> None:
        """Signal that no more translated audio will arrive."""
        with self._buffer_condition:
            self._source_finished = True
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

    def _play_session(self, show_hint: bool) -> None:
        """Buffer before start/seeks; never stutter-pause during playback."""

        lead = self.PAUSE_LEAD_SECONDS

        # Load the video immediately; audio timestamp readiness must not
        # prevent MPV from opening the source. The audio pipeline can fail or
        # be delayed independently of video playback.
        self._session_start = self.ingest.source.start_s

        if not self._direct_video_loaded:
            shared_live_hls = bool(
                self.ingest.source.live and self.ingest.video_out
            )
            if shared_live_hls:
                # The ingest worker starts the shared fMP4 writer. MPV must
                # open its existing playlist, not a separately resolved YouTube
                # live rendition that may begin at another HLS segment.
                if not self.ingest.wait_video_ready():
                    log.error("shared live HLS playlist unavailable; stopping")
                    self.stop.set()
                    return
                log.info("MPV live: loading same local HLS playlist as ASR")
            video_url = _mpv_media_url(self.ingest)
            if not shared_live_hls:
                log.info(
                    "loading original source in MPV through its yt-dlp hook"
                    if not Path(video_url).exists()
                    else "loading local source in MPV"
                )
            loaded = False
            try:
                self.ipc.command("loadfile", video_url, "replace")
                started = self._wait_for_event("start-file", timeout_s=10.0)
                loaded = (
                    started
                    and self._wait_for_event("file-loaded", timeout_s=30.0)
                )
            except (MpvError, ConnectionError, OSError):
                log.exception("MPV rejected the original source")

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

            # Apply only after file-loaded; never interfere with native seek or
            # fail the source load when NVIDIA scaling is unavailable.
            if self._vsr_requested:
                self._enable_vsr_if_eligible()

            self._uses_original_timeline = loaded and not self.ingest.source.live
            self._direct_video_loaded = True
            if self._session_start > 0 and not self.ingest.source.live:
                self.ipc.command("seek", self._session_start, "absolute+exact")
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    pos = self.ipc.get("time-pos")
                    if pos is not None and abs(float(pos) - self._session_start) <= 1.0:
                        break
                    time.sleep(0.05)

        while not self.ingest.first_pts_ready.wait(0.2):
            if self.stop.is_set():
                return

        assert self.ingest.first_pts is not None

        # Ingest updates source.start_s asynchronously after a restart. Never
        # anchor a seek to the previous audio session's source.start_s.
        self._session_start = self.ingest.source.start_s

        # MPV plays the original VOD timeline; FFmpeg independently extracts
        # audio for ASR. Anchor each audio generation to MPV's current position.
        self._mpv_anchor = None
        self._clock_source_anchor = self._session_start
        self._anchor_reason = "seek" if self.ingest.generation > 0 else "initial"
        self._last_live_mpv_pos = None
        self._last_live_sample_at = None
        current_pos = self.ipc.get("time-pos")
        # An HLS live clock may initially report 0 (or a placeholder) and
        # jump to the playlist's multi-hour media PTS after file-loaded.
        # Do not anchor live captions to that unstable startup sample.
        if current_pos is not None and not self.ingest.source.live:
            self._mpv_anchor = float(current_pos)
        self._last_direct_pos = (
            float(current_pos) if current_pos is not None else None
        )
        self._last_direct_sample_at = time.monotonic()

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

            self._subtitle_ready.clear()
            if self.ingest.video_failed.is_set():
                if self._recover_video_stream():
                    return
                return

            should_unpause = (
                unpause_at is not None
                and time.monotonic() >= unpause_at
            )

            if paused and unpause_at is None:
                should_unpause = self._translation_ready()

            # User input overrides the optional translation prebuffer. Never
            # force MPV back to pause immediately after its Play button is used.
            # Once manually playing, do not auto-pause for low translation lead.
            if paused and self.ipc.get("pause") is False:
                paused = False
                unpause_at = None
                log.info(
                    "MPV manually started; continuing playback without "
                    "waiting for the translation buffer"
                )

            if paused and should_unpause:

                self.ipc.command(
                    "set_property",
                    "pause",
                    False,
                )

                paused = False
                unpause_at = None

                log.info(
                    "mpv playing from %s",
                    fmt_hms(self._session_start),
                )

            self._manually_paused = (
                not paused and self.ipc.get("pause") is True
            )
            if (pos := self.ipc.get("time-pos")) is not None:
                if self.ingest.source.live:
                    self._sample_live_clock(float(pos), time.monotonic())
                if self._direct_video_loaded and not self.ingest.source.live:
                    sampled_at = time.monotonic()
                    if self._last_direct_pos is not None:
                        raw_step = float(pos) - self._last_direct_pos
                        elapsed = sampled_at - self._last_direct_sample_at
                        if (
                            abs(raw_step) >= 2.0
                            and abs(raw_step - elapsed) >= 1.5
                        ):
                            log.info(
                                "native MPV seek detected: %.3fs -> %.3fs",
                                self._last_direct_pos,
                                float(pos),
                            )
                            source_target = self._source_time_from_mpv(float(pos))
                            if self._cached_seek_ready(source_target):
                                log.info(
                                    "native MPV seek %.3fs: using cached translations "
                                    "without restarting audio",
                                    source_target,
                                )
                            else:
                                self._restart_audio_at(
                                    source_target,
                                    seek_video=False,
                                )
                                return
                    self._last_direct_pos = float(pos)
                    self._last_direct_sample_at = sampled_at

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

                self._now = self._source_time_from_mpv(float(pos))
                if not self.ingest.source.live:
                    self.ingest.set_playback_position(self._now)
                if self.ingest.source.duration is not None:
                    self._now = min(self._now, self.ingest.source.duration)

                self._render(self._now)

            if self._replay_requested():
                if self._now is not None and self.ipc.get("eof-reached"):
                    log.info("replay requested after VOD end; restarting from beginning")
                    if self._jump(self.ingest.source.start_s - self._now):
                        return

            if (delta := self._jump_request()) and self._jump(delta):
                return

            self._check_vsr_health()

            # Wake for newly translated text or the next queued cue boundary.
            wait_s = 0.1
            if not paused and self._now is not None:
                with self._lock:
                    for line in self._lines:
                        for boundary in (line.show_from, line.show_until):
                            if self._now < boundary < self._now + wait_s:
                                wait_s = boundary - self._now
            self._subtitle_ready.wait(max(0.005, wait_s))

    def _recover_video_stream(self) -> bool:
        """Restart shared video/audio with a bounded short-window retry policy."""
        now = time.monotonic()
        previous = self._last_video_recovery_at
        # A five-hour livestream should not be terminated by three unrelated
        # transient network faults over its entire lifetime. Limit only
        # repeated recovery loops in a three-minute window.
        if previous is None or now - previous >= 180.0:
            self._video_retries = 0
        if self._video_retries >= 3:
            log.error("fMP4 cache recovery stopped after 3 attempts within 3 minutes")
            self.stop.set()
            return False
        self._last_video_recovery_at = now

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

        self.ingest.request_restart(restart_at)
        if self.ingest.source.live and self.ingest.video_out:
            # Reopen the fresh shared playlist, never continue playing the
            # disconnected HLS session while audio restarts independently.
            self._direct_video_loaded = False
        self.reset_buffer(restart_at)
        self.ingest.go()
        self._clock_source_anchor = restart_at
        self._mpv_anchor = None
        self._anchor_reason = "video-recovery"
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
                # Preserve seek requests for the normal jump handler.
                self.ipc.events.put(ev)
                break
        return requested

    def _jump_request(self) -> float:
        """Collect jump-key requests from mpv."""
        delta = 0.0

        while not self.ipc.events.empty():
            ev = self.ipc.events.get()
            args = ev.get("args") or []

            if ev.get("event") != "client-message" or not args:
                continue
            if args[0] == JUMP_MESSAGE:
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
            target = min(target, max(0.0, src.duration - 5))

        log.info("jump %+.0fs -> %s", delta, fmt_hms(target))
        if self._jump_to_cached(target):
            # Already seeking/playing from RAM-backed subtitles. Returning
            # False keeps the current scheduler alive; True restarts it.
            return False
        return self._restart_audio_at(target, seek_video=True)

    def _sample_live_clock(self, mpv_pos: float, sampled_at: float) -> None:
        """Keep PCM-relative live subtitle time continuous across HLS PTS jumps.

        Some YouTube HLS streams initially expose time-pos=0 in MPV and then
        switch to a multi-hour media timestamp. Rebase on that discontinuity
        rather than treating new subtitles as hours late. Ordinary realtime
        advancement, including long scheduler stalls, must not trigger rebasing.
        """
        previous_pos = self._last_live_mpv_pos
        previous_at = self._last_live_sample_at
        if previous_pos is not None and previous_at is not None:
            elapsed = max(0.0, sampled_at - previous_at)
            raw_step = mpv_pos - previous_pos
            if raw_step < -5.0 or raw_step > elapsed + 5.0:
                source_now = (
                    self._now
                    if self._now is not None
                    else self._clock_source_anchor
                )
                self._mpv_anchor = (
                    mpv_pos - (source_now - self._clock_source_anchor)
                )
                log.warning(
                    "MPV live clock discontinuity: %.3f -> %.3f "
                    "(elapsed=%.3fs); reanchored subtitles at %.3fs",
                    previous_pos, mpv_pos, elapsed, source_now,
                )
        self._last_live_mpv_pos = mpv_pos
        self._last_live_sample_at = sampled_at

    def _source_time_from_mpv(self, mpv_pos: float) -> float:
        """Map MPV clock to absolute audio/subtitle timestamp.

        Direct YouTube/local VOD playback already exposes the real video
        timeline; anchoring again on each audio restart accumulates drift
        after repeated seeks. Only the fMP4 fallback needs a relative clock.
        """
        if self.ingest.source.live:
            # MPV HLS time-pos may be a playlist timestamp, whereas FFmpeg
            # advances the ASR clock from decoded PCM seconds. Anchor the two
            # clocks to their initial playback positions, not absolute PTS.
            if self._mpv_anchor is None:
                return self._clock_source_anchor
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
        return bool(
            cache is not None
            and getattr(self, "_direct_video_loaded", False)
            and cache.covered_until(target) >= target + self.PAUSE_LEAD_SECONDS
        )

    def _jump_to_cached(self, target: float) -> bool:
        """Seek within translated VOD audio without pausing or restarting ASR.

        Repeated short rewinds can use subtitles already stored in RAM while
        the ingestion/translation workers continue running further ahead.
        """
        if not self._cached_seek_ready(target):
            return False

        mpv_target = self._mpv_time_from_source(target)
        self.ipc.command("seek", mpv_target, "absolute+exact")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self.stop.is_set():
            landed = self.ipc.get("time-pos")
            if landed is not None and abs(float(landed) - mpv_target) <= 0.75:
                # Mark the new position as the native-seek detector's baseline.
                # Otherwise the next scheduler tick sees a sudden -10s step
                # and unnecessarily restarts the audio reader.
                self._last_direct_pos = float(landed)
                self._last_direct_sample_at = time.monotonic()
                self._now = target
                self._render(target)
                log.info(
                    "cached VOD seek to %.3fs: subtitles replayed without "
                    "pausing MPV or restarting translation",
                    target,
                )
                return True
            time.sleep(0.05)
        log.warning(
            "cached VOD seek position not confirmed at %.3fs; "
            "using normal audio restart",
            mpv_target,
        )
        return False

    def _restart_audio_at(self, target: float, *, seek_video: bool) -> bool:
        """Restart translation audio at a source timestamp without reloading video."""
        src = self.ingest.source
        target = max(0.0, target)
        if src.duration:
            target = min(target, max(0.0, src.duration - 1.0))

        self.ipc.command("set_property", "pause", True)
        if seek_video:
            mpv_target = self._mpv_time_from_source(target)
            self.ipc.command("seek", mpv_target, "absolute+exact")
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

        self._clock_source_anchor = target
        self._mpv_anchor = None
        self._anchor_reason = "seek"
        self.ingest.request_restart(target)
        # The queued reset marker may arrive after the MPV thread restarts.
        self.reset_buffer(target)
        self.ingest.go()

        if seek_video:
            # MPV acknowledges seek before it necessarily reaches that time.
            # Sampling the prior time-pos here would corrupt the new anchor.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not self.stop.is_set():
                landed = self.ipc.get("time-pos")
                if landed is not None and abs(float(landed) - mpv_target) <= 0.75:
                    break
                time.sleep(0.05)
            else:
                log.warning(
                    "MPV seek position not confirmed at %.3fs; "
                    "rechecking the playback anchor",
                    mpv_target,
                )

        landed = self.ipc.get("time-pos")
        self._last_direct_pos = float(landed) if landed is not None else None
        self._last_direct_sample_at = time.monotonic()

        # Previously translated captions should be visible immediately, even
        # while FFmpeg restarts audio extraction for uncached sections.
        if getattr(self, "_subtitle_cache", None) is not None:
            self._now = target
            self._render(target)
        return True

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

    def _render(self, now: float) -> None:
        with self._lock:
            self._lines = [
                ln for ln in self._lines if ln.show_until > now - 60
            ]
            active = [
                ln for ln in self._lines
                if ln.show_from <= now < ln.show_until and ln.sub.en
            ]

        # The per-video text cache survives seek resets and the rolling
        # in-memory display list. Ask it for cues at the current timestamp.
        cache = getattr(self, "_subtitle_cache", None)
        if cache is not None:
            active = [
                _Line(
                    sub,
                    sub.speech_start_s
                    if sub.speech_start_s is not None else sub.start_s,
                    sub.end_s,
                )
                for sub in cache.active_at(now)
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

        if text != self._shown:
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
