
"""Translation-synchronised MPV playback.

mpv listens on a local TCP port (paused); ingest's ffmpeg streams video+audio into it.
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
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import MpvConfig
from ..ingest import Ingest, fmt_hms
from ..types import Subtitle
from ..session_cache import SUBTITLE_LINGER_S, SessionSubtitleCache
from ..timeline import record_session_anchor

log = logging.getLogger(__name__)

WINDOWS_DEFAULT = Path(r"C:\Program Files\MPV Player\mpv.exe")
OVERLAY_ID = 1
INPUT_CONF = Path(__file__).with_name("mpv_input.conf")

JUMP_MESSAGE = "auto-translate-jump"

# Limit network demux RAM on long YouTube VODs while retaining local seekback.
# The previous 1 GiB forward + 1 GiB backward limits could retain 2 GiB
# of demux packets; 256 + 64 MiB is sufficient for normal 1080p playback.
BOUNDED_DEMUX_CACHE_OPTIONS = (
    "--cache=yes",
    "--demuxer-readahead-secs=30",
    "--cache-secs=30",
    "--demuxer-hysteresis-secs=10",
    "--demuxer-max-bytes=256MiB",
    "--demuxer-max-back-bytes=64MiB",
)


def _gpu_memory_snapshot() -> str:
    """Best-effort NVIDIA VRAM diagnostic; never changes player behavior."""
    try:
        args = [
            "nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0] + " MiB (used, free, total)"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unavailable"


from ..process_cleanup import _create_kill_job, _close_job


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


def _after_speaker_handoff(active: list[_Line]) -> list[_Line]:
    """Hide old speakers once a confirmed new turn begins.

    Keep ordinary 3-row stacking for speech with no verified diarization
    handoff; stop only older cues displaced by a detected speaker change.
    This pure timeline filter also works on subtitles replayed from VOD cache.
    """
    cutoff = max(
        (line.show_from for line in active if line.sub.speaker_handoff),
        default=None,
    )
    if cutoff is None:
        return active
    return [line for line in active if line.show_from >= cutoff - 1e-6]


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
    # Minimum translated lead before starting/resuming after a seek.
    PAUSE_LEAD_SECONDS = 5.0
    LIVE_READ_SECONDS = 2.0
    LIVE_MAX_LATENESS_SECONDS = 10.0
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
        # Only direct MPV VOD playback is paced; standalone ASR and live
        # transcription retain their original audio-ingestion behavior.
        pacing = getattr(ingest, "enable_vod_playback_pacing", None)
        if not ingest.source.live and callable(pacing):
            pacing(getattr(ingest.cfg, "vod_max_audio_lead_s", 75.0))

        self._lines: list[_Line] = []
        self._subtitle_ready = threading.Event()
        self._lock = threading.Lock()
        self._now: float | None = None
        self._shown = ""
        self._live_overlay_confirmed = False
        self._lags: deque[float] = deque(maxlen=4096)  # bounded 7h VOD statistics
        self._session_start = ingest.source.start_s
        self._original_vod_start = ingest.source.start_s
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
        self._vod_health_next_at = 0.0
        # Track actual user pause separately from startup/seek translation
        # buffering so the pipeline's stall watchdog can distinguish them.
        self._playback_started = False
        self._manual_pause = False
        self._pause_poll_after = 0.0
        self._direct_video_loaded = False
        # The original VOD's mpv time-pos is the source timestamp. Do not
        # derive a new offset from the restarted FFmpeg audio reader on seek.
        # Local fMP4 fallback retains its separate anchored timeline.
        self._uses_original_timeline = False
        self._last_direct_pos: float | None = None
        self._last_direct_sample_at = 0.0

        mpv_log = (Path.cwd() / "logs" / "mpv.log")
        mpv_log.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "Before MPV startup: hwdec=%s; GPU memory=%s",
            cfg.hwdec, _gpu_memory_snapshot(),
        )

        # Concurrent Windows launches must never attach to a previous player.
        ipc_path = (
            rf"\\.\pipe\auto-translate-mpv-{uuid4().hex}"
            if sys.platform == "win32" else "/tmp/auto-translate-mpv.sock"
        )
        ytdlp_format = (
            cfg.ytdlp_format if ingest.source.live else cfg.vod_ytdlp_format
        )
        if not Path(ingest.source.location).exists():
            log.info(
                "MPV format selection: %s",
                "live HLS preference" if ingest.source.live
                else "VOD direct HTTPS/DASH first, HLS fallback",
            )
        self._proc = subprocess.Popen([
            find_mpv(cfg.path),

            "--pause",
            f"--input-ipc-server={ipc_path}",

            "--force-window=yes",
            "--title=auto-translate",
            "--keep-open=yes",
            "--idle=yes",
            "--rebase-start-time=no",

            # Keep a modest forward/backward demux cache for the direct
            # YouTube video stream.
            *BOUNDED_DEMUX_CACHE_OPTIONS,
            # Match the audio reader's live HLS edge and remove menu seeks.
            *(
                ["--demuxer-lavf-o=live_start_index=-3", "--load-context-menu=no"]
                if ingest.source.live else []
            ),
            "--input-default-bindings=no",

            # Let MPV's yt-dlp hook open the original YouTube page URL.
            # This preserves the full VOD timeline and lets MPV select both
            # video and audio tracks with the required HTTP headers.
            *(
                [f"--ytdl-format={ytdlp_format}"]
                if not Path(ingest.source.location).exists()
                else []
            ),

            # Use MPV's built-in native on-screen controller.
            "--osc=no" if ingest.source.live else "--osc=yes",
            "--osd-on-seek=no",
            "--no-input-default-bindings",
            f"--input-conf={INPUT_CONF}",

            # Hardware video decoding.
            f"--hwdec={cfg.hwdec}",

            f"--log-file={mpv_log}",
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
            self.ipc = MpvIpc(ipc_path)

            # Wait for the idle MPV IPC endpoint to become responsive.
            deadline = time.monotonic() + 10

            while self.ipc.get("idle-active") is None:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"mpv exited during startup (code {self._proc.returncode})")
                if time.monotonic() >= deadline:
                    raise TimeoutError("mpv did not respond to the startup IPC check")
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

            if hasattr(self, "ipc"):
                self.ipc.close()
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

    # ---- Output interface ----

    def is_manually_paused(self) -> bool:
        """Cached player state; no cross-thread MPV IPC requests."""
        return bool(getattr(self, "_manual_pause", False))

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
        show_from = max(self._session_start, speech_start)
        log.debug(
            "subtitle timing: audio_start=%.3f vad_onset=%.3f "
            "mpv_show=%.3f preroll=%.3fs",
            sub.start_s,
            speech_start,
            show_from,
            max(0.0, speech_start - sub.start_s),
        )
        until = sub.end_s

        now = self._now
        if self.ingest.source.live:
            # Give short or just-completed translations a readable window,
            # while keeping cues from a stalled pipeline out of later scenes.
            visible_from = max(show_from, now) if now is not None else show_from
            if visible_from - sub.end_s <= self.LIVE_MAX_LATENESS_SECONDS:
                until = max(until, visible_from + self.LIVE_READ_SECONDS)

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

        until = max(until, sub.end_s + SUBTITLE_LINGER_S)
        if self._subtitle_cache is None:
            # Live/non-cached only. VOD cues are already in the indexed
            # SessionSubtitleCache (pipeline adds before calling show).
            # Never double-store the complete seven-hour VOD in _lines:
            # the old renderer scanned that list 10 times per second.
            with self._lock:
                self._lines.append(_Line(sub, show_from, until))
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

    def _live_playback_ready(self, now: float) -> bool:
        """Buffer the live edge without delaying indefinitely for a subtitle."""
        elapsed = now - self.ingest.first_pts_at
        if elapsed < self.cfg.delay_s:
            return False
        with self._buffer_condition:
            processed = (
                self._translated_until - self._session_start
                >= min(self.cfg.delay_s, self.PAUSE_LEAD_SECONDS)
            )
        return processed or elapsed >= max(self.cfg.delay_s, 10.0)

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
        """Keep MPV open on the final frame until the user closes it."""
        while not self.stop.is_set():
            if self._proc.poll() is not None:
                return

            try:
                eof = self.ipc.get("eof-reached")
            except ConnectionError:
                if self._exit_code_after_disconnect() == 0:
                    return
                raise
            if eof:
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
        previous_code = self._proc.poll()
        if previous_code is None:
            log.info(
                "MPV was still running when application cleanup began; "
                "closing it now (this is not an mpv.exe crash)."
            )
        else:
            if isinstance(previous_code, int):
                log.info(
                    "MPV process already exited at cleanup (code=%s, hex=0x%08X)",
                    previous_code, previous_code & 0xFFFFFFFF,
                )
            else:
                log.info("MPV process already exited at cleanup (code=%s)", previous_code)
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

    def _exit_code_after_disconnect(self) -> int | None:
        code = self._proc.poll()
        if code is not None:
            return code
        # Windows can close the pipe just before the process becomes signaled.
        try:
            return self._proc.wait(timeout=0.75)
        except subprocess.TimeoutExpired:
            return None

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
                returncode = (self._exit_code_after_disconnect()
                              if isinstance(exc, ConnectionError) else self._proc.poll())
                if returncode == 0:
                    log.info("MPV window closed; stopping translator")
                elif returncode is not None:
                    log.error(
                        "MPV exited with code %s (hex=0x%08X); IPC closed (%s). "
                        "GPU memory=%s. Check logs/mpv.log.",
                        returncode, returncode & 0xFFFFFFFF, exc,
                        _gpu_memory_snapshot(),
                        exc_info=True,
                    )
                else:
                    log.exception(
                        "MPV IPC connection closed while the player may "
                        "still be running; check logs/mpv.log"
                    )
                self.stop.set()
        except Exception:
            if not self.stop.is_set():
                log.exception("MPV scheduler failed unexpectedly")
                self.stop.set()

    def _play_session(self, show_hint: bool) -> None:
        """Buffer before start/seeks; never stutter-pause during playback."""

        lead = self.PAUSE_LEAD_SECONDS

        # Load the video immediately; audio timestamp readiness must not
        # prevent MPV from opening the source. The audio pipeline can fail or
        # be delayed independently of video playback.
        self._session_start = self.ingest.source.start_s

        if not self._direct_video_loaded:
            video_url = self.ingest.source.location
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
            except ConnectionError:
                raise
            except (MpvError, OSError):
                log.exception(
                    "MPV rejected the original source; trying local fMP4 fallback"
                )

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

            # Live HLS PTS may be large/rebased; only VOD has identity mapping.
            self._uses_original_timeline = loaded and not self.ingest.source.live
            self._direct_video_loaded = True
            if self._session_start > 0:
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
        mapping = (
            "original MPV source timeline (identity; no per-seek rebasing)"
            if getattr(self, "_uses_original_timeline", False)
            else "local video fallback timeline with source/session anchor"
        )
        current_pos = None if self.ingest.source.live else self.ipc.get("time-pos")
        if current_pos is not None:
            self._mpv_anchor = float(current_pos)
        self._last_direct_pos = (
            float(current_pos) if current_pos is not None else None
        )
        self._last_direct_sample_at = time.monotonic()

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
            unpause_at = self.ingest.first_pts_at + self.cfg.delay_s
            log.info(
                "live playback delay: %.1fs (up to 10s for initial processing)",
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
        self._manual_pause = False
        self._playback_started = False
        self._pause_poll_after = 0.0
        last_audio_pts = None
        logged_live_sync = False
        live_tick = time.monotonic()
        if self.ingest.source.live:
            self._now = self._session_start

        while not self.stop.is_set():

            self._subtitle_ready.clear()
            # MPV can be manually paused while 'paused' (the startup-only
            # flag) is False. Poll ~1/s, not 10/s; the watchdog reads this
            # cached state without adding IPC traffic on another thread.
            pause_now = time.monotonic()
            if pause_now >= self._pause_poll_after:
                self._pause_poll_after = pause_now + 1.0
                actual_pause = self.ipc.get("pause")
                if actual_pause is False:
                    self._playback_started = True
                manually_paused = (
                    self._playback_started and actual_pause is True
                    and not self.ingest.source.live
                )
                if manually_paused != self._manual_pause:
                    log.info(
                        "MPV manual VOD %s; pipeline inactivity watchdog %s",
                        "pause" if manually_paused else "resume",
                        "suspended" if manually_paused else "active",
                    )
                self._manual_pause = manually_paused
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
            elif paused and unpause_at is not None and self.ingest.source.live:
                # Wait 5-10s for initial speech processing; never indefinitely.
                should_unpause = self._live_playback_ready(time.monotonic())

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

                if (
                    self._anchor_reason == "seek"
                    and not self._audio_clock_diag_done
                    and log.isEnabledFor(logging.DEBUG)
                ):
                    self._audio_clock_diag_until = time.monotonic() + 12.0
                    self._audio_clock_diag_done = True
                    log.debug("MPV audio-clock diagnostics started after seek")

                log.info(
                    "mpv playing from %s",
                    fmt_hms(self._session_start),
                )

            if self.ingest.source.live:
                # Live cues use elapsed decoded PCM time. A live HLS time-pos
                # can be missing or jump to a different transport timestamp.
                playing = (
                    not paused
                    and self.ipc.get("pause") is False
                    and not self.ipc.get("paused-for-cache")
                )
                tick = time.monotonic()
                if playing:
                    elapsed = tick - live_tick
                    self._now += elapsed
                    anchor = self.ingest.live_audio_anchor
                    if anchor is not None:
                        pts = self.ipc.get("audio-pts")
                        if pts is not None and last_audio_pts is not None and pts > last_audio_pts:
                            transport_start, cue_start = anchor
                            error = cue_start + (pts - transport_start) - self._now
                            # Independent renditions can have unrelated PTS origins.
                            # Correct plausible HLS edge differences gradually.
                            if abs(error) <= max(12.0, self.cfg.delay_s * 2):
                                limit = elapsed * 0.5
                                self._now += max(-limit, min(limit, error))
                                if not logged_live_sync:
                                    log.info("live subtitles aligned to MPV audio PTS (initial error %+.2fs)", error)
                                    logged_live_sync = True
                        last_audio_pts = pts
                    self._render(self._now)
                live_tick = tick
            elif (pos := self.ipc.get("time-pos")) is not None:
                if self._direct_video_loaded:
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
                if self.ingest.source.duration is not None:
                    self._now = min(self._now, self.ingest.source.duration)

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
                    log.debug(
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

                set_playback_pos = getattr(self.ingest, "set_playback_position", None)
                if callable(set_playback_pos):
                    set_playback_pos(self._now)
                self._render(self._now)
                # Periodic diagnostics separate video decode/network hitches
                # from subtitle inference falling behind playback.
                if (sample_now >= getattr(self, "_vod_health_next_at", float("inf"))
                        and not self.ingest.source.live):
                    self._vod_health_next_at = sample_now + 30.0
                    with self._buffer_condition:
                        translated_until = self._translated_until
                    log.info(
                        "VOD playback health: video=%.1fs translated=%.1fs "
                        "translation_lead=%+.1fs dropped_frames=%s "
                        "paused_for_cache=%s cache_duration=%s "
                        "hwdec_requested=%s hwdec_active=%s "
                        "audio_pts=%s avsync=%s decoder_drops=%s "
                        "vo_delayed=%s estimated_vf_fps=%s",
                        self._now, translated_until,
                        translated_until - self._now,
                        self.ipc.get("frame-drop-count"),
                        self.ipc.get("paused-for-cache"),
                        self.ipc.get("demuxer-cache-duration"),
                        self.cfg.hwdec, self.ipc.get("hwdec-current"),
                        self.ipc.get("audio-pts"), self.ipc.get("avsync"),
                        self.ipc.get("decoder-frame-drop-count"),
                        self.ipc.get("vo-delayed-frame-count"),
                        self.ipc.get("estimated-vf-fps"),
                    )

                sample_at = time.monotonic()
                if (
                    sample_at < self._audio_clock_diag_until
                    and sample_at - self._audio_clock_diag_last_at >= 0.5
                ):
                    audio_pts = self.ipc.get("audio-pts")
                    avsync = self.ipc.get("avsync")
                    mpv_paused = self.ipc.get("pause")
                    log.debug(
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
                    if self._jump(self._original_vod_start - self._now):
                        return

            if (delta := self._jump_request()) and self._jump(delta):
                return

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
        self.reset_buffer(restart_at)
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

    def _source_time_from_mpv(self, mpv_pos: float) -> float:
        """Map MPV clock to absolute audio/subtitle timestamp.

        Direct YouTube/local VOD playback already exposes the real video
        timeline; anchoring again on each audio restart accumulates drift
        after repeated seeks. Only the fMP4 fallback needs a relative clock.
        """
        if self.ingest.source.live:
            # Live uses decoded-PCM elapsed time in _play_session. MPV's HLS
            # time-pos can be multi-hour, discontinuous, or temporarily zero.
            # Never let that transport timestamp replace our live cue clock.
            return (
                self._now
                if self._now is not None
                else self._clock_source_anchor
            )
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

        self._pending_seek_offset = None
        self._clock_source_anchor = target
        self._mpv_anchor = None
        self._anchor_reason = "seek"
        self._clock_sample_logged = False
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
        # For VOD, the indexed cache is authoritative. Query only this
        # second's cues, even after hours of unpaced audio ingestion.
        # NEVER filter/scan _lines in the cached path: it may contain
        # thousands of future captions from a long recording.
        cache = getattr(self, "_subtitle_cache", None)
        if cache is not None:
            active = [
                _Line(
                    sub,
                    sub.speech_start_s
                    if sub.speech_start_s is not None else sub.start_s,
                    sub.end_s + SUBTITLE_LINGER_S,
                )
                for sub in cache.active_at(now, include_linger=True)
            ]
        else:
            # Live-only rolling subtitles (no per-video cache).
            with self._lock:
                self._lines = [
                    ln for ln in self._lines if ln.show_until > now - 60
                ]
                active = [
                    ln for ln in self._lines
                    if ln.show_from <= now < ln.show_until and ln.sub.en
                ]

        # An interrupted speaker's previous subtitle must stop when the
        # new speaker's cue begins. Apply the same rule to live display and
        # RAM-replayed cached VOD cues, without modifying speech timestamps.
        active = _after_speaker_handoff(active)
        active.sort(key=lambda ln: (ln.show_from, ln.sub.start_s), reverse=True)
        active = active[:max(1, self.cfg.max_subtitle_rows)]

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

            if len(self._logged_displayed_subtitles) >= 8192:
                # First-visible diagnostics need only a recent window.
                # Retaining every event over a multi-hour VOD is wasteful.
                self._logged_displayed_subtitles.clear()
            self._logged_displayed_subtitles.add(key)
            delta = now - ln.show_from
            self._lags.append(delta)
            log.debug(
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
            if text and self.ingest.source.live and not self._live_overlay_confirmed:
                self._live_overlay_confirmed = True
                log.info("MPV accepted first live subtitle overlay at %.2fs", now)
            self._shown = text

            if text:
                log.debug(
                    "MPV subtitle overlay updated: active=%d chars=%d",
                    len(active),
                    len(text),
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
