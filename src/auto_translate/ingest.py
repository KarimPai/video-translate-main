
"""yt-dlp | ffmpeg → audio chunks for VAD.

Normal mode:
    FFmpeg → 16 kHz mono → pipeline

Vocal separation mode:
    FFmpeg → stereo 44.1 kHz → UVR → 16 kHz mono → pipeline

MPV video/audio stream-copy output is kept separate from audio processing.
"""

import functools
from concurrent.futures import CancelledError
import http.server
import logging
import math
import os
import shutil
import queue
import re
import subprocess
from .windows_subprocess import owned_popen, stopping

from .windows_subprocess import hidden_console_flags
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import IngestConfig, SeparationConfig, DeepFilterConfig
from .types import SAMPLE_RATE, AudioChunk, SourceEOF
from .timeline import load_vod_metadata
from .ytdlp import extract
from .runtime_diagnostics import record_event, update_metrics

log = logging.getLogger(__name__)


class _QuietHLSHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the local fMP4 cache without noisy per-request logging."""

    def log_message(self, format: str, *args: object) -> None:
        return


CHUNK_SAMPLES = 4096
SEPARATION_READ_FRAMES = 8192
SEPARATION_INPUT_RATE = 44100
PCM_READ_TIMEOUT_S = 60.0
DEEPFILTER_INPUT_RATE = 48000

@dataclass
class Source:
    location: str
    live: bool
    start_s: float = 0.0
    paced: bool = True
    duration: float | None = None
    timeline: dict | None = None


# ---- source resolution -------------------------------------------------------------------

_HMS = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?$")


def parse_time(text: str) -> float:
    """Convert common timestamp formats to seconds."""
    text = text.strip().lower()

    if ":" in text:
        seconds = 0.0
        for part in text.split(":"):
            seconds = seconds * 60 + float(part)
        return seconds

    m = _HMS.match(text)
    if not text or not m:
        raise ValueError(
            f"bad time {text!r}; use e.g. 962, 16:02, 1:02:03 or 16m2s"
        )

    h, mins, s = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(s or 0)


def start_from_url(url: str) -> float:
    """Read YouTube's t= or start= parameter."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query) | parse_qs(parsed.fragment)

    for key in ("t", "start"):
        if key in params:
            return parse_time(params[key][0])

    return 0.0


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def resolve_source(
    location: str,
    mode: str,
    start: str | None,
    fast: bool,
    cfg: IngestConfig,
) -> Source:
    """Resolve CLI arguments into a Source."""
    is_file = Path(location).exists()
    duration: float | None = None
    timeline: dict | None = None

    if mode == "auto":
        if is_file:
            mode = "vod"
        else:
            timeline = load_vod_metadata(location, cfg)
            status = str(timeline.get("live_status") or "not_live")
            duration = (
                float(timeline["duration_s"])
                if timeline.get("duration_s") is not None else None
            )
            log.info(
                "yt-dlp: live_status=%s duration=%s",
                status,
                fmt_hms(duration) if duration else "?",
            )

            if status == "is_upcoming":
                raise SystemExit("this stream hasn't started yet")

            mode = "live" if status == "is_live" else "vod"

    if mode == "live":
        if start or start_from_url(location):
            log.warning(
                "live mode: ignoring the start time "
                "(we always start at the live edge)"
            )
        return Source(location, live=True, timeline=timeline)

    if not is_file and timeline is None:
        timeline = load_vod_metadata(location, cfg)
        status = str(timeline.get("live_status") or "not_live")

        if status == "is_upcoming":
            raise SystemExit("this stream hasn't started yet")

        # The launcher may request VOD mode, but YouTube live URLs have no
        # duration. Detect the actual source type before attempting VOD math.
        if status == "is_live":
            log.info("yt-dlp: detected live stream; using live ingest")
            return Source(location, live=True, timeline=timeline)

        raw_duration = timeline.get("duration_s")
        duration = (
            float(raw_duration) if raw_duration is not None else None
        )
    start_s = (
        parse_time(start)
        if start
        else (0.0 if is_file else start_from_url(location))
    )

    if duration and start_s >= duration:
        raise SystemExit(
            f"start {fmt_hms(start_s)} is past the end of the video "
            f"({fmt_hms(duration)})"
        )

    return Source(
        location,
        live=False,
        start_s=start_s,
        paced=not fast,
        duration=duration,
        timeline=timeline,
    )


# ---- ingest ------------------------------------------------------------------------------


def _owner_gone(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if handle:
            try:
                return kernel.WaitForSingleObject(handle, 0) == 0  # WAIT_OBJECT_0
            finally:
                kernel.CloseHandle(handle)
        return ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER for an absent PID
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):
        pass
    return False


def _prune_abandoned_sessions(parent: Path) -> None:
    for path in parent.glob("auto-translate-fmp4-pid*-*"):
        match = re.fullmatch(r"auto-translate-fmp4-pid([1-9][0-9]{0,8})-[a-z0-9_]{8}", path.name)
        if match and not path.is_symlink() and not path.is_junction() and _owner_gone(int(match[1])):
            try:
                shutil.rmtree(path)
                log.info("Removed abandoned fMP4 session %s", path.name)
            except OSError as exc:
                log.debug("Could not prune %s: %s", path.name, exc)


class Ingest:
    def __init__(
        self,
        source: Source,
        cfg: IngestConfig,
        video_out: str | None = None,
        video_format: str = "",
        burst_s: float = 0.0,
        separation_cfg: SeparationConfig | None = None,
        deepfilter_cfg: DeepFilterConfig | None = None,
    ):
        self.source = source
        # Older versions archived raw FFmpeg reports, including signed URLs.
        reports = Path(__file__).resolve().parents[2] / "logs" / "ffmpeg"
        for report in reports.glob("session-*-ffmpeg-*.log"):
            try:
                if report.stat().st_mtime < time.time() - 7 * 86400:
                    report.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not prune FFmpeg report %s: %s", report.name, exc)
        self._vod_resume_s: float | None = None
        self._vod_audio_url: str | None = None
        self._audio_client_fallback = False  # retry known 2026 YouTube client 403s
        self.cfg = cfg
        self.pitch_semitones = cfg.pitch_semitones
        self.video_format = video_format
        # Keep the temporary fMP4 cache beside the project instead of
        # using the per-user Windows TEMP directory. Each run still gets
        # its own unique session directory and is removed during cleanup.
        cache_parent = Path(__file__).resolve().parents[2] / "temp" / "fmp4"
        self._cache_temp = None
        if video_out:
            cache_parent.mkdir(parents=True, exist_ok=True)
            _prune_abandoned_sessions(cache_parent)
            self._cache_temp = tempfile.TemporaryDirectory(
                prefix=f"auto-translate-fmp4-pid{os.getpid()}-",
                dir=str(cache_parent),
            )
        self._cache_root = (
            Path(self._cache_temp.name) if self._cache_temp else None
        )
        self.video_ready = threading.Event()
        self._video_encoder = "copy"
        self._cache_sequence = 0
        self._http_server: http.server.ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._cache_source_start_s = source.start_s
        self.video_out = (
            str(self._new_video_playlist(source.start_s))
            if self._cache_root else None
        )
        self.burst_s = burst_s

        self.separation_cfg = separation_cfg
        self.separation_enabled = bool(
            separation_cfg and separation_cfg.enabled
        )

        self.deepfilter_cfg = deepfilter_cfg
        self.deepfilter_enabled = bool(
            deepfilter_cfg and deepfilter_cfg.enabled
        )

        if self.separation_enabled and self.deepfilter_enabled:
            raise ValueError(
                "UVR separation and DeepFilterNet cannot both be enabled."
            )

        self._procs: list[subprocess.Popen] = []
        self._video_proc: subprocess.Popen | None = None
        self._video_progress_lock = threading.Lock()
        self._video_progress_at = 0.0
        self._video_failed = threading.Event()
        self._proc_lock = threading.Condition()
        self._inputs: list[str] | None = None
        self._stopped = False
        self._vod_restart_enabled = False
        self._session = 0
        self._generation = 0

        self.first_pts: float | None = None
        self.first_pts_at = 0.0
        self.first_pts_ready = threading.Event()
        # first_pts/first_pts_at are PCM readiness, used for latency/chat only.
        # A presentation anchor must come from the decoded media, never from
        # when a pipe read happened to complete.
        self.audio_media_pts: float | None = None
        self.audio_media_pts_ready = threading.Event()
        self._audio_pts_samples = 0
        self._audio_pts_rate = 0
        self._audio_pts_invalid = False
        self._source_pts_origin: float | None = None
        self._source_pts_samples = 0
        self._source_pts_rate = 0

        self._restart_at: float | None = None
        self._go = threading.Event()
        # MPV alone enables the VOD audio lead cap. Live audio never parks.
        self._vod_max_audio_lead_s = 0.0
        self._playback_position_s = float(source.start_s)
        self._vod_pacing_waiting_generation: int | None = None

    def _new_video_playlist(self, source_start_s: float) -> Path:
        """Create a new HLS event playlist while retaining prior session fragments."""
        if self._cache_root is None:
            raise RuntimeError("fMP4 cache is not enabled")
        self._cache_sequence += 1
        self._cache_source_start_s = float(source_start_s)
        directory = self._cache_root / f"session-{self._cache_sequence:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "index.m3u8"

    def _start_cache_server(self) -> None:
        """Serve HLS files over loopback HTTP so Windows readers do not lock the playlist."""
        if self._http_server is not None:
            return
        if self._cache_root is None:
            raise RuntimeError("fMP4 cache is not enabled")

        handler = functools.partial(
            _QuietHLSHandler,
            directory=str(self._cache_root),
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        self._http_server = server
        self._http_thread = threading.Thread(
            target=server.serve_forever,
            name="fmp4-cache-http",
            daemon=True,
        )
        self._http_thread.start()
        log.info("local fMP4 cache HTTP server listening on 127.0.0.1:%d", server.server_port)

    @property
    def video_playlist_url(self) -> str:
        """HTTP URL for the current playlist and its relative fMP4 assets."""
        if self._http_server is None or self._cache_root is None:
            raise RuntimeError("fMP4 cache HTTP server is not running")
        relative = Path(self.video_playlist).relative_to(self._cache_root).as_posix()
        return f"http://127.0.0.1:{self._http_server.server_port}/{relative}"

    @property
    def video_playlist(self) -> str:
        """Current progressive fMP4 HLS playlist consumed by MPV and audio ingest."""
        if self.video_out is None:
            raise RuntimeError("MPV fMP4 cache is not enabled")
        return self.video_out

    def wait_video_ready(self, timeout_s: float = 45.0) -> bool:
        """Wait until the current playlist contains a complete initial fMP4 segment."""
        playlist = Path(self.video_playlist)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            proc = self._video_proc
            if self._stopped or (proc is not None and proc.poll() is not None):
                log.error("fMP4 cache writer exited before producing a segment")
                return False
            try:
                content = playlist.read_text(encoding="utf-8")
                if "#EXTINF:" in content and ".m4s" in content:
                    init_file = playlist.parent / "init.mp4"
                    segments = list(playlist.parent.glob("segment_*.m4s"))
                    if init_file.exists() and any(p.stat().st_size > 0 for p in segments):
                        return True
            except OSError:
                pass
            time.sleep(0.1)
        log.error("timed out waiting for initial fMP4 HLS segment: %s", playlist)
        return False

    @property
    def generation(self) -> int:
        with self._proc_lock:
            return self._generation

    def request_restart(
        self,
        start_s: float,
        preserve_video_cache: bool = False,
    ) -> bool:
        """Restart audio processing, optionally reusing already-downloaded fMP4 fragments."""
        with self._proc_lock:
            if self._stopped:
                return False

            preserve_video_cache = bool(
                preserve_video_cache
                and self._video_proc is not None
                and self._video_proc.poll() is None
                and not self._video_failed.is_set()
            )

            self._generation += 1
            self._proc_lock.notify_all()
            self._session += 1

            self._go.clear()
            self._playback_position_s = float(start_s)
            self._vod_pacing_waiting_generation = None
            # Reuse the same VOD audio URL across seeks. A failed read
            # already refreshes expired URLs and retries the alternate client.
            self.first_pts = None
            self.audio_media_pts = None
            self.audio_media_pts_ready.clear()
            self._audio_pts_samples = 0
            self._audio_pts_rate = 0
            self._audio_pts_invalid = False
            self._source_pts_origin = None
            self._source_pts_samples = 0
            self._source_pts_rate = 0
            self._restart_at = start_s
            self.first_pts_ready.clear()

            if not preserve_video_cache:
                self.video_ready.clear()
                self._inputs = None
                self._video_proc = None
                if self._cache_root is not None:
                    self.video_out = str(self._new_video_playlist(start_s))
                self._video_failed.clear()

            self._terminate_locked(preserve_video=preserve_video_cache)
            return preserve_video_cache

    def enable_vod_restart(self) -> None:
        if not self.source.live:
            self._vod_restart_enabled = True

    def enable_vod_playback_pacing(self, lead_s: float) -> None:
        """Bound background FFmpeg/ASR processing ahead of direct VOD playback.

        FFmpeg remains unpaced to quickly replenish the buffer after seeks.
        Throttling is inactive for live streams and local fMP4 video outputs.
        """
        if self.source.live or self.video_out is not None:
            return
        lead = max(0.0, float(lead_s))
        with self._proc_lock:
            self._vod_max_audio_lead_s = lead
            self._proc_lock.notify_all()
            self._playback_position_s = float(self.source.start_s)
            self._vod_pacing_waiting_generation = None
        if lead:
            log.info(
                "VOD audio read-ahead limited to %.0fs of MPV playback; "
                "background inference yields when sufficiently buffered",
                lead,
            )

    def set_playback_position(self, position_s: float) -> None:
        """Update current MPV VOD time for the background PCM reader."""
        with self._proc_lock:
            if self._vod_max_audio_lead_s > 0:
                self._playback_position_s = max(0.0, float(position_s))
                self._proc_lock.notify_all()

    def is_vod_audio_lead_limited(self) -> bool:
        """True only while the CURRENT VOD PCM reader is intentionally parked.

        The pipeline watchdog must not treat this deliberate queue starvation
        as a failure. The flag clears on resume, seek and stop.
        """
        with self._proc_lock:
            return (
                not self._stopped
                and self._vod_pacing_waiting_generation == self._generation
                and self._vod_max_audio_lead_s > 0
            )

    def _wait_for_vod_lead(self, sample: int, generation: int) -> bool:
        """Return False immediately when a stop/seek invalidates this read."""
        with self._proc_lock:
            while not self._stopped and generation == self._generation:
                lead = self._vod_max_audio_lead_s
                if lead <= 0 or sample / SAMPLE_RATE <= self._playback_position_s + lead:
                    self._vod_pacing_waiting_generation = None
                    return True
                self._vod_pacing_waiting_generation = generation
                self._proc_lock.wait()
            if self._vod_pacing_waiting_generation == generation:
                self._vod_pacing_waiting_generation = None
            return False

    def go(self) -> None:
        self._go.set()

    def _commands(self) -> list[list[str]]:
        if self.video_out:
            return [
                self._video_command(),
                self._video_audio_command(),
            ]
        return self._audio_commands()

    def start_video(self, start_s: float | None = None) -> None:
        """Start the paced video stream at the source or requested recovery position."""
        if not self.video_out:
            return

        with self._proc_lock:
            if self._stopped:
                return

            if (
                self._video_proc is not None
                and self._video_proc.poll() is None
            ):
                return

            session = self._session
            # FFmpeg resolves the fMP4 init filename relative to its
            # working directory. Keep that file inside this session's cache
            # directory instead of the repository/launcher working directory.
            self._start_cache_server()
            proc = owned_popen(
                self._video_command(start_s=start_s),
                cwd=str(Path(self.video_playlist).parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=hidden_console_flags(),
            )

            self._video_proc = proc
            self._procs.append(proc)

        with self._video_progress_lock:
            self._video_progress_at = time.monotonic()

        assert proc.stderr is not None

        threading.Thread(
            target=self._read_stderr,
            args=(proc.stderr, session, False),
            name="ffmpeg-video-log",
            daemon=True,
        ).start()

        threading.Thread(
            target=self._watch_video_progress,
            args=(proc,),
            name="ffmpeg-video-progress",
            daemon=True,
        ).start()

        log.info(
            "progressive fMP4 cache writer started: %s",
            self.video_playlist,
        )

    @property
    def video_failed(self) -> threading.Event:
        """Set when the current MPV video FFmpeg process stalls or exits badly."""
        return self._video_failed

    def prepare_video_retry(self) -> bool:
        """Stop a failed video sender and refresh signed source URLs for retry."""
        with self._proc_lock:
            if self._stopped:
                return False
            proc = self._video_proc
            self._video_proc = None
            self._inputs = None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass

        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

        self._video_failed.clear()
        log.info("prepared FFmpeg video retry with refreshed YouTube URLs")
        return True

    def _video_command(self, start_s: float | None = None) -> list[str]:
        """Build the paced video stream sent to MPV.

        Fetch source audio/video once and package them into a local fMP4 HLS
        playlist. MPV and the ASR audio reader consume this same growing cache.
        """
        src = self.source
        start_s = src.start_s if start_s is None else max(0.0, start_s)
        inputs = self._resolve_video_inputs()

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
        ]

        for url in inputs:
            if not src.live:
                ffmpeg += ["-readrate", "1"]

                if self.burst_s:
                    ffmpeg += [
                        "-readrate_initial_burst",
                        f"{self.burst_s:g}",
                    ]

                if start_s:
                    ffmpeg += ["-ss", f"{start_s:.3f}"]

            # Retry transient HTTP/HLS segment connection failures before
            # allowing the video sender process to exit.
            ffmpeg += [
                "-http_persistent", "0",
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
                # Regenerate missing presentation timestamps and ignore
                # potentially unreliable input indexes on YouTube HLS.
                "-fflags", "+genpts+ignidx",
                "-i", url,
            ]

        audio = f"{len(inputs) - 1}:a:0"

        assert self.video_out is not None

        playlist = Path(self.video_playlist)
        segment_pattern = str(playlist.parent / "segment_%06d.m4s")

        hls_options = [
            "-f", "hls",
            "-hls_time", "2",
            "-hls_segment_type", "fmp4",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", segment_pattern,
        ]
        if src.live:
            # Live sessions retain a bounded rolling cache; VOD sessions keep
            # every fragment so backward seeking within the session is possible.
            hls_options += [
                "-hls_list_size", "30",
                "-hls_flags", "delete_segments+independent_segments",
            ]
        else:
            hls_options += [
                "-hls_list_size", "0",
                "-hls_playlist_type", "event",
                "-hls_flags", "temp_file+independent_segments",
            ]

        video_options = ["-c:v", "copy"]
        if self._video_encoder != "copy":
            quality = (["-preset", "p1", "-rc", "vbr", "-cq", "16", "-b:v", "0"]
                       if self._video_encoder == "h264_nvenc" else ["-preset", "veryfast", "-crf", "20"])
            video_options = [
                "-vf", "setpts=N/(FRAME_RATE*TB)", "-c:v", self._video_encoder, *quality,
                "-fps_mode", "passthrough", "-force_key_frames", "expr:gte(t,n_forced*2)",
            ]
        return ffmpeg + [
            "-map", "0:v:0?", "-map", audio, *video_options,
            "-c:a", "copy",
            "-bsf:a", "aac_adtstoasc",
            *hls_options,
            # Emit a low-frequency progress heartbeat for stall diagnosis.
            "-stats_period", "5",
            "-progress", "pipe:2",
            self.video_playlist,
        ]

    def _video_audio_command(self) -> list[str]:
        """Extract ASR audio from progressive fMP4 without frame-by-frame logs."""
        if not self.video_out:
            raise RuntimeError("MPV fMP4 cache is not enabled")

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info" if self.source.live else "warning",
            "-copyts",
        ]

        offset = max(0.0, self.source.start_s - self._cache_source_start_s)
        # The growing VOD EVENT playlist is also treated as live until its
        # writer closes. Match MPV by starting at the oldest retained segment.
        ffmpeg += ["-live_start_index", "0"]
        if not self.source.live and offset > 0.001:
            ffmpeg += ["-ss", f"{offset:.3f}"]
        ffmpeg += ["-i", self.video_playlist_url]

        if self.separation_enabled:
            output_rate = SEPARATION_INPUT_RATE
        elif self.deepfilter_enabled:
            output_rate = DEEPFILTER_INPUT_RATE
        else:
            output_rate = SAMPLE_RATE

        audio_filters = []
        if self.source.live and abs(self.pitch_semitones) >= 1e-6:
            # atempo generates continuous output PTS and can hide a source
            # timestamp jump. Validate the real source before that transform.
            audio_filters += [f"aresample={output_rate}", f"asettb=1/{output_rate}",
                              "ashowinfo@source"]
        audio_filters += self._pitch_filter(output_rate)
        output_layout = "stereo" if self.separation_enabled else "mono"
        # Raw PCM has no timestamps. Keep the first *output* sample's PTS and
        # validate subsequent PTS against sample counts before removing them.
        # The stderr consumer discards frame diagnostics without logging them.
        audio_filters += [
            f"aresample={output_rate}",
            f"aformat=sample_fmts=s16:channel_layouts={output_layout}",
        ]
        if self.source.live:
            audio_filters += [f"asettb=1/{output_rate}", "ashowinfo"]
        log.info(
            "ASR audio from fMP4 cache: source_start=%.3fs playlist_offset=%.3fs rate=%dHz",
            self.source.start_s,
            offset,
            output_rate,
        )
        ffmpeg += ["-map", "0:a:0", "-af", ",".join(audio_filters)]

        if self.separation_enabled:
            ffmpeg += [
                "-ac", "2",
                "-ar", str(SEPARATION_INPUT_RATE),
                "-f", "s16le",
                "pipe:1",
            ]
        elif self.deepfilter_enabled:
            ffmpeg += [
                "-ac", "1",
                "-ar", str(DEEPFILTER_INPUT_RATE),
                "-f", "s16le",
                "pipe:1",
            ]
        else:
            ffmpeg += [
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-f", "s16le",
                "pipe:1",
            ]

        return ffmpeg

    def _resolve_video_inputs(self) -> list[str]:
        """Resolve source URLs once for both FFmpeg processes."""
        if self._inputs is None:
            if Path(self.source.location).exists():
                self._inputs = [self.source.location]
            else:
                self._inputs = extract(self.cfg, self.source.location, [
                    "-f", self.video_format, "-g",
                ]).stdout.split()
                if not self._inputs:
                    raise RuntimeError("yt-dlp returned no video URLs")
        return self._inputs

    def _read_stderr(
        self,
        stream: IO[bytes],
        session: int,
        capture_pts: bool = True,
    ) -> None:
        """Consume timestamp evidence privately and retain warning diagnostics."""
        receiver_gone = False

        last_output_time = None
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if capture_pts and "ashowinfo" in line:
                self._capture_audio_pts(line, session)
                continue

            # FFmpeg emits a progress heartbeat when transcoding fallback video.
            if not capture_pts and line.startswith("out_time_us=") and line != last_output_time:
                last_output_time = line
                with self._video_progress_lock:
                    self._video_progress_at = time.monotonic()
                continue

            receiver_gone |= any(
                value in line
                for value in (
                    "-10053",
                    "-10054",
                    "Broken pipe",
                    "Connection reset",
                )
            )

            receiver_gone |= (
                self.video_out is not None
                and "muxing a packet" in line
            )

            lower = line.lower()

            important = any(
                value in lower
                for value in (
                    "error",
                    "failed",
                    "invalid",
                    "unable to",
                    "connection refused",
                    "connection reset",
                    "broken pipe",
                )
            )

            line = re.sub(r"https?://[^\s'\"]+", "[stream URL]", line)
            if important:
                log.error("ffmpeg: %s", line)
            elif receiver_gone:
                log.debug("ffmpeg: %s", line)

    def live_clock_anchor(self) -> tuple[int, float, float] | None:
        """Atomic (generation, PCM origin, shared media PTS) evidence."""
        with self._proc_lock:
            if (self.audio_media_pts is None or self._audio_pts_invalid
                    or self.first_pts is None):
                return None
            return self._generation, float(self.first_pts), self.audio_media_pts

    def _capture_audio_pts(self, line: str, session: int) -> None:
        frame = re.search(r"\bn:(\d+)\b", line)
        pts = re.search(r"\bpts:(-?\d+)\b", line)
        rate = re.search(r"\brate:(\d+)\b", line)
        samples = re.search(r"\bnb_samples:(\d+)\b", line)
        if not all((frame, pts, rate, samples)):
            return
        try:
            sample_rate = int(rate[1])
            if sample_rate <= 0:
                return
            media_s = int(pts[1]) / sample_rate
            count = int(samples[1])
            frame_number = int(frame[1])
        except (ValueError, OverflowError):
            return
        if not math.isfinite(media_s) or sample_rate <= 0 or count <= 0:
            return
        with self._proc_lock:
            if session != self._session or self._stopped or self._audio_pts_invalid:
                return
            if "ashowinfo@source" in line:
                # This observer detects source gaps only; it must never
                # establish an output PCM anchor or signal media readiness.
                if self._source_pts_origin is None:
                    if frame_number != 0:
                        return
                    self._source_pts_origin = media_s
                    self._source_pts_rate = sample_rate
                expected = self._source_pts_origin + self._source_pts_samples / self._source_pts_rate
                if sample_rate != self._source_pts_rate or abs(media_s - expected) > 0.1:
                    self._invalidate_audio_clock(expected, media_s, "pre_pitch")
                else:
                    self._source_pts_samples += count
                return
            if self.audio_media_pts is None:
                # A later frame cannot establish where the unobserved first
                # PCM samples belonged. Fail closed and let readiness recover.
                if frame_number != 0:
                    return
                self.audio_media_pts = media_s
                self._audio_pts_rate = sample_rate
                self.audio_media_pts_ready.set()
                record_event("audio_media_anchor", generation=self._generation,
                             pcm_origin_s=self.source.start_s, media_pts_s=media_s,
                             sample_rate=sample_rate)
            expected = self.audio_media_pts + self._audio_pts_samples / self._audio_pts_rate
            error = media_s - expected
            # asettb makes integer PTS use the output sample rate. Do not use
            # pts_time's rounded decimal at multi-hour HLS origins.
            if sample_rate != self._audio_pts_rate or abs(error) > 0.1:
                self._invalidate_audio_clock(expected, media_s, "pcm_output")
                return
            previous_samples = self._audio_pts_samples
            self._audio_pts_samples += count
            if previous_samples == 0 or previous_samples // sample_rate != self._audio_pts_samples // sample_rate:
                update_metrics(playback={"audio_media_pts_s": media_s,
                                         "audio_pcm_s": self.source.start_s + self._audio_pts_samples / sample_rate,
                                         "audio_clock_error_s": error})

    def _invalidate_audio_clock(self, expected: float, media_s: float, stage: str) -> None:
        """Called under the process lock by either media timestamp observer."""
        self._audio_pts_invalid = True
        self.audio_media_pts_ready.clear()
        self._video_failed.set()
        record_event("audio_clock_discontinuity", generation=self._generation,
                     expected_pts_s=expected, media_pts_s=media_s,
                     error_s=media_s - expected, stage=stage)
        log.warning("Shared audio timeline discontinuity (%.3fs, %s); restarting alignment",
                    media_s - expected, stage)

    def _watch_video_progress(self, proc: subprocess.Popen) -> None:
        """Warn when FFmpeg stops reporting output progress for too long."""
        last_seen = 0.0
        warned = False

        while proc.poll() is None:
            time.sleep(2.0)

            with self._video_progress_lock:
                progress_at = self._video_progress_at

            if progress_at != last_seen:
                last_seen = progress_at
                warned = False
                continue

            stalled_for = time.monotonic() - progress_at
            if stalled_for >= 60.0 and self.source.live:
                with self._proc_lock:
                    if self._video_proc is proc and not self._stopped:
                        log.error("Live HLS output stalled for %.1fs", stalled_for)
                        self._video_failed.set()
                return
            if stalled_for >= 12.0 and not warned:
                with self._proc_lock:
                    current = self._video_proc is proc
                if current:
                    log.warning(
                        "FFmpeg video output made no progress for %.1fs; "
                        "the sender may be blocked or source delivery may be stalled",
                        stalled_for,
                    )
                warned = True

        returncode = proc.poll()
        with self._proc_lock:
            current = self._video_proc is proc
        if current and returncode not in (None, 0):
            log.error(
                "FFmpeg video sender exited with code %s; requesting MPV stream recovery",
                returncode,
            )
            self._video_failed.set()

    def _pitch_filter(self, sample_rate: int) -> list[str]:
        """Shift ASR input pitch while preserving its original duration."""
        semitones = self.pitch_semitones

        if abs(semitones) < 1e-6:
            return []

        # asetrate rounds its rate to an integer. Use that same effective
        # ratio for the inverse tempo so a long stream does not drift.
        shifted_rate = round(sample_rate * 2 ** (semitones / 12.0))
        tempo = sample_rate / shifted_rate

        log.info("ASR audio pitch shift: %.1f semitones", semitones)

        return [
            # Native YouTube audio is commonly 48kHz. Relabelling it as
            # ~18kHz before this conversion would almost triple its duration.
            f"aresample={sample_rate}",
            f"asetrate={shifted_rate}",
            f"aresample={sample_rate}",
            f"atempo={tempo:.12f}",
            # asetrate also scales the absolute PTS origin. Restore that
            # origin, while retaining atempo's contiguous output samples.
            f"asetpts=PTS-STARTPTS+STARTPTS*{shifted_rate}/{sample_rate}",
        ]

    def _audio_start_s(self) -> float:
        resume = self._vod_resume_s
        return self.source.start_s if resume is None else resume

    def _audio_commands(self) -> list[list[str]]:
        """Build audio-only FFmpeg commands without per-frame PTS diagnostics.

        The direct MPV path uses decoded PCM arrival for clock readiness;
        ashowinfo is unnecessary here and floods stderr with every frame.
        """
        src = self.source

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "warning",
        ]

        if not src.live:
            if src.paced:
                ffmpeg.append("-re")

            start_s = self._audio_start_s()
            if start_s:
                ffmpeg += ["-ss", f"{start_s:.6f}"]

        if self.separation_enabled:
            output_rate = SEPARATION_INPUT_RATE
            out = [
                "-vn", "-ac", "2",
                "-ar", str(output_rate),
                "-f", "s16le", "pipe:1",
            ]
        elif self.deepfilter_enabled:
            output_rate = DEEPFILTER_INPUT_RATE
            out = [
                "-vn", "-ac", "1",
                "-ar", str(output_rate),
                "-f", "s16le", "pipe:1",
            ]
        else:
            output_rate = SAMPLE_RATE
            out = [
                "-vn", "-ac", "1",
                "-ar", str(output_rate),
                "-f", "s16le", "pipe:1",
            ]

        output_layout = "stereo" if self.separation_enabled else "mono"
        filters = self._pitch_filter(output_rate)
        filters += [
            f"aresample={output_rate}",
            f"aformat=sample_fmts=s16:channel_layouts={output_layout}",
        ]
        out = ["-af", ",".join(filters)] + out

        if Path(src.location).exists():
            return [ffmpeg + ["-i", src.location] + out]

        audio_url = self._vod_audio_url
        if audio_url is None:
            audio_url = self._resolve_audio_url()
            if not src.live:
                self._vod_audio_url = audio_url

        return [ffmpeg + [
            *([] if src.live else ["-xerror", "-rw_timeout", "30000000"]),
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10",
            "-i", audio_url,
        ] + out]

    def _resolve_audio_url(self) -> str:
        resolve_cmd = ["-f", self.cfg.ytdlp_format, "-g"]
        # Some signed URLs from the 2026 default YouTube client return
        # HTTP 403 when opened by FFmpeg. Retry with an alternative client
        # only after the default URL actually fails to provide PCM.
        if self._audio_client_fallback and "youtu" in urlparse(self.source.location).netloc.lower():
            resolve_cmd += [
                "--extractor-args", "youtube:player_client=default,-web_embedded"
            ]
        audio_urls = extract(self.cfg, self.source.location, resolve_cmd).stdout.split()
        if not audio_urls:
            raise RuntimeError("yt-dlp did not return an audio stream URL")

        return audio_urls[0]

    def _wait_for_shared_live_restart(self) -> None:
        """Keep shared live MPV and PCM on the same HLS session after errors."""
        with self._proc_lock:
            if self._stopped or self._restart_at is not None:
                return
        log.warning(
            "shared live HLS audio ended; requesting synchronized "
            "MPV/video/audio session restart"
        )
        self._video_failed.set()
        # MpvOutput responds by calling request_restart(), which replaces the
        # playlist and audio generation together. Do not independently reopen
        # an older rolling HLS segment while MPV continues the old stream.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            with self._proc_lock:
                if self._stopped or self._restart_at is not None:
                    return
            time.sleep(0.1)
        raise RuntimeError("MPV did not restart the failed shared live session within 60s")

    def run(self, out_q: "queue.Queue[AudioChunk | SourceEOF | None]") -> None:
        """Read sessions and send audio chunks to the pipeline."""
        try:
            while True:
                with self._proc_lock:
                    generation = self._generation
                # A failure on the FIRST direct VOD URL must be retried too.
                # The previous logic checked whether the URL was cached before
                # the first read, so a fresh signed URL returning 403 stopped
                # all ASR without ever refreshing it.
                self._vod_resume_s = None
                retries = 0
                while True:
                    try:
                        read_failed = self._read_session(out_q)
                    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                        if not (self.source.live and self.video_out):
                            raise
                        log.warning("shared live HLS read failed: %s", exc)
                        read_failed = True

                    with self._proc_lock:
                        remote_vod = (
                            not self.source.live and not self.video_out
                            and not Path(self.source.location).exists()
                        )
                        current = (
                            not self._stopped and self._restart_at is None
                            and generation == self._generation
                        )
                        refresh = remote_vod and current and read_failed and retries < 2
                        if refresh:
                            self._vod_audio_url = None
                            # Attempt 1: fresh default URL; attempt 2:
                            # alternative player client for persistent HTTP 403.
                            self._audio_client_fallback = retries >= 1
                        give_up = remote_vod and current and read_failed and not refresh
                    if not refresh:
                        break
                    retries += 1
                    log.warning(
                        "YouTube VOD audio interrupted at %.3fs; refreshing signed URL "
                        "(retry %d/2, alternate client=%s)",
                        self._audio_start_s(), retries, self._audio_client_fallback,
                    )
                    time.sleep(0.5)
                if give_up:
                    raise RuntimeError(
                        "YouTube VOD audio inaccessible: FFmpeg could not sustain "
                        "audio after URL refresh and alternate-client retry. "
                        "Check HTTP 403 above, update yt-dlp, and try another "
                        "YouTube client or browser cookies."
                    )

                if self.source.live and self.video_out:
                    self._wait_for_shared_live_restart()

                with self._proc_lock:
                    completed = (self._vod_restart_enabled and not self.source.live
                                 and not read_failed and not self._stopped
                                 and self._restart_at is None and generation == self._generation)
                if completed:
                    out_q.put(SourceEOF(generation))
                    with self._proc_lock:
                        while not self._stopped and self._restart_at is None:
                            self._proc_lock.wait()

                with self._proc_lock:
                    if self._stopped or self._restart_at is None:
                        break

                    restart_at = self._restart_at
                    self._restart_at = None

                    self.source = replace(
                        self.source,
                        start_s=restart_at,
                    )

                    generation = self._generation

                log.info(
                    "ingest seek reset: generation=%d position=%.2fs",
                    generation,
                    self.source.start_s,
                )
                out_q.put(
                    AudioChunk(
                        round(self.source.start_s * SAMPLE_RATE),
                        np.zeros(0, dtype=np.float32),
                        time.monotonic(),
                        reset=True,
                        generation=generation,
                    )
                )

                while not self._go.wait(0.2):
                    with self._proc_lock:
                        if self._stopped:
                            return

        finally:
            out_q.put(None)

    def _read_session(
        self,
        out_q: "queue.Queue[AudioChunk | None]",
    ) -> bool:
        """Read a session. Return True on an abnormal FFmpeg exit."""
        with self._proc_lock:
            if self._stopped or self._restart_at is not None:
                return False
            generation = self._generation
        first_chunk_after_seek = generation > 0

        if self.video_out:
            # One FFmpeg process fetches and muxes the YouTube audio/video
            # into a growing fMP4 HLS event playlist. Both MPV and ASR then
            # consume this same local media timeline.
            encoders = ("copy", "h264_nvenc", "libx264")
            for encoder in encoders[encoders.index(self._video_encoder):]:
                self._video_encoder = encoder
                self.start_video(start_s=self.source.start_s)
                if self.wait_video_ready():
                    self.video_ready.set()
                    break
                if generation != self.generation or self._stopped:
                    return False
                log.warning("Live HLS writer failed with %s", encoder)
                self.prepare_video_retry()
                self.video_out = str(self._new_video_playlist(self.source.start_s))
            else:
                # A concurrent seek deliberately terminates this generation's
                # cache writer. Let the ingest loop advance to the new session.
                if generation != self.generation or self._stopped:
                    return False
                raise RuntimeError("could not initialize progressive fMP4 cache")
            cmds = [self._video_audio_command()]
        else:
            cmds = self._commands()

        procs: list[subprocess.Popen] = []
        stdin = None

        # Register every process before releasing the lock. A seek or stop
        # therefore cannot miss a process that has just been spawned.
        with self._proc_lock:
            if generation != self._generation or self._stopped:
                return False

            self._session += 1
            session = self._session

            try:
                for cmd in cmds:
                    p = owned_popen(
                        cmd,
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE if self.video_out else None,
                        creationflags=hidden_console_flags(),
                    )

                    procs.append(p)
                    self._procs.append(p)

                    if stdin is not None:
                        stdin.close()

                    stdin = p.stdout

            except BaseException:
                # Stop any processes already created by this failed startup.
                for p in procs:
                    if p.poll() is None:
                        try:
                            p.terminate()
                        except ProcessLookupError:
                            pass

                if stdin is not None:
                    try:
                        stdin.close()
                    except OSError:
                        pass

                for p in procs:
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=2)

                    for pipe in (p.stdout, p.stderr):
                        if pipe is not None:
                            try:
                                pipe.close()
                            except OSError:
                                pass

                self._procs = [
                    p for p in self._procs if p not in procs
                ]
                raise

        if self.video_out:
            assert procs[0].stderr is not None

            threading.Thread(
                target=self._read_stderr,
                args=(procs[0].stderr, session),
                name="ffmpeg-audio-log",
                daemon=True,
            ).start()

        log.info(
            "ingest started (%s): %s",
            self._describe(),
            " | ".join(c[0] for c in cmds),
        )

        stdout = procs[-1].stdout
        assert stdout is not None

        start_s = self._audio_start_s()
        first = round(start_s * SAMPLE_RATE)
        sample = first
        input_frames = 0

        # Arm only during a blocking PCM read. Queue backpressure and pauses
        # between reads are intentional and must never trigger recovery.
        read_deadline = None
        finished = threading.Event()
        shared_live = bool(self.source.live and self.video_out)
        remote_vod = not self.source.live and not self.video_out and not Path(self.source.location).exists()
        watch_reads = shared_live or remote_vod

        def watch_pcm(pcm_process):
            while not finished.wait(1.0):
                with self._proc_lock:
                    if read_deadline is not None and time.monotonic() >= read_deadline:
                        if generation == self._generation and not self._stopped:
                            log.error("FFmpeg PCM read stalled for %gs", PCM_READ_TIMEOUT_S)
                            if shared_live:
                                self._video_failed.set()
                            pcm_process.kill()
                        return
        if watch_reads:
            threading.Thread(target=watch_pcm, args=(procs[-1],), name="pcm-read-watch", daemon=True).start()

        worker = None
        deepfilter_worker = None
        pending = b""

        try:
            if self.separation_enabled:
                assert self.separation_cfg is not None

                from .separation import SeparationWorker

                worker = SeparationWorker(
                    out_q,
                    first,
                    self.separation_cfg.model,
                    generation=generation,
                    gate=getattr(self, "inference_gate", None),
                    current=lambda: self.generation,
                )

                log.info(
                    "vocal separation enabled: stereo %d Hz input",
                    SEPARATION_INPUT_RATE,
                )

            elif self.deepfilter_enabled:
                from .deepfilter import DeepFilterWorker

                deepfilter_worker = DeepFilterWorker(
                    out_q,
                    first,
                    generation=generation,
                )

                log.info(
                    "DeepFilterNet enabled: mono %d Hz input",
                    DEEPFILTER_INPUT_RATE,
                )

            while True:
                # No unbounded 7-hour VOD ingestion while MPV is paused.
                if worker:
                    reader_sample = first + round(
                        input_frames * SAMPLE_RATE / SEPARATION_INPUT_RATE
                    )
                elif deepfilter_worker:
                    reader_sample = first + round(
                        input_frames * SAMPLE_RATE / DEEPFILTER_INPUT_RATE
                    )
                else:
                    reader_sample = sample
                if not self._wait_for_vod_lead(reader_sample, generation):
                    break

                if watch_reads:
                    with self._proc_lock:
                        read_deadline = time.monotonic() + PCM_READ_TIMEOUT_S
                if worker:
                    data = stdout.read(SEPARATION_READ_FRAMES * 4)
                else:
                    data = stdout.read(CHUNK_SAMPLES * 2)

                if watch_reads:
                    with self._proc_lock:
                        read_deadline = None
                if not data:
                    break

                # Byte arrival establishes decoding readiness and the delay
                # timer only. Shared live presentation alignment separately
                # requires the output filter's actual media timestamp.
                with self._proc_lock:
                    if (
                        generation == self._generation
                        and self.first_pts is None
                    ):
                        self.first_pts = self.source.start_s
                        self.first_pts_at = time.monotonic()
                        self.first_pts_ready.set()
                if first_chunk_after_seek:
                    log.info(
                        "first audio data after seek: generation=%d position=%.2fs",
                        generation,
                        self.source.start_s,
                    )
                    first_chunk_after_seek = False

                if worker:
                    data = pending + data

                    usable = len(data) - (len(data) % 4)
                    pending = data[usable:]
                    data = data[:usable]

                    if not data:
                        continue

                    pcm = (
                        np.frombuffer(data, dtype="<i2")
                        .reshape(-1, 2)
                        .astype(np.float32)
                        / 32768.0
                    )

                    input_frames += len(pcm)
                    worker.put(pcm)

                elif deepfilter_worker:
                    data = pending + data

                    usable = len(data) - (len(data) % 2)
                    pending = data[usable:]
                    data = data[:usable]

                    if not data:
                        continue

                    pcm = (
                        np.frombuffer(data, dtype="<i2")
                        .astype(np.float32)
                        / 32768.0
                    )

                    input_frames += len(pcm)
                    deepfilter_worker.put(pcm)
                    deepfilter_worker.put_progress()

                else:
                    pcm = (
                        np.frombuffer(data, dtype="<i2")
                        .astype(np.float32)
                        / 32768.0
                    )

                    if generation != self.generation:
                        break

                    out_q.put(
                        AudioChunk(
                            sample,
                            pcm,
                            time.monotonic(),
                            generation=generation,
                        )
                    )

                    sample += len(pcm)

            if pending:
                log.warning(
                    "Discarded %d incomplete bytes from FFmpeg output",
                    len(pending),
                )

        except CancelledError:
            # Optional separation admission can be invalidated by a seek or
            # shutdown. Unwind this audio generation and let run() restart it.
            gate = getattr(self, "inference_gate", None)
            if (generation == self.generation and not self._stopped
                    and not stopping.is_set()
                    and not (gate is not None and gate.stop.is_set())):
                raise
        finally:
            finished.set()
            try:
                if worker:
                    worker.close()

                if deepfilter_worker:
                    deepfilter_worker.close()

            finally:
                for p in procs:
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=2)

                # Remove only this session's audio processes. Do not replace
                # the shared list: it may also contain the MPV video process.
                with self._proc_lock:
                    self._procs = [
                        p for p in self._procs if p not in procs
                    ]

        if worker:
            processed_seconds = input_frames / SEPARATION_INPUT_RATE
        elif deepfilter_worker:
            processed_seconds = input_frames / DEEPFILTER_INPUT_RATE
        else:
            processed_seconds = (sample - first) / SAMPLE_RATE

        log.info(
            "ingest session ended after %.1fs of audio (exit codes %s)",
            processed_seconds,
            [p.poll() for p in procs],
        )

        if input_frames == 0 and (worker or deepfilter_worker):
            log.error(
                "no audio received from FFmpeg; check the source and audio stream"
            )

        elif sample == first and not (worker or deepfilter_worker):
            if self.source.start_s and self._restart_at is None:
                log.error(
                    "no audio after %s: is the start time past the end of the video?",
                    fmt_hms(self.source.start_s),
                )

        failed = any(p.returncode for p in procs)
        if failed and remote_vod:
            self._vod_resume_s = start_s + processed_seconds
        return failed

    def _describe(self) -> str:
        src = self.source
        video = f", video → {self.video_out}" if self.video_out else ""

        if src.live:
            return "live" + video

        return (
            f"vod from {src.start_s:.1f}s, "
            f"{'1x' if src.paced else 'fast'}{video}"
        )

    def stop(self) -> None:
        with self._proc_lock:
            self._stopped = True
            self._proc_lock.notify_all()
            self._video_proc = None
            self._video_failed.clear()
            self._go.set()
            processes = list(self._procs)
            self._terminate_locked()

        # Wait for FFmpeg to release its files before removing the temporary
        # cache. Windows does not allow deleting files still held open.
        for proc in processes:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    log.warning("could not fully stop FFmpeg process %s", proc.pid)

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
            self._http_thread = None

        if self._cache_temp is not None:
            self._cache_temp.cleanup()
            self._cache_temp = None

    def _terminate_locked(self, preserve_video: bool = False) -> None:
        """Terminate registered processes, optionally keeping the cache writer alive."""
        for p in self._procs:
            if preserve_video and p is self._video_proc:
                continue
            if p.poll() is None:
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass

    def _terminate(self) -> None:
        with self._proc_lock:
            self._terminate_locked()
