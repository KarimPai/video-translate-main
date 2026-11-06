
"""yt-dlp | ffmpeg → audio chunks for VAD and diarization.

Normal mode:
    FFmpeg → 16 kHz mono → pipeline

Vocal separation mode:
    FFmpeg → stereo 44.1 kHz → UVR → 16 kHz mono → pipeline

MPV video/audio stream-copy output is kept separate from audio processing.
"""

import logging
import queue
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import IngestConfig, SeparationConfig, DeepFilterConfig
from .types import SAMPLE_RATE, AudioChunk
from .timeline import load_vod_metadata

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 4096
SEPARATION_READ_FRAMES = 8192
SEPARATION_INPUT_RATE = 44100
DEEPFILTER_INPUT_RATE = 48000

_PTS = re.compile(r"pts_time:(-?[\d.]+)")
_ASHOWINFO_SAMPLES = re.compile(r"nb_samples:(\d+)")


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


def _cookie_args(cfg: IngestConfig) -> list[str]:
    return (
        ["--cookies-from-browser", cfg.cookies_from_browser]
        if cfg.cookies_from_browser else []
    )


def probe(url: str, cfg: IngestConfig) -> tuple[str, float | None]:
    """Fetch VOD metadata and return live status and duration."""
    metadata = load_vod_metadata(url, cfg)
    duration = metadata.get("duration_s")
    return (
        str(metadata.get("live_status") or "not_live"),
        float(duration) if duration is not None else None,
    )


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
        return Source(location, live=True)

    if not is_file and timeline is None:
        timeline = load_vod_metadata(location, cfg)
        duration = float(timeline["duration_s"])
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
        self.cfg = cfg
        self.pitch_semitones = cfg.pitch_semitones
        self.video_format = video_format
        self._cache_temp = (
            tempfile.TemporaryDirectory(prefix="auto-translate-fmp4-")
            if video_out else None
        )
        self._cache_root = (
            Path(self._cache_temp.name) if self._cache_temp else None
        )
        self._cache_sequence = 0
        self.video_out = (
            str(self._new_video_playlist()) if self._cache_root else None
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
        self._proc_lock = threading.Lock()
        self._inputs: list[str] | None = None
        self._stopped = False
        self._session = 0
        self._generation = 0

        self.first_pts: float | None = None
        self.first_pts_at = 0.0
        self.first_pts_ready = threading.Event()

        self._restart_at: float | None = None
        self._go = threading.Event()

    def _new_video_playlist(self) -> Path:
        """Create a new HLS event playlist while retaining prior session fragments."""
        if self._cache_root is None:
            raise RuntimeError("fMP4 cache is not enabled")
        self._cache_sequence += 1
        directory = self._cache_root / f"session-{self._cache_sequence:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "index.m3u8"

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
            if proc is not None and proc.poll() not in (None, 0):
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

    def request_restart(self, start_s: float) -> None:
        """Invalidate the current generation and stop its FFmpeg processes."""
        with self._proc_lock:
            if self._stopped:
                return

            self._generation += 1
            self._session += 1

            self._inputs = None
            self._video_proc = None
            if self._cache_root is not None:
                self.video_out = str(self._new_video_playlist())
            self._video_failed.clear()
            self._go.clear()
            self.first_pts = None
            self._restart_at = start_s
            self.first_pts_ready.clear()

            self._terminate_locked()

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
            proc = subprocess.Popen(
                self._video_command(start_s=start_s),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
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

        log.info("video stream started after translation buffer")

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
                "-hls_flags", "delete_segments+temp_file+independent_segments",
            ]
        else:
            hls_options += [
                "-hls_list_size", "0",
                "-hls_playlist_type", "event",
                "-hls_flags", "temp_file+independent_segments",
            ]

        return ffmpeg + [
            "-map", "0:v:0?",
            "-map", audio,
            "-c", "copy",
            *hls_options,
            # Emit a low-frequency progress heartbeat for stall diagnosis.
            "-stats_period", "5",
            "-progress", "pipe:2",
            self.video_playlist,
        ]

    def _video_audio_command(self) -> list[str]:
        """Extract ASR audio from the same progressive fMP4 cache MPV plays."""
        if not self.video_out:
            raise RuntimeError("MPV fMP4 cache is not enabled")

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-copyts",
            "-i", self.video_playlist,
        ]

        if self.separation_enabled:
            output_rate = SEPARATION_INPUT_RATE
        elif self.deepfilter_enabled:
            output_rate = DEEPFILTER_INPUT_RATE
        else:
            output_rate = SAMPLE_RATE

        audio_filters = self._pitch_filter(output_rate)
        audio_filters.append("ashowinfo")
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
                cmd = [
                    "yt-dlp",
                    "--no-warnings",
                    "--remote-components",
                    "ejs:github",
                    "-f",
                    self.video_format,
                    "-g",
                ]
                cmd += _cookie_args(self.cfg) + [self.source.location]

                self._inputs = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()

        return self._inputs

    def _read_stderr(
        self,
        stream: IO[bytes],
        session: int,
        capture_pts: bool = True,
    ) -> None:
        """Read FFmpeg logs, capture first audio PTS, and filter noisy output."""
        receiver_gone = False
        first_frame_pts: float | None = None
        audio_samples = 0
        next_pts_log_sample = 0
        output_rate = (
            SEPARATION_INPUT_RATE if self.separation_enabled
            else DEEPFILTER_INPUT_RATE if self.deepfilter_enabled
            else SAMPLE_RATE
        )

        for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip()

            # FFmpeg writes one progress=continue/end marker per heartbeat.
            if not capture_pts and line.startswith("progress="):
                with self._video_progress_lock:
                    self._video_progress_at = time.monotonic()
                continue

            if capture_pts:
                with self._proc_lock:
                    current = session == self._session

                    if current and self.first_pts is None:
                        if m := _PTS.search(line):
                            self.first_pts = float(m.group(1))
                            self.first_pts_at = time.monotonic()
                            self.first_pts_ready.set()
                            log.info(
                                "first audio pts %.3f",
                                self.first_pts,
                            )

            if "ashowinfo" in line:
                if capture_pts:
                    pts_match = _PTS.search(line)
                    samples_match = _ASHOWINFO_SAMPLES.search(line)
                    if pts_match and samples_match:
                        frame_pts = float(pts_match.group(1))
                        frame_samples = int(samples_match.group(1))
                        if first_frame_pts is None:
                            first_frame_pts = frame_pts
                            log.info(
                                "audio PTS baseline: %.3fs at %d Hz",
                                first_frame_pts,
                                output_rate,
                            )
                        audio_samples += frame_samples
                        if audio_samples >= next_pts_log_sample:
                            expected_pts = (
                                first_frame_pts + (audio_samples - frame_samples) / output_rate
                            )
                            pts_error = frame_pts - expected_pts
                            log.info(
                                "audio PTS continuity: pts=%.3f "
                                "sample_clock=%.3f drift=%+.3fs",
                                frame_pts,
                                expected_pts,
                                pts_error,
                            )
                            next_pts_log_sample = (
                                (audio_samples // (output_rate * 10)) + 1
                            ) * output_rate * 10
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

            if important:
                log.error("ffmpeg: %s", line)
            elif receiver_gone:
                log.debug("ffmpeg: %s", line)

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

        ratio = 2 ** (semitones / 12.0)
        tempo = 1.0 / ratio

        log.info("ASR audio pitch shift: %.1f semitones", semitones)

        return [
            f"asetrate={sample_rate * ratio:.8f}",
            f"aresample={sample_rate}",
            f"atempo={tempo:.8f}",
        ]

    def _audio_commands(self) -> list[list[str]]:
        """Build audio-only FFmpeg commands."""
        src = self.source

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
        ]

        if not src.live:
            if src.paced:
                ffmpeg.append("-re")

            if src.start_s:
                ffmpeg += ["-ss", f"{src.start_s:.3f}"]

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

        filters = self._pitch_filter(output_rate)
        if filters:
            out = ["-af", ",".join(filters)] + out

        if Path(src.location).exists():
            return [ffmpeg + ["-i", src.location] + out]

        ytdlp = [
            "yt-dlp",
            "-f",
            self.cfg.ytdlp_format,
            "--quiet",
            "--no-warnings",
            "-o",
            "-",
        ]

        if not src.live:
            ytdlp += ["--http-chunk-size", "10M"]

        ytdlp += _cookie_args(self.cfg) + [src.location]

        return [
            ytdlp,
            ffmpeg + ["-i", "pipe:0"] + out,
        ]

    def run(self, out_q: "queue.Queue[AudioChunk | None]") -> None:
        """Read sessions and send audio chunks to the pipeline."""
        try:
            while True:
                self._read_session(out_q)

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
    ) -> None:
        """Read FFmpeg PCM and optionally enhance or separate speech."""
        generation = self.generation
        first_chunk_after_seek = generation > 0

        if self.video_out:
            # One FFmpeg process fetches and muxes the YouTube audio/video
            # into a growing fMP4 HLS event playlist. Both MPV and ASR then
            # consume this same local media timeline.
            self.start_video(start_s=self.source.start_s)
            if not self.wait_video_ready():
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
                return

            self._session += 1
            session = self._session

            try:
                for cmd in cmds:
                    p = subprocess.Popen(
                        cmd,
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE if self.video_out else None,
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
                        p.wait()

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

        first = round(self.source.start_s * SAMPLE_RATE)
        sample = first
        input_frames = 0

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
                # Stop consuming this session as soon as a seek invalidates it.
                if generation != self.generation:
                    break

                if worker:
                    data = stdout.read(SEPARATION_READ_FRAMES * 4)
                else:
                    data = stdout.read(CHUNK_SAMPLES * 2)

                if not data:
                    break

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

        finally:
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
                        p.wait()

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

        if self._cache_temp is not None:
            self._cache_temp.cleanup()
            self._cache_temp = None

    def _terminate_locked(self) -> None:
        """Terminate all registered processes while holding _proc_lock."""
        for p in self._procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass

    def _terminate(self) -> None:
        with self._proc_lock:
            self._terminate_locked()
