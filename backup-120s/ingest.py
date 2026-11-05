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
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import IngestConfig, SeparationConfig, DeepFilterConfig
from .types import SAMPLE_RATE, AudioChunk

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 4096
SEPARATION_READ_FRAMES = 8192
SEPARATION_INPUT_RATE = 44100
DEEPFILTER_INPUT_RATE = 48000

_PTS = re.compile(r"pts_time:(-?[\d.]+)")


@dataclass
class Source:
    location: str
    live: bool
    start_s: float = 0.0
    paced: bool = True
    duration: float | None = None


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
    """Return live status and duration."""
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--skip-download",
        "--print",
        "%(live_status)s|%(duration)s",
    ]
    cmd += _cookie_args(cfg) + [url]

    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    status, _, duration = out.partition("|")

    return (
        status,
        float(duration) if duration not in ("", "NA", "None") else None,
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

    if mode == "auto":
        if is_file:
            mode = "vod"
        else:
            status, duration = probe(location, cfg)
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
        self.video_out = video_out
        self.video_format = video_format
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
        self._inputs: list[str] | None = None
        self._stopped = False
        self._session = 0

        self.first_pts: float | None = None
        self.first_pts_at = 0.0
        self.first_pts_ready = threading.Event()

        self._restart_at: float | None = None
        self._go = threading.Event()

    def request_restart(self, start_s: float) -> None:
        """Stop the current FFmpeg session and prepare a VOD jump."""
        self._go.clear()
        self.first_pts = None
        self._restart_at = start_s
        self.first_pts_ready.clear()
        self._terminate()

    def go(self) -> None:
        self._go.set()

    def _commands(self) -> list[list[str]]:
        if self.video_out:
            return [
                self._video_command(),
                self._video_audio_command(),
            ]
        return self._audio_commands()

    def _video_command(self) -> list[str]:
        """Build the paced FFmpeg stream sent to MPV."""
        src = self.source
        inputs = self._resolve_video_inputs()

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-copyts",
        ]

        for url in inputs:
            if not src.live:
                ffmpeg += ["-readrate", "1"]

                if self.burst_s:
                    ffmpeg += [
                        "-readrate_initial_burst",
                        f"{self.burst_s:g}",
                    ]

                if src.start_s:
                    ffmpeg += ["-ss", f"{src.start_s:.3f}"]

            ffmpeg += ["-http_persistent", "0", "-i", url]

        audio = f"{len(inputs) - 1}:a:0"

        assert self.video_out is not None

        return ffmpeg + [
            "-map", "0:v:0?",
            "-map", audio,
            "-c", "copy",
            "-f", "mpegts",
            "-muxdelay", "0",
            "-muxpreload", "0",
            self.video_out,
        ]

    def _video_audio_command(self) -> list[str]:
        """Build the unpaced FFmpeg audio-processing stream."""
        src = self.source
        inputs = self._resolve_video_inputs()

        ffmpeg = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-copyts",
        ]

        for url in inputs:
            if not src.live and src.start_s:
                ffmpeg += ["-ss", f"{src.start_s:.3f}"]

            ffmpeg += ["-http_persistent", "0", "-i", url]

        audio = f"{len(inputs) - 1}:a:0"
        ffmpeg += ["-map", audio, "-af", "ashowinfo"]

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

        for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip()
            current = session == self._session

            if capture_pts and current and self.first_pts is None:
                if m := _PTS.search(line):
                    self.first_pts = float(m.group(1))
                    self.first_pts_at = time.monotonic()
                    self.first_pts_ready.set()
                    log.info("first audio pts %.3f", self.first_pts)

            if "ashowinfo" in line:
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
            out = [
                "-vn", "-ac", "2",
                "-ar", str(SEPARATION_INPUT_RATE),
                "-f", "s16le", "pipe:1",
            ]
        elif self.deepfilter_enabled:
            out = [
                "-vn", "-ac", "1",
                "-ar", str(DEEPFILTER_INPUT_RATE),
                "-f", "s16le", "pipe:1",
            ]
        else:
            out = [
                "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "pipe:1",
            ]

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

                if self._stopped or self._restart_at is None:
                    break

                self.source = replace(
                    self.source,
                    start_s=self._restart_at,
                )
                self._restart_at = None

                out_q.put(
                    AudioChunk(
                        round(self.source.start_s * SAMPLE_RATE),
                        np.zeros(0, dtype=np.float32),
                        time.monotonic(),
                        reset=True,
                    )
                )

                while not self._go.wait(0.2):
                    if self._stopped:
                        return

        finally:
            out_q.put(None)
            self.stop()

    def _read_session(
        self,
        out_q: "queue.Queue[AudioChunk | None]",
    ) -> None:
        """Read FFmpeg PCM and optionally enhance or separate speech."""
        cmds = self._commands()
        procs: list[subprocess.Popen] = []

        if self.video_out:
            video_proc = subprocess.Popen(
                cmds[0],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            procs.append(video_proc)

            audio_proc = subprocess.Popen(
                cmds[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(audio_proc)
        else:
            stdin = None

            for cmd in cmds:
                p = subprocess.Popen(
                    cmd,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=None,
                )

                if stdin is not None:
                    stdin.close()

                stdin = p.stdout
                procs.append(p)

        self._procs = procs
        self._session += 1

        if self.video_out:
            threading.Thread(
                target=self._read_stderr,
                args=(procs[0].stderr, self._session, False),
                name="ffmpeg-video-log",
                daemon=True,
            ).start()

            threading.Thread(
                target=self._read_stderr,
                args=(procs[1].stderr, self._session),
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
                )

                log.info(
                    "DeepFilterNet enabled: mono %d Hz input",
                    DEEPFILTER_INPUT_RATE,
                )

            while True:
                if worker:
                    data = stdout.read(SEPARATION_READ_FRAMES * 4)
                else:
                    data = stdout.read(CHUNK_SAMPLES * 2)

                if not data:
                    break

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

                else:
                    pcm = (
                        np.frombuffer(data, dtype="<i2")
                        .astype(np.float32)
                        / 32768.0
                    )

                    out_q.put(
                        AudioChunk(
                            sample,
                            pcm,
                            time.monotonic(),
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
        self._stopped = True
        self._go.set()
        self._terminate()

    def _terminate(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()