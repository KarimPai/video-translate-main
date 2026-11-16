"""Data passed between pipeline stages.

Media timestamps (`start_s` / `end_s`) are seconds since ingest start, computed from the
PCM sample count. Wall-clock times live only in `Timings` and are used for latency logging.
"""

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Timings:
    """time.monotonic() checkpoints for one utterance (for latency logs only)."""

    audio_read: float = 0.0  # (estimated) the utterance's last sample arrived from ffmpeg
    segmented: float = 0.0  # VAD decided the utterance was complete
    asr_start: float = 0.0
    asr_end: float = 0.0
    mt_start: float = 0.0  # translation
    mt_end: float = 0.0


@dataclass
class AudioChunk:
    start_sample: int
    pcm: np.ndarray  # float32 mono 16 kHz, values in [-1, 1]
    read_at: float  # time.monotonic() when it was read from ffmpeg
    reset: bool = False  # discontinuity (vod jump): audio continues at start_sample, pcm is empty
    generation: int = 0

@dataclass
class AudioProgress:
    sample: int  # processed audio position at 16 kHz
    reset: bool = False  # new VOD seek position; replace the previous watermark
    generation: int = 0


@dataclass
class Utterance:
    start_s: float
    end_s: float
    audio: np.ndarray  # float32 mono 16 kHz
    probe: bool = False  # VAD heard no speech; send fallback audio to Haas
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None  # first FireRed speech-positive frame; excludes VAD pre-roll
    vad_confidence: float | None = None  # mean VAD speech confidence

@dataclass
class Transcript:
    start_s: float
    end_s: float
    ja: str
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None

    diagnostic_session: str | None = None


@dataclass
class Subtitle:
    start_s: float
    end_s: float
    ja: str
    en: str  # "" when translation is off
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None
    diagnostic_session: str | None = None
