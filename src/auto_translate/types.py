"""Timestamped audio, VAD utterances, and English-only subtitle packets."""

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Timings:
    audio_read: float = 0.0
    segmented: float = 0.0
    # First interval: Whisper Japanese ASR; second: PLaMo translation.
    asr_start: float = 0.0
    asr_end: float = 0.0
    mt_start: float = 0.0
    mt_end: float = 0.0


@dataclass
class AudioChunk:
    start_sample: int
    pcm: np.ndarray
    read_at: float
    reset: bool = False
    generation: int = 0


@dataclass
class AudioProgress:
    sample: int
    reset: bool = False
    generation: int = 0


@dataclass
class Utterance:
    start_s: float
    end_s: float
    audio: np.ndarray
    probe: bool = False
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None
    speaker_handoff: bool = False
    speaker_split: bool = False
    vad_confidence: float | None = None


@dataclass
class Transcript:
    start_s: float
    end_s: float
    ja: str
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None
    speaker_handoff: bool = False


@dataclass
class Subtitle:
    start_s: float
    end_s: float
    ja: str  # recognized Japanese transcript
    en: str
    timings: Timings = field(default_factory=Timings)
    generation: int = 0
    speech_start_s: float | None = None
    speaker_handoff: bool = False
