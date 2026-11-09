"""Japanese ASR backend for the ReazonSpeech k2 v2 experiment."""

from ..config import AsrConfig
from .base import ASR
from .reazon import ReazonK2ASR


def create_asr(cfg: AsrConfig) -> ASR:
    return ReazonK2ASR(cfg)
