"""Japanese ASR backend for the ReazonSpeech k2 v2 experiment."""

from ..config import AsrConfig
from .base import ASR
from .reazon_ipc import ReazonWorkerASR


def create_asr(cfg: AsrConfig) -> ASR:
    return ReazonWorkerASR(cfg)
