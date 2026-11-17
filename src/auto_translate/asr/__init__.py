"""The application's single local Japanese ASR backend."""
from ..config import AsrConfig
from .base import ASR
from .voxtral import VoxtralASR


def create_asr(cfg: AsrConfig) -> ASR:
    return VoxtralASR(cfg)
