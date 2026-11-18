"""The application's single local Japanese ASR backend."""
from ..config import AsrConfig
from .buzz import BuzzASR


def create_asr(cfg: AsrConfig) -> BuzzASR:
    return BuzzASR(cfg)
