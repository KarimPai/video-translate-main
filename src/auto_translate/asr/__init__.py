from ..config import AsrConfig
from .base import ASR
from .haas import HaasASR


def create_asr(cfg: AsrConfig) -> ASR:
    return HaasASR(cfg)
