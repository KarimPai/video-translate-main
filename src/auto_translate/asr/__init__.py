from ..config import AsrConfig
from .base import ASR
from .qwen_asr import QwenASR


def create_asr(cfg: AsrConfig) -> ASR:
    return QwenASR(cfg)
