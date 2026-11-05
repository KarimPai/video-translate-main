from ..config import AsrConfig
from .base import ASR


def create_asr(cfg: AsrConfig) -> ASR:
    if cfg.backend == "haas":
        from .haas import HaasASR

        return HaasASR(cfg)

    if cfg.backend == "faster-whisper":
        from .faster_whisper import FasterWhisperASR

        return FasterWhisperASR(cfg)

    raise ValueError(
        f"unsupported ASR backend {cfg.backend!r}; use 'haas' or 'faster-whisper'"
    )
