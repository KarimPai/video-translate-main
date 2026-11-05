import logging

from ..config import AsrConfig
from .base import ASR

log = logging.getLogger(__name__)


def create_asr(cfg: AsrConfig) -> ASR:
    # Migrate old local configs without retaining the obsolete Haas backend.
    if cfg.backend == "haas":
        log.warning(
            "ASR backend 'haas' is obsolete; using faster-whisper "
            "with model %s on %s",
            cfg.model,
            cfg.device,
        )
        cfg.backend = "faster-whisper"

    if cfg.backend == "faster-whisper":
        from .faster_whisper import FasterWhisperASR

        return FasterWhisperASR(cfg)

    raise ValueError(
        f"unsupported ASR backend {cfg.backend!r}; use 'faster-whisper'"
    )
