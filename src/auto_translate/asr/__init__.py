import logging

from ..config import AsrConfig
from .base import ASR

log = logging.getLogger(__name__)


def create_asr(cfg: AsrConfig) -> ASR:
    # Older local configs used "haas", whose Vulkan backend is disabled in
    # this CUDA-only build. Preserve compatibility by routing those configs
    # through the configured faster-whisper model instead.
    if cfg.backend == "haas":
        log.warning(
            "ASR backend 'haas' is no longer supported; using faster-whisper "
            "with model %s on %s",
            cfg.model,
            cfg.device,
        )
        cfg.backend = "faster-whisper"

    if cfg.backend == "faster-whisper":
        from .faster_whisper import FasterWhisperASR

        return FasterWhisperASR(cfg)

    if cfg.backend == "qwen3-asr":
        from .qwen3 import Qwen3ASR

        return Qwen3ASR(cfg)

    if cfg.backend == "parakeet":
        from .parakeet import ParakeetASR

        return ParakeetASR(cfg)
    if cfg.backend == "haas":
        from .haas import HaasASR

        return HaasASR(cfg)

    raise ValueError(f"unknown ASR backend: {cfg.backend!r}")
