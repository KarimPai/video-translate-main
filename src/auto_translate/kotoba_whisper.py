"""Japanese Kotoba-Whisper v2.2, CTranslate2 inference for live subtitles.

Official checkpoint: kotoba-tech/kotoba-whisper-v2.2
Community conversion: RoachLin/kotoba-whisper-v2.2-faster

The community CTranslate2 conversion avoids Transformers remote-code,
punctuation and diarization dependencies; punctuation is model-decoded,
and the project's existing FireRedVAD handles speech segmentation.
This test branch does NOT claim the official v2.2 custom pipeline's
punctuation/diarization stack. Download stays on Hugging Face; inference
is entirely local. The same decoded-audio API as the Whisper-v3 baseline
keeps JP transcript timing, confidence filters and PLaMo unchanged.
"""
from __future__ import annotations

from .config import WhisperConfig
from .whisper_v3 import WhisperDecode, WhisperLargeV3

KOTOBA_OFFICIAL_ID = "kotoba-tech/kotoba-whisper-v2.2"
KOTOBA_CT2_ID = "RoachLin/kotoba-whisper-v2.2-faster"


class KotobaWhisperASR(WhisperLargeV3):
    """Kotoba v2.2 CTranslate2 model, NEVER OpenAI Whisper large-v3."""

    def __init__(self, cfg: WhisperConfig, *, model: object | None = None) -> None:
        if cfg.model != KOTOBA_CT2_ID:
            raise ValueError(
                "Kotoba test branch requires the verified Kotoba v2.2 "
                "CTranslate2 conversion: " + KOTOBA_CT2_ID
            )
        super().__init__(cfg, model=model)
