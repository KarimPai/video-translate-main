"""VoicePing Qwen3-ASR direct Japanese-audio-to-English speech translation.

No separate Japanese ASR pass, machine translator, word aligner or quantizer.
FireRedVAD supplies timestamped PCM. The model outputs English text only.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import VoicePingAstConfig
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)


class VoicePingAST:
    """One persistent BF16 CUDA model, independent utterances, batch size one."""

    def __init__(self, cfg: VoicePingAstConfig, *, model: object | None = None) -> None:
        self.cfg = cfg
        if cfg.dtype != "bfloat16":
            raise ValueError("This AST experiment uses BF16 only; quantization is disabled")
        if cfg.max_inference_batch_size != 1:
            raise ValueError("Realtime AST must use inference batch size 1")
        if not 1 <= cfg.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if model is not None:
            self._model = model
            return

        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError("VoicePing BF16 AST requires an NVIDIA CUDA GPU")
        log.info("Loading VoicePing JA->EN AST %s on %s (BF16)", cfg.model, cfg.device)
        try:
            self._model = Qwen3ASRModel.from_pretrained(
                cfg.model,
                dtype=torch.bfloat16,
                device_map=cfg.device,
                max_inference_batch_size=1,
                max_new_tokens=cfg.max_new_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load VoicePing AST in BF16. Check CUDA, the model "
                "download and free VRAM; no automatic quantization is enabled."
            ) from exc
        log.info("VoicePing AST ready: Japanese speech -> English text; no second translator")

    def translate_audio(self, audio: np.ndarray) -> str:
        pcm = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not pcm.size or not np.isfinite(pcm).all():
            return ""
        if pcm.size > 30 * SAMPLE_RATE:
            raise ValueError("VoicePing AST supports at most 30 seconds per utterance")
        # Qwen3-ASR's 'language' specifies the OUTPUT language in this
        # fine-tune. This must be English, unlike Whisper's language='ja'.
        import torch

        with torch.inference_mode():
            output = self._model.transcribe(
                audio=(pcm, SAMPLE_RATE),
                language="English",
            )
        if not output:
            return ""
        return str(getattr(output[0], "text", "") or "").strip()

    def close(self) -> None:
        self._model = None
