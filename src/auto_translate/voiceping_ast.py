"""VoicePing Qwen3-ASR direct Japanese-audio-to-English speech translation.

No separate Japanese ASR pass, machine translator, word aligner or quantizer.
FireRedVAD supplies timestamped PCM. The model outputs English text only.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .config import VoicePingAstConfig
from .hololive_names import TALENTS
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)


def _name_key(text: str) -> str:
    """Exact comparison after removing punctuation/case, not fuzzy name matching."""
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold()))


def build_ast_context(cfg: VoicePingAstConfig) -> str:
    """Use bounded name hints and style, never past translated subtitles."""
    if not cfg.context_enabled:
        return ""
    parts = []
    if cfg.japanese_source_only:
        # The fine-tuned model still receives language="English": that is the
        # target output language. Context is an unverified source preference,
        # not an acoustic language-ID gate.
        parts.append(
            "Translate Japanese speech to English only. For English-only speech, "
            "laughter or game sounds, output nothing. Keep English words "
            "embedded in Japanese sentences."
        )
    if cfg.concise_context:
        parts.append(
            "Use concise, natural English. Preserve meaning, negation and "
            "proper names. Do not invent details or repeat phrases unnecessarily."
        )
    names: list[str] = []
    seen = set()
    for candidate in [*cfg.detected_names, *cfg.name_hints]:
        if not isinstance(candidate, str):
            continue
        name = " ".join(candidate.split())
        if (2 <= len(name) <= 36 and len(name.split()) <= 3
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '\-]*", name)
                and name.casefold() not in seen):
            seen.add(name.casefold())
            names.append(name)
        if len(names) >= 8:
            break
    if names:
        prefix = "Relevant names (only when actually spoken): "
        available = 450 - len(" ".join(parts)) - len(prefix) - 2
        selected = []
        for name in names:
            candidate = ", ".join([*selected, name])
            if len(candidate) > available:
                break  # never truncate a name mid-word
            selected.append(name)
        if selected:
            parts.append(prefix + ", ".join(selected) + ".")
    return " ".join(parts)


class VoicePingAST:
    """One persistent BF16 CUDA model, independent utterances, batch size one."""

    def __init__(self, cfg: VoicePingAstConfig, *, model: object | None = None) -> None:
        self.cfg = cfg
        self.context = build_ast_context(cfg)
        self._standalone_names = {
            _name_key(name) for name in
            [*(talent.hint for talent in TALENTS), *cfg.detected_names, *cfg.name_hints]
            if isinstance(name, str) and name.strip()
        }
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
        log.info("VoicePing native context: %s", self.context or "off")

    def is_standalone_name(self, text: str) -> bool:
        """Candidate single-name hallucination, only when text context was used."""
        key = _name_key(text.strip())
        return bool(self.context and key and key in self._standalone_names)

    def translate_audio(self, audio: np.ndarray, *, use_context: bool = True) -> str:
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
                context=self.context if use_context else "",
            )
        if not output:
            return ""
        return str(getattr(output[0], "text", "") or "").strip()

    def close(self) -> None:
        self._model = None
