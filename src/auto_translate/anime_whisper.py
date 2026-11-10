"""Anime Whisper Japanese-only ASR adapter for the existing PLaMo pipeline.

The developer explicitly warns that initial prompts cause hallucinations.
Do not pass previous subtitles, initial_prompt, prompt_ids, or hotwords.
All input clips are bounded, independently decoded 16-kHz mono speech.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .config import WhisperConfig
from .types import SAMPLE_RATE
from .whisper_v3 import WhisperDecode

log = logging.getLogger(__name__)
MODEL_ID = "litagin/anime-whisper"


class AnimeWhisperASR:
    """One local Hugging Face Transformers speech-to-text model per session."""

    def __init__(
        self,
        cfg: WhisperConfig,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.cfg = cfg
        if cfg.model != MODEL_ID:
            raise ValueError(f"Anime Whisper experiment supports only {MODEL_ID}")
        if cfg.language != "ja" or cfg.task != "transcribe":
            raise ValueError("Anime Whisper must use Japanese transcription, not translation")
        if cfg.device not in ("cuda", "cpu"):
            raise ValueError("Anime Whisper device must be 'cuda' or 'cpu'")
        if cfg.compute_type not in ("float16", "float32"):
            raise ValueError("Anime Whisper compute_type must be float16 or float32")
        if cfg.beam_size < 1:
            raise ValueError("Anime Whisper beam_size must be >= 1")
        if not 1 <= cfg.max_new_tokens <= 400:
            raise ValueError("Anime Whisper max_new_tokens must be 1..400")
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor, or neither")
        self.device = cfg.device
        if model is not None:
            self._model = model
            self._processor = processor
            return

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Anime Whisper requires CUDA, but PyTorch CUDA is unavailable")
        dtype = (
            torch.float16 if cfg.device == "cuda" and cfg.compute_type == "float16"
            else torch.float32
        )
        log.info("Loading %s via Transformers on %s (%s)", MODEL_ID, cfg.device, dtype)
        try:
            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID, dtype=dtype,
            ).to(cfg.device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed loading Anime Whisper (about 3 GB initial checkpoint download). "
                "Check model access, compatible Transformers/PyTorch, and GPU VRAM: "
                + str(exc)
            ) from exc
        log.info(
            "Anime Whisper ready: Japanese ASR only; initial prompts/hotwords disabled"
        )

    def decode_audio(
        self,
        audio: np.ndarray,
        *,
        prompt: str | None = None,
        hotwords: str | None = None,
        temperature: float | None = None,
        beam_size: int | None = None,
    ) -> WhisperDecode:
        """Transcribe bounded Japanese audio without any textual priming."""
        import torch

        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return WhisperDecode("")
        if not np.isfinite(pcm).all():
            raise ValueError("Anime Whisper received non-finite audio samples")
        if pcm.size > 30 * SAMPLE_RATE:
            raise ValueError("Anime Whisper input must not exceed 30 seconds")
        pcm = np.ascontiguousarray(np.clip(pcm, -1.0, 1.0))
        if float(np.max(np.abs(pcm))) <= 1.0 / 32768:
            return WhisperDecode("")
        if prompt or hotwords:
            log.debug("Ignoring ASR prompt/hotwords: Anime Whisper must remain unprimed")

        t = self.cfg.temperature if temperature is None else float(temperature)
        beams = self.cfg.beam_size if beam_size is None else int(beam_size)
        if not 1 <= beams <= 16:
            raise ValueError("Anime Whisper beam_size must be 1..16")
        if not 0.0 <= t <= 1.0:
            raise ValueError("Anime Whisper temperature must be 0..1")
        start = time.monotonic()
        inputs = self._processor(
            pcm, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True,
        )
        dtype = torch.float16 if (
            self.device == "cuda" and self.cfg.compute_type == "float16"
        ) else torch.float32
        features = inputs["input_features"].to(device=self.device, dtype=dtype)
        # The model pads Whisper features to a 30s window even when the speech
        # is only a second long. Limit decoder tokens for short cues so it has
        # less opportunity to invent extra sentences after the audio ends.
        # Maintain a generous budget for rapid Japanese and longer phrases.
        max_tokens = self.cfg.max_new_tokens
        if getattr(self.cfg, "adaptive_max_tokens", True):
            duration_s = pcm.size / SAMPLE_RATE
            max_tokens = min(max_tokens, max(64, int(duration_s * 30) + 32))
        arguments = {
            "input_features": features,
            "language": "japanese",
            "task": "transcribe",
            "num_beams": beams,
            "max_new_tokens": max_tokens,
            "no_repeat_ngram_size": self.cfg.no_repeat_ngram_size,
            "repetition_penalty": self.cfg.repetition_penalty,
            "do_sample": t > 0.0,
        }
        collect_scores = bool(getattr(self.cfg, "collect_token_scores", True))
        if collect_scores:
            arguments["return_dict_in_generate"] = True
            arguments["output_scores"] = True
        if t > 0.0:
            arguments["temperature"] = t
        if "attention_mask" in inputs:
            arguments["attention_mask"] = inputs["attention_mask"].to(self.device)
        with torch.inference_mode():
            predicted = self._model.generate(**arguments)
        # Model outputs may be a tensor or GenerationOutput, depending on flags.
        sequences = getattr(predicted, "sequences", predicted)
        # Critical Transformers 5.x Whisper beam-search workaround:
        # Whisper's outputs.scores have ALREADY been selected by beam_indices.
        # compute_transition_scores(sequences, scores, beam_indices) would
        # index the beams a second time and can trigger a sticky CUDA
        # ScatterGatherKernel out-of-bounds assertion (HF issue #48621).
        # Read the finished beam's sequences_scores directly instead.
        # This is the length-penalized beam score, not a calibrated probability.
        avg_logprob = None
        if collect_scores:
            sequence_scores = getattr(predicted, "sequences_scores", None)
            if sequence_scores is not None:
                values = sequence_scores.detach().float().reshape(-1)
                if values.numel() == 1 and bool(torch.isfinite(values[0])):
                    avg_logprob = float(values[0].item())
                else:
                    log.debug(
                        "Anime Whisper beam sequence score unavailable "
                        "(unexpected count or non-finite result)"
                    )
        decoded = self._processor.batch_decode(
            sequences, skip_special_tokens=True,
        )
        text = str(decoded[0] if decoded else "").strip()
        return WhisperDecode(
            text=text,
            avg_logprob=avg_logprob,
            duration_s=time.monotonic() - start,
            temperature=t,
            decoded_segments=1 if text else 0,
        )

    def transcribe_audio(self, audio: np.ndarray) -> str:
        return self.decode_audio(audio).text

    def translate_audio(self, audio: np.ndarray) -> str:
        """Compatibility method: output remains Japanese, never English."""
        return self.transcribe_audio(audio)

    def close(self) -> None:
        # This process is about to exit; OS/PyTorch release its GPU allocations.
        # CUDA_ERROR_ASSERT is sticky: calling empty_cache() after a kernel fault
        # triggers a second exception and hides the original ASR/VAD traceback.
        # Never issue additional CUDA commands during failed-device shutdown.
        self._model = None
        self._processor = None
