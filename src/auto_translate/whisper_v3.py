"""Japanese transcription-only Whisper Large-v3 inference.

The model is loaded once. The public decode_audio() method exposes Whisper's
own evidence about reliability for optional one-pass retry and diagnostics.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from .config import WhisperConfig
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperDecode:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    decoded_segments: int = 0
    discarded_segments: int = 0
    duration_s: float = 0.0
    temperature: float = 0.0

    @property
    def suspicious(self) -> bool:
        # No speech at all is not automatically a failed translation.
        if self.discarded_segments:
            return True
        if not self.text or self.avg_logprob is None:
            return False
        return self.avg_logprob < -0.75 or (
            self.compression_ratio is not None and
            self.compression_ratio > 2.0
        )


class WhisperLargeV3:
    """One reusable GPU CTranslate2 model, no remote services."""

    def __init__(self, cfg: WhisperConfig, *, model: object | None = None) -> None:
        self.cfg = cfg
        if cfg.language != "ja" or cfg.task != "transcribe":
            raise ValueError("Japanese ASR requires language='ja' and task='transcribe'")
        if cfg.beam_size < 1:
            raise ValueError("Whisper beam_size must be at least 1")
        if model is not None:
            self._model = model
            return

        # __main__ initializes Windows CUDA DLL directories before importing.
        from faster_whisper import WhisperModel

        log.info("Loading Whisper %s on %s (%s)", cfg.model, cfg.device, cfg.compute_type)
        try:
            self._model = WhisperModel(
                cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                download_root=cfg.download_root or None,
                num_workers=1,
            )
        except Exception as exc:
            raise RuntimeError(
                "Whisper Large-v3 failed to load. Verify RTX 3070 CUDA libraries "
                "(CUDA 12 cuBLAS and cuDNN 9), VRAM and faster-whisper installation: "
                + str(exc)
            ) from exc
        log.info("Whisper Large-v3 ready: Japanese audio -> Japanese transcription only")

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Japanese ASR; translation must be performed separately by PLaMo."""
        return self.decode_audio(audio).text

    def translate_audio(self, audio: np.ndarray) -> str:
        """Legacy API retained for test doubles; now returns JAPANESE only."""
        return self.transcribe_audio(audio)

    def decode_audio(
        self,
        audio: np.ndarray,
        *,
        prompt: str | None = None,
        hotwords: str | None = None,
        temperature: float | None = None,
    ) -> WhisperDecode:
        """Transcribe one bounded Japanese utterance (no native translation).

        The caller owns the utterance's source timestamps; adding context audio
        must never shift those timestamps or bypass duplicate-text validation.
        """
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            raise ValueError("cannot translate empty audio")
        if not np.all(np.isfinite(pcm)):
            raise ValueError("audio contains non-finite samples")

        assert SAMPLE_RATE == 16000
        used_temperature = self.cfg.temperature if temperature is None else temperature
        started = time.monotonic()
        segments, _ = self._model.transcribe(
            np.ascontiguousarray(pcm),
            language="ja",
            task="transcribe",
            beam_size=self.cfg.beam_size,
            best_of=1,
            repetition_penalty=self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=used_temperature,
            # Each FireRedVAD utterance is independent. Never allow an old
            # config.toml to reintroduce hallucination feedback across windows.
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            initial_prompt=prompt or None,
            hotwords=hotwords or None,
            word_timestamps=False,
        )

        # The faster-whisper segments generator runs inference on iteration.
        texts: list[str] = []
        scores: list[tuple[int, float]] = []
        ratios: list[float] = []
        nospeech: list[float] = []
        rejected_nospeech: list[float] = []
        total = discarded = 0
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            total += 1
            if self.cfg.reject_low_confidence and self._unreliable(segment):
                discarded += 1
                # Retain the rejected segment's silence evidence; otherwise
                # the empty decode looks like an ordinary ASR retry candidate.
                rejected_prob = getattr(segment, "no_speech_prob", None)
                if rejected_prob is not None and math.isfinite(float(rejected_prob)):
                    rejected_nospeech.append(float(rejected_prob))
                log.info(
                    "Discarded uncertain Whisper segment: no_speech=%s "
                    "avg_logprob=%s compression=%s text=%r",
                    getattr(segment, "no_speech_prob", None),
                    getattr(segment, "avg_logprob", None),
                    getattr(segment, "compression_ratio", None),
                    text[:100],
                )
                continue
            texts.append(text)
            lp = getattr(segment, "avg_logprob", None)
            ratio = getattr(segment, "compression_ratio", None)
            ns = getattr(segment, "no_speech_prob", None)
            if lp is not None and math.isfinite(float(lp)):
                scores.append((max(1, len(text)), float(lp)))
            if ratio is not None and math.isfinite(float(ratio)):
                ratios.append(float(ratio))
            if ns is not None and math.isfinite(float(ns)):
                nospeech.append(float(ns))
        avg = (
            sum(weight * score for weight, score in scores) /
            sum(weight for weight, _ in scores)
        ) if scores else None

        return WhisperDecode(
            text=(
                "".join(texts) if any(
                    "\u3040" <= c <= "\u9fff"
                    for part in texts for c in part
                ) else " ".join(texts)
            ).strip(),
            avg_logprob=avg,
            no_speech_prob=(
                max(nospeech) if nospeech else
                max(rejected_nospeech) if rejected_nospeech else None
            ),
            compression_ratio=max(ratios) if ratios else None,
            decoded_segments=total,
            discarded_segments=discarded,
            duration_s=time.monotonic() - started,
            temperature=float(used_temperature),
        )

    def _unreliable(self, segment: object) -> bool:
        """Use model confidence rather than a hard-coded profanity/phrase list."""
        cfg = self.cfg
        avg = getattr(segment, "avg_logprob", None)
        ratio = getattr(segment, "compression_ratio", None)
        silent = getattr(segment, "no_speech_prob", None)
        for value in (avg, ratio, silent):
            if value is not None and not math.isfinite(float(value)):
                return True
        if avg is not None and float(avg) < cfg.min_avg_logprob:
            return True
        if ratio is not None and float(ratio) > cfg.max_compression_ratio:
            return True
        if (silent is not None and avg is not None and
                float(silent) >= cfg.noisy_segment_no_speech_prob and
                float(avg) < cfg.noisy_segment_min_avg_logprob):
            return True
        return False

    def close(self) -> None:
        self._model = None
