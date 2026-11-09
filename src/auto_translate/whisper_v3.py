"""Whisper Large-v3 direct Japanese speech-to-English translation via faster-whisper.

This is one speech model, not a Japanese ASR + translation LLM cascade.
FireRedVAD decides where each utterance begins/ends; Whisper handles JP->EN.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .config import WhisperConfig
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)


class WhisperLargeV3:
    """Load one CTranslate2 Whisper Large-v3 model and reuse it for all utterances."""

    def __init__(self, cfg: WhisperConfig, *, model: object | None = None) -> None:
        self.cfg = cfg
        if cfg.language != "ja" or cfg.task != "translate":
            raise ValueError("Whisper direct AST requires language='ja' and task='translate'")
        if cfg.beam_size < 1:
            raise ValueError("Whisper beam_size must be at least 1")
        if model is not None:
            self._model = model
            return

        # Import only after Windows CUDA DLL directories have been initialized
        # by the application entry point.
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
        log.info("Whisper Large-v3 ready: Japanese audio -> English, no second model")

    def translate_audio(self, audio: np.ndarray) -> str:
        """Decode one 16-kHz mono utterance directly to English.

        Every call is stateless across utterances; VAD already stripped
        non-speech so faster-whisper's additional Silero VAD is disabled.
        """
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            raise ValueError("cannot translate empty audio")
        if not np.all(np.isfinite(pcm)):
            raise ValueError("audio contains non-finite samples")

        # PCM arrives from FFmpeg/FireRedVAD at exactly SAMPLE_RATE.
        # For numpy input, faster-whisper expects this 16-kHz waveform.
        assert SAMPLE_RATE == 16000
        segments, _ = self._model.transcribe(
            np.ascontiguousarray(pcm),
            language="ja",
            task="translate",
            beam_size=self.cfg.beam_size,
            best_of=1,
            repetition_penalty=self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            initial_prompt=None,
            hotwords=None,
            word_timestamps=False,
        )
        # IMPORTANT: faster-whisper segments are lazy; inference occurs only
        # during iteration. Internal no_speech detection checks confidence as
        # well, so we also reject *final* poor-confidence/noisy segments.
        text_parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if self.cfg.reject_low_confidence and self._unreliable(segment):
                log.info(
                    "Discarded uncertain Whisper segment: no_speech=%s "
                    "avg_logprob=%s compression=%s text=%r",
                    getattr(segment, "no_speech_prob", None),
                    getattr(segment, "avg_logprob", None),
                    getattr(segment, "compression_ratio", None),
                    text[:100],
                )
                continue
            text_parts.append(text)
        return " ".join(text_parts).strip()

    def _unreliable(self, segment: object) -> bool:
        """Use decoder evidence, not a blanket ban on particular words."""
        cfg = self.cfg
        avg = getattr(segment, "avg_logprob", None)
        ratio = getattr(segment, "compression_ratio", None)
        silent = getattr(segment, "no_speech_prob", None)
        # Tests and third-party backends can omit scores. Do not invent them.
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
        """Drop our model reference; no llama-server or subprocess is used."""
        self._model = None
