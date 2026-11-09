"""Whisper Large-v3 direct Japanese speech-to-English translation via faster-whisper.

This is one speech model, not a Japanese ASR + translation LLM cascade.
FireRedVAD decides where each utterance begins/ends; Whisper handles JP->EN.
"""

from __future__ import annotations

import logging

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
        # IMPORTANT: segments is lazy; inference occurs upon iteration.
        return " ".join(segment.text.strip() for segment in segments
                        if segment.text.strip()).strip()

    def close(self) -> None:
        """Drop our model reference; no llama-server or subprocess is used."""
        self._model = None
