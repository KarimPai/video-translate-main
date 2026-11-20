"""Whisper Large-v3 Turbo ASR via faster-whisper (CTranslate2).

Whisper is a multilingual speech recognition model. The Large-v3 Turbo variant
is optimized for speed while maintaining accuracy. Runs in-process via the
faster-whisper library with CTranslate2 backend, supporting both CPU and CUDA.

For Japanese transcription, the model is configured with ``language="ja"`` and
``task="transcribe"`` to force Japanese output.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import deque

import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from .context_echo import speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_WHISPER_MODEL = "Systran/faster-whisper-large-v3-turbo"


class WhisperError(RuntimeError):
    """The Whisper ASR model failed repeatedly; transcription cannot continue."""


class WhisperASR:
    """Whisper Large-v3 Turbo ASR via faster-whisper in-process.

    Uses the faster-whisper library with CTranslate2 backend for efficient
    inference on CPU or CUDA. No external server process is needed.
    """

    def __init__(self, cfg: AsrConfig, context: str = "") -> None:
        self.cfg = cfg
        self.context = context.strip()[: cfg.max_context_chars]
        self._recent: deque[str] = deque(maxlen=cfg.recent_context_lines)
        self._accepted_keys: deque[str] = deque(maxlen=32)
        self.last_match_score = 0.0
        self.last_raw_transcript = ""
        self.last_rejection: str | None = None
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._failures = 0
        self.last_timing: dict | None = None

        # Determine device and compute type
        device = "cpu"
        compute_type = "int8"
        if cfg.cuda_device is not None:
            device = "cuda"
            compute_type = "float16"

        # Model path: use configured path or default Hugging Face model
        model_path = cfg.model_path or _WHISPER_MODEL

        # Load the model — this may download from Hugging Face on first use
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=cfg.threads,
                num_workers=1,
            )
        except ImportError:
            raise WhisperError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper"
            )
        except Exception as exc:
            raise WhisperError(
                f"Failed to load Whisper model '{model_path}': {exc}"
            ) from exc

        update_metrics(workers={"asr": f"ready (Whisper Turbo, {device})"})
        log.info(
            "Whisper Large-v3 Turbo ready: model=%s device=%s compute=%s threads=%d",
            model_path, device, compute_type, cfg.threads,
        )

    # ------------------------------------------------------------------ #
    # ASRBackend protocol
    # ------------------------------------------------------------------ #

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        """Transcribe 16 kHz mono float32 audio to Japanese text."""
        with self._state_lock:
            generation = self._generation
            self.last_raw_transcript = ""
            self.last_rejection = None

        duration_s = len(audio) / 16000
        if duration_s < 0.05:
            return ""
        if duration_s > 60:
            raise ValueError("ASR utterance exceeds 60 seconds; check VAD segmentation")
        if not len(audio):
            return ""

        started = time.monotonic()
        with self._inference_lock:
            if generation != self._generation or self._closed:
                return ""
            try:
                text = self._transcribe_impl(audio, probe)
            except Exception as exc:
                self._failures += 1
                self.last_rejection = "error"
                log.warning("Whisper request failed (%d consecutive): %s", self._failures, exc)
                if self._failures >= 3:
                    raise WhisperError(
                        f"Whisper failed three consecutive utterances: {exc}"
                    ) from exc
                return ""

        self._failures = 0
        with self._state_lock:
            if generation != self._generation:
                return ""
            self.last_raw_transcript = text

        # Quality guard
        reason = transcript_problem(text, duration_s)
        if reason:
            self.last_rejection = reason
            log.debug("Discarding ASR output (%s): %s", reason, " ".join(text[:200].split()))
            return ""

        # Accept
        with self._state_lock:
            key = speech_key(text)
            if key:
                self._accepted_keys.append(key)
            self._recent.append(text)
            self.last_rejection = None

        elapsed = time.monotonic() - started
        rtf = elapsed / max(duration_s, 1e-3)
        update_metrics(
            stages={"asr_s": round(elapsed, 3), "asr_rtf": round(rtf, 3)},
        )
        return text

    def cancel_pending(self) -> None:
        """Invalidate in-flight transcription requests."""
        with self._state_lock:
            self._generation += 1
            self.last_raw_transcript = ""
            self._recent.clear()
            self._accepted_keys.clear()

    def close(self) -> None:
        """Release the model and free GPU memory."""
        self._closed = True
        with self._state_lock:
            self._generation += 1
        model = getattr(self, "_model", None)
        if model is not None:
            # CTranslate2 models release GPU memory when garbage collected
            del self._model
            self._model = None
        update_metrics(workers={"asr": "closed"})
        log.info("Whisper ASR closed, GPU memory released")

    def remember(self, text: str) -> None:
        """Record an accepted transcript for context echo detection."""
        with self._state_lock:
            key = speech_key(text)
            if key:
                self._accepted_keys.append(key)
            self._recent.append(text)

    def forget(self, n: int = 1) -> None:
        """Remove recent accepted transcripts (e.g. after seek)."""
        with self._state_lock:
            for _ in range(n):
                if self._recent:
                    self._recent.pop()

    def recover_context_echo(self, audio: np.ndarray, probe: bool = False,
                             vad_confidence: float | None = None,
                             timeout: float = 0.0) -> str:
        """Retry transcription without context to recover from echo."""
        with self._state_lock:
            self.last_raw_transcript = ""
            self.last_rejection = None
        if not len(audio):
            return ""
        try:
            text = self._transcribe_impl(audio, probe, no_context=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Whisper context echo recovery failed: %s", exc)
            return ""
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
        if not text:
            self.last_rejection = "no_speech"
            return ""
        return text

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _transcribe_impl(
        self,
        audio: np.ndarray,
        probe: bool = False,
        no_context: bool = False,
    ) -> str:
        """Transcribe audio using the Whisper model."""
        samples = np.asarray(audio, dtype=np.float32)
        samples = np.ascontiguousarray(np.clip(samples, -1, 1))

        # Build decode options
        initial_prompt = None
        with self._state_lock:
            if self.context and not no_context:
                initial_prompt = self.context[:256]
            elif not no_context and self._recent:
                # Use recent transcripts as prompt for continuity
                recent = list(self._recent)
                initial_prompt = " ".join(recent[-3:])[:256]

        segments, _info = self._model.transcribe(
            samples,
            language="ja",
            task="transcribe",
            beam_size=self.cfg.threads,
            patience=1.0,
            length_penalty=1.0,
            temperature=0.0 if not probe else 0.3,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            suppress_tokens=[-1],
            word_timestamps=False,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=False,
            initial_prompt=initial_prompt,
        )

        # Collect segments, filtering out no-speech
        phrases = []
        for segment in segments:
            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            if no_speech_prob > 0.75 and avg_logprob < -0.75:
                continue
            text = segment.text.strip()
            if text:
                phrases.append(trim_repetition_tail(text))

        transcript = " ".join(phrases).strip()

        # Clean up
        transcript = " ".join(unicodedata.normalize("NFC", transcript).split())
        transcript = trim_repetition_tail(
            "".join(c for c in transcript if unicodedata.category(c) != "Cc")
        )
        transcript = re.sub(r"\s+", " ", transcript).strip()

        self.last_raw_transcript = transcript
        self._failures = 0

        duration_s = len(audio) / 16000
        log.debug(
            "Whisper: %.1fs audio -> %s",
            duration_s, transcript[:60],
        )

        return transcript
