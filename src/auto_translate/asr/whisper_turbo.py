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
from pathlib import Path

import numpy as np
from huggingface_hub import try_to_load_from_cache

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from .context_echo import speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_WHISPER_MODEL = "Systran/faster-whisper-large-v3-turbo"

_DEFAULT_BEAM_SIZE = 5

_CT2_MODEL_FILES = {"model.bin", "config.json", "tokenizer.json", "vocabulary.json"}


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

        if cfg.cuda_device < 0:
            device = "cpu"
            compute_type = "int8"
            device_index = None
        else:
            device = "cuda"
            compute_type = "float16"
            device_index = cfg.cuda_device

        model_path = self._resolve_model_path(cfg)

        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cfg.threads,
                num_workers=1,
                local_files_only=True,
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
            "Whisper Large-v3 Turbo ready: model=%s device=%s compute=%s threads=%d device_index=%s",
            model_path, device, compute_type, cfg.threads, device_index,
        )

    # ------------------------------------------------------------------ #
    # Model resolution
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_model_path(cls, cfg: AsrConfig, is_explicit: bool = False) -> str:
        """Resolve a Whisper model path without triggering downloads.

        Prefers an explicitly configured local model directory, then an
        already-cached Hugging Face model. Raises if no usable local model
        exists — never falls back to a Haas model or silent download.

        Args:
            cfg: ASR configuration containing model_path.
            is_explicit: True if model_path was explicitly set by the user
                (GUI override or backend profile). False if inherited from
                common config (e.g., Haas default).

        When ``is_explicit=True`` and the path is invalid, raises a clear
        error. When ``is_explicit=False`` (inherited), falls through to HF
        cache search so Whisper can find its own cached model.
        """
        model_path = (cfg.model_path or "").strip()

        candidate = None
        if model_path:
            candidate = Path(model_path).expanduser()
            if candidate.exists():
                if cls._looks_like_whisper_model(candidate):
                    return str(candidate.resolve())
                # Configured path exists but is not a Whisper model.
                if is_explicit:
                    # User explicitly set this path — reject with clear error.
                    raise WhisperError(
                        f"Explicitly configured model path is not a valid Whisper model: "
                        f"{candidate}. Expected a CTranslate2 Whisper directory "
                        f"containing model.bin, config.json, etc."
                    )
                # Inherited path (e.g., Haas default) — fall through to cache search.
                log.warning(
                    "Configured ASR model path is not a Whisper model: %s — "
                    "falling back to Hugging Face cache search",
                    candidate,
                )
            else:
                # Path doesn't exist.
                if is_explicit:
                    raise WhisperError(
                        f"Explicitly configured model path does not exist: {candidate}"
                    )
                # Inherited non-existent path — fall through to cache search.

        # Search for cached Whisper model in Hugging Face cache.
        # Try the configured path as a repo id first, then the default model.
        repo_ids = []
        if model_path:
            repo_ids.append(model_path)
        if _WHISPER_MODEL not in repo_ids:
            repo_ids.append(_WHISPER_MODEL)

        for repo_id in repo_ids:
            cached = cls._find_cached_model(repo_id)
            if cached is not None:
                return str(cached)

        raise WhisperError(
            "No Whisper model available locally. Set asr.model_path to a local "
            f"faster-whisper model directory or cache {_WHISPER_MODEL}. "
            "Auto-download is disabled."
        )

    @classmethod
    def _find_cached_model(cls, repo_id: str) -> Path | None:
        """Return the cached snapshot directory for a HF repo id, or None."""
        for filename in ("model.bin", "config.json"):
            try:
                hit = try_to_load_from_cache(repo_id, filename, revision="main")
            except Exception:  # noqa: BLE001
                hit = None
            if hit:
                return Path(hit).resolve().parent
        return None

    @classmethod
    def _looks_like_whisper_model(cls, path: Path) -> bool:
        """Accept only directories that contain CTranslate2 Whisper model files.

        Requires at least one file from _CT2_MODEL_FILES (model.bin, config.json,
        tokenizer.json, vocabulary.json). A directory name containing "whisper"
        is not sufficient — the files must actually be present.
        """
        if not path.is_dir():
            return False
        names = {child.name.lower() for child in path.iterdir()}
        return bool(names & _CT2_MODEL_FILES)

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
        """Retry transcription without context to recover from echo.

        Enforces the supplied remaining timeout budget. If the budget has
        expired, returns early without starting inference.

        NOTE: The timeout is a soft deadline. CTranslate2 inference blocks
        the calling thread and cannot be interrupted from Python without
        process-level termination. The deadline is checked after inference
        returns: if it expired during inference, the result is discarded.
        Hard interruption of CTranslate2 is not supported.
        """
        if timeout <= 0:
            return ""
        with self._state_lock:
            if self._closed:
                return ""
            gen = self._generation
            self.last_raw_transcript = ""
            self.last_rejection = None
        if not len(audio):
            return ""
        deadline = time.monotonic() + timeout
        try:
            text = self._transcribe_impl(audio, probe, no_context=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Whisper context echo recovery failed: %s", exc)
            return ""
        if time.monotonic() >= deadline:
            return ""
        if gen != self._generation or self._closed:
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
                beam_size=_DEFAULT_BEAM_SIZE,
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
