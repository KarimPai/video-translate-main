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

_WHISPER_MODEL = "h2oai/faster-whisper-large-v3-turbo"

_DEFAULT_BEAM_SIZE = 5

_CT2_MODEL_FILES = {"model.bin", "config.json", "tokenizer.json", "vocabulary.json"}
# Minimum required files for a valid CTranslate2 Whisper model directory.
# Requires model.bin + config.json + at least one tokenizer/vocab file.
_CT2_REQUIRED_CORE = {"model.bin", "config.json"}
_CT2_REQUIRED_TOKENIZER = {"tokenizer.json", "vocabulary.json"}


class WhisperError(RuntimeError):
    """The Whisper ASR model failed repeatedly; transcription cannot continue."""


class WhisperASR:
    """Whisper Large-v3 Turbo ASR via faster-whisper in-process.

    Uses the faster-whisper library with CTranslate2 backend for efficient
    inference on CPU or CUDA. No external server process is needed.
    """

    def __init__(self, cfg: AsrConfig, context: str = "", *, is_explicit: bool = False) -> None:
        self.cfg = cfg
        self.context = context.strip()[: cfg.max_context_chars]
        self._name_hint_override = False  # Preserve legacy ASR context unless fusion opts in.
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
        self.last_detected_language: str = "ja"
        self.last_detected_language_probability: float = 1.0

        if cfg.cuda_device < 0:
            device = "cpu"
            compute_type = "int8"
            device_index = None
        else:
            device = "cuda"
            compute_type = "float16"
            device_index = cfg.cuda_device

        model_path = self._resolve_model_path(cfg, is_explicit=is_explicit)

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

        Distinguishes:
        1. Valid explicit local model directory.
        2. Invalid explicit local directory.
        3. Valid Hugging Face repository identifier already present in cache.
        4. Missing or incomplete cached repository.
        5. Inherited unrelated Haas model path.
        """
        model_path = (cfg.model_path or "").strip()

        if model_path:
            # Always check local directory first, regardless of path format.
            # This handles Windows paths with forward slashes (models/whisper-large-v3-turbo),
            # absolute paths, relative paths, and paths with ~ expansion.
            candidate = Path(model_path).expanduser()
            if candidate.is_dir():
                if cls._looks_like_whisper_model(candidate):
                    return str(candidate.resolve())
                # Directory exists but is not a Whisper model.
                if is_explicit:
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
                # Path doesn't exist as a local directory.
                # Check if this looks like an HF repo ID: contains a slash but is not
                # a Windows absolute path (no drive letter like C:) and doesn't use
                # backslashes. However, since we already checked the local path above,
                # any remaining path with forward slashes that doesn't exist locally
                # is treated as a potential HF repo ID.
                looks_like_repo_id = "/" in model_path and ":" not in model_path

                if looks_like_repo_id:
                    # Potential HF repo ID — check if it's cached
                    if is_explicit:
                        cached = cls._find_cached_model(model_path)
                        if cached is not None:
                            return str(cached)
                        raise WhisperError(
                            f"Explicitly configured Hugging Face repository '{model_path}' "
                            f"is not present in the local cache. Cache it manually first; "
                            f"auto-download is disabled."
                        )
                    # Non-explicit repo ID (inherited) — fall through to cache search below
                elif is_explicit:
                    # Explicit local path that doesn't exist and doesn't look like a repo ID
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
        """Return the cached snapshot directory for a HF repo id, or None.

        Only returns a path if it contains a genuinely complete CTranslate2
        Whisper model (model.bin + config.json + tokenizer/vocab).
        Incomplete cache snapshots are rejected.
        """
        # First try to locate model.bin to find the snapshot directory
        try:
            hit = try_to_load_from_cache(repo_id, "model.bin", revision="main")
        except Exception:  # noqa: BLE001
            hit = None
        if not hit:
            return None
        snapshot_dir = Path(hit).resolve().parent
        # Validate that the snapshot is a complete Whisper model
        if cls._looks_like_whisper_model(snapshot_dir):
            return snapshot_dir
        return None

    @classmethod
    def _looks_like_whisper_model(cls, path: Path) -> bool:
        """Accept only directories that contain a complete CTranslate2 Whisper model.

        Requires ALL of:
        - model.bin (the model weights)
        - config.json (model configuration)
        - At least one of: tokenizer.json, vocabulary.json (tokenizer assets)

        A directory containing only one or two of these files is rejected
        as an incomplete cache snapshot.
        """
        if not path.is_dir():
            return False
        names = {child.name.lower() for child in path.iterdir()}
        # Must have both core files
        if not _CT2_REQUIRED_CORE.issubset(names):
            return False
        # Must have at least one tokenizer/vocab file
        return bool(names & _CT2_REQUIRED_TOKENIZER)

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

    def set_context(self, context: str) -> None:
        """Opt in to bounded hints; an empty hint must NOT fall back to history."""
        with self._state_lock:
            self._name_hint_override = True
            self.context = context.strip()[: self.cfg.max_context_chars]

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
    # Language detection helpers
    # ------------------------------------------------------------------ #

    _JA_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
    _EN_CHARS = re.compile(r"[A-Za-z]")

    def _detect_language(self, text: str) -> str:
        """Detect the dominant language of a transcript.

        Returns 'ja' for Japanese, 'en' for English, 'none' for silence,
        or 'mixed' when both scripts are present.
        """
        if not text or not text.strip():
            return "none"
        has_ja = bool(self._JA_CHARS.search(text))
        has_en = bool(self._EN_CHARS.search(text))
        if has_ja and has_en:
            return "mixed"
        if has_ja:
            return "ja"
        if has_en:
            return "en"
        return "none"

    def _transcribe_impl(
        self,
        audio: np.ndarray,
        probe: bool = False,
        no_context: bool = False,
    ) -> str:
        """Transcribe audio using the Whisper model with language auto-detection.

        Uses automatic language detection per utterance so Japanese and English
        speech are correctly identified. Sets ``last_detected_language`` for the
        pipeline's language-aware routing.
        """
        samples = np.asarray(audio, dtype=np.float32)
        samples = np.ascontiguousarray(np.clip(samples, -1, 1))

        # Early exit for silence: if the entire clip has negligible energy,
        # skip inference entirely. This prevents Whisper from hallucinating
        # "Thank you" / "Obrigado" on silence or near-silence.
        audio_rms = float(np.sqrt(np.mean(samples ** 2)))
        if audio_rms < 0.003:
            self.last_raw_transcript = ""
            self._failures = 0
            with self._state_lock:
                self.last_detected_language = "none"
                self.last_detected_language_probability = 1.0
            return ""

        # Build decode options
        initial_prompt = None
        with self._state_lock:
            if not no_context and getattr(self, "_name_hint_override", False):
                # Fusion supplies names only. Never fall back to previously
                # accepted free-form dialogue when no name candidate exists.
                initial_prompt = self.context[:160] or None
            elif self.context and not no_context:
                initial_prompt = self.context[:256]
            elif not no_context and self._recent:
                # Legacy mode: use recent transcripts for continuity.
                recent = list(self._recent)
                initial_prompt = " ".join(recent[-3:])[:256]

            # Auto-detect language — do not force Japanese.
            # When a name hint is active, Whisper may still detect the true
            # spoken language from audio content.
            segments, info = self._model.transcribe(
                samples,
                language=None,
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

        # Record the detected language from Whisper's info object
        detected_lang = getattr(info, "language", None) or "ja"
        detected_lang_prob = getattr(info, "language_probability", 0.0)
        with self._state_lock:
            self.last_detected_language = detected_lang
            self.last_detected_language_probability = detected_lang_prob

        # Collect segments, filtering out no-speech
        # With auto-detect (language=None), no_speech_prob is less reliable.
        # Use a combination of no_speech_prob, avg_logprob, and audio energy
        # to reject hallucinations from silence and noise.
        audio_rms = float(np.sqrt(np.mean(samples ** 2)))
        phrases = []
        for segment in segments:
            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            # Reject segments with high no-speech probability and poor logprob
            if no_speech_prob > 0.75 and avg_logprob < -0.75:
                continue
            # Extra guard for silence/noise: if audio energy is very low,
            # require strong evidence (high no_speech_prob threshold lowered)
            if audio_rms < 0.005 and no_speech_prob > 0.5:
                continue
            # Reject very poor logprob segments regardless of no_speech_prob
            if avg_logprob < -1.0 and audio_rms < 0.01:
                continue
            text = segment.text.strip()
            if text:
                phrases.append(trim_repetition_tail(text))

        transcript = " ".join(phrases).strip()

        # Cross-validate: if Whisper reports Japanese but the output contains
        # no Japanese characters, trust the text content over the model's
        # language tag (common with short English interjections).
        text_lang = self._detect_language(transcript)
        if text_lang != "none" and text_lang != detected_lang:
            # Prefer text-based detection when it contradicts the model tag
            if text_lang == "en" and detected_lang == "ja":
                # Whisper claimed Japanese but output is pure English —
                # likely a false positive on the language detector.
                with self._state_lock:
                    self.last_detected_language = "en"
            elif text_lang == "ja" and detected_lang == "en":
                with self._state_lock:
                    self.last_detected_language = "ja"
            elif text_lang == "mixed":
                with self._state_lock:
                    self.last_detected_language = "mixed"

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
            "Whisper: %.1fs audio -> lang=%s (prob=%.2f) text_lang=%s -> %s",
            duration_s, detected_lang, detected_lang_prob, text_lang, transcript[:60],
        )

        return transcript

    # ------------------------------------------------------------------ #
    # Mixed-language recovery (pipeline hook)
    # ------------------------------------------------------------------ #

    def recover_japanese_from_mixed_audio(
        self, audio: np.ndarray, probe: bool = False
    ) -> list[tuple[float, float, str]]:
        """Recover Japanese segments from mixed-language audio.

        When the primary pass detects dominant English (e.g., GTA character
        dialogue), re-transcribe short windows with forced Japanese to catch
        the streamer's Japanese speech that was masked.

        Returns a list of (start_offset_s, end_offset_s, japanese_text).
        """
        samples = np.asarray(audio, dtype=np.float32)
        duration_s = len(samples) / 16000
        if duration_s < 2.0:
            return []

        candidates: list[tuple[float, float, str]] = []
        # Slide 3-second windows with 50% overlap
        window_s = 3.0
        step_s = 1.5
        offset = 0.0
        while offset + window_s <= duration_s:
            start_sample = int(offset * 16000)
            end_sample = int((offset + window_s) * 16000)
            window = samples[start_sample:end_sample]
            if len(window) < 1600:  # skip tiny windows
                offset += step_s
                continue
            try:
                with self._inference_lock:
                    segments, _info = self._model.transcribe(
                        np.ascontiguousarray(np.clip(window, -1, 1)),
                        language="ja",
                        task="transcribe",
                        beam_size=_DEFAULT_BEAM_SIZE,
                        temperature=0.0,
                        no_speech_threshold=0.5,
                        log_prob_threshold=-0.5,
                        compression_ratio_threshold=2.4,
                        suppress_tokens=[-1],
                        word_timestamps=False,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        without_timestamps=False,
                    )
                for seg in segments:
                    no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
                    avg_logprob = getattr(seg, "avg_logprob", 0.0)
                    if no_speech_prob > 0.75 and avg_logprob < -0.75:
                        continue
                    text = seg.text.strip()
                    if not text:
                        continue
                    text = trim_repetition_tail(text)
                    if not self._JA_CHARS.search(text):
                        continue
                    # Segment timestamps are relative to the window
                    seg_start = offset + getattr(seg, "start", 0.0)
                    seg_end = offset + getattr(seg, "end", seg_start + 1.0)
                    candidates.append((seg_start, seg_end, text))
            except Exception as exc:
                log.debug("Mixed-language recovery window failed: %s", exc)
            offset += step_s

        # Deduplicate overlapping segments — keep the one with better logprob
        if not candidates:
            return []
        candidates.sort(key=lambda x: x[0])
        merged: list[tuple[float, float, str]] = [candidates[0]]
        for start, end, text in candidates[1:]:
            last = merged[-1]
            if start < last[1]:
                # Overlapping — keep the longer text
                if len(text) > len(last[2]):
                    merged[-1] = (min(last[0], start), max(last[1], end), text)
            else:
                merged.append((start, end, text))
        return merged
