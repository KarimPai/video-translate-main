"""Whisper Large-v3 Turbo ASR via faster-whisper (CTranslate2).

Whisper is a multilingual speech recognition model. The Large-v3 Turbo variant
is optimized for speed while maintaining accuracy. Runs in-process via the
faster-whisper library with CTranslate2 backend, supporting both CPU and CUDA.

Japanese and English are identified without forced-language decoding. Short
uncertain foreign-language windows can be rechecked without prompts, but
language detection is not speaker separation.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from huggingface_hub import try_to_load_from_cache

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from .context_echo import context_match, speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_WHISPER_MODEL = "h2oai/faster-whisper-large-v3-turbo"

_DEFAULT_BEAM_SIZE = 5
# A zero-byte or header-only download is not a usable CTranslate2 checkpoint.
_MIN_MODEL_BYTES = 1024 * 1024

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
        self.last_detected_language: str = "und"
        self.last_detected_language_probability: float = 0.0
        self.last_language_spans: list[tuple[float, float, str, str]] = []
        self._last_decode_prompt = ""

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
        # Must have at least one tokenizer/vocab file and real model weights.
        # The loader performs the final format/integrity verification.
        if not names & _CT2_REQUIRED_TOKENIZER:
            return False
        try:
            return (path / "model.bin").stat().st_size >= _MIN_MODEL_BYTES
        except OSError:
            return False

    # ------------------------------------------------------------------ #
    # ASRBackend protocol
    # ------------------------------------------------------------------ #

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        """Transcribe 16 kHz mono float32 audio to Japanese text."""
        with self._state_lock:
            generation = self._generation
            self.last_raw_transcript = ""
            self.last_rejection = None
            self.last_detected_language = "und"
            self.last_detected_language_probability = 0.0
            self.last_language_spans = []
            self.last_match_score = 0.0

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
            self.last_rejection = self.last_rejection or reason
            log.debug("Discarding ASR output (%s): %s", self.last_rejection, " ".join(text[:200].split()))
            return ""

        # A long verbatim reproduction of the supplied decoder prompt is
        # suspicious, not automatically proof of speech. Require a fresh
        # context-free decode through the pipeline instead of displaying it.
        prompt = getattr(self, "_last_decode_prompt", "")
        if prompt and len(speech_key(text)) >= 24:
            score = context_match(text, prompt)
            if score >= 0.97:
                self.last_match_score = score
                self.last_rejection = "context_echo"
                self.last_language_spans = []
                log.info("Whisper suspected prompt echo (score=%.2f)", score)
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
            self.last_language_spans = []
            self.last_detected_language = "und"
            self.last_detected_language_probability = 0.0

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

    def forget(self, n: int | None = None) -> None:
        """Clear history after seek, or remove a specified number of entries."""
        with self._state_lock:
            if n is None:
                self._recent.clear()
                self._accepted_keys.clear()
            else:
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
            with self._inference_lock:
                if gen != self._generation or self._closed:
                    return ""
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
        reason = transcript_problem(text, len(audio) / 16000)
        if reason:
            self.last_rejection = reason
            return ""
        self.last_match_score = 0.0
        self.last_rejection = None
        return text

    # ------------------------------------------------------------------ #
    # Language detection and acoustic evidence
    # ------------------------------------------------------------------ #

    _JA_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
    _JA_KANA = re.compile(r"[\u3040-\u30ff]")
    _EN_CHARS = re.compile(r"[A-Za-z]")
    _EN_WORDS = re.compile(r"[A-Za-z]{2,}(?:['’][A-Za-z]+)?")

    @classmethod
    def _detect_language(cls, text: str) -> str:
        """Classify written content, not the actual audio speaker's identity.

        One embedded romanized name (Jasonが来た) is still Japanese; a
        whitespace-separated English clause plus Japanese is mixed.
        """
        if not isinstance(text, str) or not text.strip():
            return "none"
        japanese = bool(cls._JA_CHARS.search(text))
        english = bool(cls._EN_CHARS.search(text))
        if japanese and english:
            separated = bool(
                re.search(r"[A-Za-z][.!?]\s*[\u3040-\u30ff\u3400-\u9fff]", text)
                or re.search(r"[A-Za-z]\s+[\u3040-\u30ff\u3400-\u9fff]", text)
                or re.search(r"[\u3040-\u30ff\u3400-\u9fff][.!?。！？]?\s+[A-Za-z]", text)
            )
            return "mixed" if separated else "ja"
        if japanese:
            return "ja"
        if english:
            return "en"
        return "none"

    @staticmethod
    def _reject_segment(segment) -> bool:
        """Require multiple bad decoder signals before discarding real speech.

        No single RMS threshold or poor logprob may suppress a quiet Japanese
        reaction. This gate targets no-speech and decoder-loop combinations.
        """
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        ratio = float(getattr(segment, "compression_ratio", 0.0) or 0.0)
        return (
            (no_speech > 0.75 and logprob < -0.75)
            or (no_speech > 0.90 and logprob < -0.40)
            or (ratio > 2.8 and logprob < -0.70)
            or (no_speech > 0.40 and logprob < -1.35)
        )

    @classmethod
    def _split_text_languages(cls, text: str) -> list[tuple[str, str]]:
        """Separate clear English clauses from Japanese without splitting names.

        Only split where script/sentence boundaries provide evidence. Ambiguous
        text stays Japanese to avoid dropping a streamer's genuine reaction.
        This is text routing, NOT source separation of overlapping speakers.
        """
        text = text.strip()
        if not text:
            return []
        if cls._detect_language(text) != "mixed":
            return [(cls._detect_language(text), text)]

        # Whisper often produces a single segment for several short sentences.
        # Split at strong punctuation first; preserve each clause's meaning.
        clauses = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text)
        output: list[tuple[str, str]] = []
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            kind = cls._detect_language(clause)
            if kind == "mixed":
                # A separate English phrase preceding Japanese can be hidden
                # from subtitles while still supplying conversational context.
                prefix = re.match(
                    r"^([A-Za-z][A-Za-z0-9'’ ,.!?\-]*?)([\u3040-\u30ff\u3400-\u9fff].*)$",
                    clause,
                )
                if prefix and (
                    len(cls._EN_WORDS.findall(prefix.group(1))) >= 2
                    or bool(re.search(r"[.!?]\s*$", prefix.group(1)))
                    or prefix.group(1).strip(" .,!?").casefold()
                    in {"yes", "yeah", "no", "okay", "ok", "hello", "stop",
                        "go", "hey", "wait"}
                ):
                    output.extend([("en", prefix.group(1).strip()),
                                   ("ja", prefix.group(2).strip())])
                    continue
                # Japanese first, then an independent English clause. Roman
                # names embedded in Japanese are not separated unless this is
                # multiword English or an isolated NPC interjection.
                suffix = re.match(
                    r"^(.*?[\u3040-\u30ff\u3400-\u9fff][^A-Za-z]*?)\s+([A-Za-z].+)$",
                    clause,
                )
                if suffix and (
                    len(cls._EN_WORDS.findall(suffix.group(2))) >= 2
                    or suffix.group(2).strip(" .,!?").casefold()
                    in {"yes", "yeah", "no", "okay", "ok", "hello", "stop",
                        "go", "hey", "wait"}
                ):
                    output.extend([("ja", suffix.group(1).strip()),
                                   ("en", suffix.group(2).strip())])
                    continue
                kind = "ja"
            output.append((kind, clause))
        return output

    @classmethod
    def _classify_segment(cls, text: str, detected_lang: str) -> list[tuple[str, str]]:
        parts = cls._split_text_languages(text)
        result = []
        for lang, part in parts:
            if lang == "none":
                continue
            # Do not relabel Spanish/other Latin-script speech as English
            # solely because its transcript uses the Latin alphabet.
            if lang == "en" and detected_lang not in ("ja", "en"):
                lang = detected_lang
            if lang == "ja" and detected_lang not in ("ja", "en") and not cls._JA_KANA.search(part):
                lang = detected_lang
            result.append((lang, part))
        return result

    def _transcribe_impl(
        self,
        audio: np.ndarray,
        probe: bool = False,
        no_context: bool = False,
    ) -> str:
        """Auto-detect language, retain segment timing, and suppress weak noise."""
        samples = np.ascontiguousarray(
            np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
        )
        # Only bypass truly digital silence; absolute RMS such as 0.003 can
        # erase soft-spoken Japanese and does not distinguish game noises.
        if not len(samples) or float(np.max(np.abs(samples))) <= 1e-6:
            self.last_raw_transcript = ""
            self.last_language_spans = []
            self.last_detected_language = "none"
            self.last_detected_language_probability = 1.0
            self._last_decode_prompt = ""
            return ""

        with self._state_lock:
            initial_prompt = None
            if not no_context and getattr(self, "_name_hint_override", False):
                initial_prompt = self.context[:160] or None
            elif self.context and not no_context:
                initial_prompt = self.context[:256]
            elif not no_context and self._recent:
                initial_prompt = " ".join(list(self._recent)[-3:])[:256]
            self._last_decode_prompt = initial_prompt or ""

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
        detected = (getattr(info, "language", None) or "und").lower()
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        phrases = []
        spans: list[tuple[float, float, str, str]] = []
        duration_s = len(samples) / 16000.0
        # Generator decoding can be lazy; consume before handing back the lock
        # held by transcribe() / recover_context_echo().
        for segment in segments:
            if self._reject_segment(segment):
                continue
            text = trim_repetition_tail(str(getattr(segment, "text", "")).strip())
            problem = transcript_problem(
                text, max(0.1, float(getattr(segment, "end", duration_s))
                               - float(getattr(segment, "start", 0.0)))
            )
            if problem:
                self.last_rejection = problem
                continue
            start = max(0.0, min(duration_s, float(getattr(segment, "start", 0.0))))
            end = max(start, min(duration_s, float(getattr(segment, "end", duration_s))))
            parts = self._classify_segment(text, detected)
            if not parts:
                continue
            phrases.append(text)
            length = sum(max(1, len(part)) for _, part in parts)
            at = start
            for i, (lang, part) in enumerate(parts):
                fraction = max(1, len(part)) / max(1, length)
                next_at = end if i == len(parts) - 1 else at + (end - start) * fraction
                spans.append((at, next_at, lang, part))
                at = next_at

        transcript = " ".join(phrases).strip()
        transcript = " ".join(unicodedata.normalize("NFC", transcript).split())
        transcript = trim_repetition_tail(
            "".join(c for c in transcript if unicodedata.category(c) != "Cc")
        )
        langs = {kind for _, _, kind, _ in spans}
        if not transcript:
            label = "none"
        elif "ja" in langs and len(langs) > 1:
            label = "mixed"
        elif len(langs) == 1:
            label = next(iter(langs))
        else:
            label = detected
        self.last_detected_language = label
        self.last_detected_language_probability = probability
        self.last_language_spans = spans
        self.last_raw_transcript = transcript
        log.debug("Whisper: %.1fs audio -> %s (p=%.2f) %s",
                  duration_s, label, probability, transcript[:80])
        return transcript

    # ------------------------------------------------------------------ #
    # Mixed-language recovery (pipeline hook)
    # ------------------------------------------------------------------ #

    def recover_japanese_from_mixed_audio(
        self, audio: np.ndarray, probe: bool = False
    ) -> list[tuple[float, float, str]]:
        """Bounded auto-language rechecks; never force English audio to Japanese.

        Returns only short Japanese windows independently labeled Japanese by
        Whisper with sufficient language confidence. This cannot isolate
        simultaneous voices occupying the same acoustic window.
        """
        samples = np.asarray(audio, dtype=np.float32)
        duration_s = len(samples) / 16000.0
        if probe or duration_s < 2.2 or self._closed:
            return []
        generation = self._generation
        deadline = time.monotonic() + min(10.0, float(self.cfg.response_timeout_s))
        windows = 2 if duration_s <= 4.5 else (3 if duration_s <= 9 else 4)
        step = duration_s / windows
        found: list[tuple[float, float, str, float]] = []
        with self._inference_lock:
            for i in range(windows):
                if self._closed or generation != self._generation or time.monotonic() >= deadline:
                    break
                start = max(0.0, i * step - (0.2 if i else 0))
                end = min(duration_s, (i + 1) * step + (0.2 if i < windows - 1 else 0))
                window = np.ascontiguousarray(np.clip(
                    samples[round(start * 16000):round(end * 16000)], -1, 1
                ))
                if len(window) < 16000 or float(np.max(np.abs(window))) <= 1e-6:
                    continue
                try:
                    segments, info = self._model.transcribe(
                        window,
                        language=None,  # never force ja on English/game noise
                        task="transcribe",
                        beam_size=_DEFAULT_BEAM_SIZE,
                        temperature=0.0,
                        no_speech_threshold=0.6,
                        log_prob_threshold=-1.0,
                        compression_ratio_threshold=2.4,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        initial_prompt=None,
                    )
                    if (getattr(info, "language", None) != "ja"
                            or float(getattr(info, "language_probability", 0.0) or 0.0) < 0.70):
                        continue
                    # Consume the generator before starting another inference.
                    for seg in segments:
                        if self._reject_segment(seg):
                            continue
                        txt = trim_repetition_tail(str(getattr(seg, "text", "")).strip())
                        if len(self._JA_KANA.findall(txt)) < 2:
                            continue
                        seg_start = max(0.0, start + float(getattr(seg, "start", 0.0)))
                        seg_end = min(duration_s, start + float(getattr(seg, "end", end - start)))
                        if seg_end <= seg_start or transcript_problem(txt, seg_end - seg_start):
                            continue
                        found.append((seg_start, seg_end, txt,
                                      float(getattr(seg, "avg_logprob", -1.0) or -1.0)))
                except Exception as exc:  # noqa: BLE001
                    log.debug("Whisper mixed-language recheck unavailable: %s", exc)
                if self._closed or generation != self._generation or time.monotonic() >= deadline:
                    return []

        # Keep distinct utterances even when timestamps overlap; only remove
        # near-identical hypotheses, preferring better ASR confidence.
        deduped: list[tuple[float, float, str, float]] = []
        for candidate in sorted(found, key=lambda x: (x[0], -x[3])):
            start, end, txt, score = candidate
            match = next((i for i, prev in enumerate(deduped)
                          if start < prev[1] and prev[0] < end
                          and SequenceMatcher(None, speech_key(txt), speech_key(prev[2]),
                                              autojunk=False).ratio() >= 0.90), None)
            if match is None:
                deduped.append(candidate)
            elif score > deduped[match][3]:
                deduped[match] = candidate
        return [(start, end, txt) for start, end, txt, _ in sorted(deduped)]
