"""Qwen3-ASR speech recognition via llama.cpp HTTP server.

Qwen3-ASR is a multimodal model that transcribes audio to text. It requires
both a main model GGUF and an audio projector GGUF (mmproj). Audio is sent
as base64 WAV via the ``/v1/chat/completions`` endpoint with ``input_audio``
content type.

Supports 52 languages. Japanese decoding is the project default; automatic
language detection remains available for multilingual audio and experiments.
"""
from __future__ import annotations

import base64
import logging
import re
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager

import httpx
import numpy as np

from ..asr.haas import wav_bytes
from ..config import AsrConfig
from ..llama_server import (
    DeadlineHttpClient,
    LlamaServer,
    detect_device,
    require_gguf,
    resolve_executable,
)
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running, stopping
from .context_echo import speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_QWEN3_SYSTEM = (
    "You are a speech recognition assistant. Transcribe the audio exactly as "
    "spoken. Do not add commentary, translations, or descriptions."
)

# Default llama-server path for Qwen3-ASR
LLAMA_SERVER_DEFAULT = ".runtime/llama-b12719-cuda12/llama-server.exe"

# Measured holoGTA: 14.4s split-first lost dialogue; 15.8s preserved it.
# Restrict the production fast path to the longer, tested class of cues.
LONG_TRUNCATION_SPLIT_MIN_S = 15.0

# Forced language (asr.language = "ja", the project default). The assistant turn is
# prefilled with the model's own language header so the decoder skips language
# identification and starts directly on the transcript.
_FORCED_LANGUAGE_PREFIX = {"ja": "language Japanese<asr_text>"}


_LANGUAGE_HEADER = re.compile(r"^\s*language\s+([A-Za-z][A-Za-z ,\-]*)\s*$", re.IGNORECASE)
_JAPANESE_KANA = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
_NON_SPEECH = {"none", "no speech", "no_speech", "silence", "null"}


def has_japanese_recovery_text(text: str) -> bool:
    """Require kana for recovered Japanese, matching the downstream subtitle gate."""
    return bool(_JAPANESE_KANA.search(unicodedata.normalize("NFKC", text)))


def recovery_split_ranges(audio: np.ndarray, overlap_s: float = 0.0) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose a low-energy midpoint for truncated-audio recovery.

    Optional overlap is *experimental*, measured in total shared seconds.
    Production keeps 0.0 until audio-grounded results justify a change.
    No text rewriting or guessed de-duplication occurs here.
    """
    if not 0.0 <= overlap_s <= 0.30:
        raise ValueError("Qwen recovery overlap must be between 0.0 and 0.30s")
    length = len(audio)
    if length < 4 * 16000:
        raise ValueError("Qwen recovery requires at least four seconds of audio")
    center = length // 2
    radius = length // 10
    frame = 1280  # 80 ms at 16 kHz
    best = center
    best_energy = float("inf")
    for pos in range(max(frame, center - radius), min(length - frame, center + radius) + 1, frame):
        window = audio[pos - frame // 2:pos + frame // 2]
        energy = float(np.mean(np.square(window.astype(np.float64))))
        if energy < best_energy:
            best_energy = energy
            best = pos
    shared_half = round(overlap_s * 16000 / 2)
    left_end = min(length, best + shared_half)
    right_start = max(0, best - shared_half)
    return (0, left_end), (right_start, length)


def parse_qwen_asr_response(raw: str) -> tuple[str, str]:
    """Separate audio-derived language metadata from spoken text.

    Qwen3-ASR normally emits 'language English<asr_text>spoken words'.
    Kana can classify untagged text. Clear Japanese dialogue must also survive
    an English-dominant language tag in mixed gameplay audio.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "none", ""
    raw = raw.strip().replace("<|im_end|>", "").strip()
    infer_from_text = True
    if "<asr_text>" in raw:
        prefix, text = raw.split("<asr_text>", 1)
        prefix = prefix.strip()
        if prefix:
            match = _LANGUAGE_HEADER.fullmatch(prefix)
            if match is None:
                return "und", ""  # malformed metadata cannot enter subtitles
            language = match.group(1).strip().lower()
            infer_from_text = language in _NON_SPEECH
            if language in _NON_SPEECH:
                # Qwen's 'language None' normally signals nonspeech. In the
                # unusual case where it also provides actual words, preserve
                # the text as uncertain rather than silently discarding it.
                language = "und" if text.strip() else "none"
                if language == "none":
                    return "none", ""
            language = {
                "japanese": "ja", "ja": "ja",
                "english": "en", "en": "en",
                "indonesian": "id", "id": "id",
                "malay": "ms", "ms": "ms",
            }.get(language, "und")
            # The auto language tag describes the dominant voice, not every
            # speaker. Preserve clearly recognized Japanese in mixed gameplay;
            # plain English without kana continues to be marked foreign.
            if language != "ja" and len(_JAPANESE_KANA.findall(text)) >= 2:
                language = "ja"
        else:
            language = "und"
    else:
        if raw.lower().startswith("language "):
            return "und", ""  # truncated metadata is not a transcript
        language, text = "und", raw
    text = " ".join(unicodedata.normalize("NFC", text).split())
    text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
    if not text or text.casefold() in {"<|nospeech|>", "<|audio_sentinel|>"}:
        return "none", ""
    if language == "und" and infer_from_text and _JAPANESE_KANA.search(text):
        language = "ja"
    return language, text



class Qwen3ASRError(RuntimeError):
    """The Qwen3-ASR server failed repeatedly; transcription cannot continue."""


class Qwen3ASRTruncationError(Qwen3ASRError):
    """A single utterance exceeded its safe output-token budget."""

    reason = "token_limit"


class Qwen3ASRResponseError(Qwen3ASRError):
    """The server returned no reliable completion for this utterance."""

    reason = "invalid_response"


class Qwen3ASRTransportError(Qwen3ASRError):
    """A transient transport failure exhausted one utterance's recovery."""

    reason = "transport_error"


class _UtteranceDeadline(Qwen3ASRError):
    """An utterance's time budget expired.

    When ``transport=True``, the deadline was consumed by genuine transport
    failures (HTTP timeouts, connection errors) rather than seek cancellations
    or inference-queue delays. Those count toward the consecutive-transport-
    failure shutdown threshold.
    """
    reason = "deadline"

    def __init__(self, message: str, *, transport: bool = False) -> None:
        super().__init__(message)
        self.transport = transport


class Qwen3ASR:
    """Qwen3-ASR speech recognition via llama.cpp HTTP server.

    Uses the multimodal chat completions API with input_audio content type.
    Requires both a model GGUF and an audio projector (mmproj) GGUF.
    """

    def __init__(self, cfg: AsrConfig, context: str = "") -> None:
        self.cfg = cfg
        # Only after >=15s audio demonstrably hits the 256-token ceiling,
        # prefer the existing bounded two-window recovery rather than spending
        # more time retrying the entire audio with 512 tokens. Normal decodes,
        # short segments, language checks and quality gates are unchanged.
        # The diagnostic harness explicitly disables this for its control arm.
        self._prefer_split_on_truncation = True
        # Audio-grounded 0.16/0.24s recovery overlap experiment only.
        # Production has no shared samples or speculative text de-duplication.
        self._truncation_recovery_overlap_s = 0.0
        self.context = context.strip()[: cfg.max_context_chars]
        # Dedicated opt-in experiment; generic stream/name context is NOT
        # automatically sent to the model (avoids vocabulary hallucinations).
        self._vocabulary_hint = ""
        self._recent: deque[str] = deque(maxlen=cfg.recent_context_lines)
        self._accepted_keys: deque[str] = deque(maxlen=32)
        self.last_match_score = 0.0
        self.last_raw_transcript = ""
        self.last_detected_language = "und"
        self.last_rejection: str | None = None
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._generation = 0
        self._cancel_event = threading.Event()
        self._closed = False
        self._restarts = 0
        self._transport_failures = 0
        self._restart_timestamps: deque[float] = deque()
        self.last_timing: dict | None = None

        # Resolve model and mmproj paths
        model_path = require_gguf(cfg.model_path, "Qwen3-ASR model")
        if not cfg.mmproj_path:
            raise Qwen3ASRError(
                "Qwen3-ASR requires mmproj_path (audio projector GGUF). "
                "Configure asr.mmproj_path in config.toml."
            )
        mmproj_path = require_gguf(cfg.mmproj_path, "Qwen3-ASR mmproj")

        # Resolve llama-server executable
        server_path = cfg.server_path or LLAMA_SERVER_DEFAULT
        executable = resolve_executable(server_path)
        device = detect_device(executable, cfg.cuda_device)

        # Build server arguments
        args = [
            "--model", str(model_path),
            "--mmproj", str(mmproj_path),
            "--alias", "qwen3-asr",
            "--threads", str(cfg.threads),
            "--ctx-size", str(cfg.context_size),
        ]
        if device:
            args.extend(["--device", device])

        self.server = LlamaServer(
            name="qwen3-asr",
            executable=executable,
            args=args,
            startup_timeout_s=cfg.startup_timeout_s,
            strict_http_deadlines=True,
        )
        try:
            self.server.start()
            try:
                self._warmup(time.monotonic() + cfg.response_timeout_s, self._generation)
            except _UtteranceDeadline:
                # Initial warmup is optional after a successful health check.
                # Restart warmup must still propagate the utterance deadline.
                log.warning("Qwen3-ASR initial warmup timed out after %.1fs", cfg.response_timeout_s)
            check_running()
        except BaseException:
            self.close()
            raise

        update_metrics(workers={"asr": f"ready (Qwen3-ASR, {model_path.name})"})
        log.info(
            "Qwen3-ASR ready: model=%s mmproj=%s device=%s",
            model_path.name, mmproj_path.name, device,
        )

    # ------------------------------------------------------------------ #
    # ASRBackend protocol
    # ------------------------------------------------------------------ #

    def transcribe(self, audio: np.ndarray, probe: bool = False, *,
                   generation: int | None = None, deadline: float | None = None) -> str:
        """Decode under one budget, including retries, restart and speech recovery."""
        if len(audio) < 800:
            return ""
        generation = self._generation if generation is None else generation
        deadline = time.monotonic() + self.cfg.response_timeout_s if deadline is None else deadline
        try:
            with self._inference(deadline, generation):
                with self._state_lock:
                    self._remaining(deadline, generation)
                    self.last_detected_language = "und"
                    self.last_raw_transcript = ""
                    self.last_rejection = None
                try:
                    result = self._recognize(audio, probe, deadline, generation)
                    self._transport_failures = 0
                    return result
                except Qwen3ASRTransportError as exc:
                    self._remaining(deadline, generation)
                    self._transport_failures += 1
                    if self._transport_failures >= 3:
                        raise Qwen3ASRError("Qwen3-ASR transport failed for three consecutive utterances") from exc
                    self._reject_request(generation, exc)
                    return ""
                except _UtteranceDeadline as exc:
                    # A deadline consumed by genuine transport failures (HTTP
                    # timeouts, connection errors) counts toward the consecutive
                    # transport-failure threshold. Non-transport deadlines (seek
                    # cancellation, inference-queue delay) do not.
                    if exc.transport:
                        self._transport_failures += 1
                        if self._transport_failures >= 3:
                            raise Qwen3ASRError(
                                "Qwen3-ASR transport failed for three consecutive utterances"
                            ) from exc
                    self._reject_request(generation, exc)
                    return ""
                except (Qwen3ASRResponseError, Qwen3ASRTruncationError) as exc:
                    # Non-transport failures reset the consecutive counter.
                    # Invalid responses and token limits are not transport issues.
                    self._transport_failures = 0
                    self._reject_request(generation, exc)
                    return ""
        except (CancelledError, _UtteranceDeadline):
            # A caller that never acquired the inference lock cannot replace
            # the active request's metadata.
            return ""

    def _recognize(self, audio: np.ndarray, probe: bool, deadline: float, generation: int) -> str:
        text = self._transcribe_with_retries(audio, probe, deadline, generation)
        reason = transcript_problem(text, len(audio) / 16000)
        if self._should_salvage(audio, probe, deadline, reason):
            candidate = self._salvage(audio, text, deadline, generation)
            if candidate:
                text, reason = candidate, None

        with self._state_lock:
            self._remaining(deadline, generation)
            self.last_rejection = reason
            key = speech_key(text)
            if not reason:
                return text
            if reason in ("context_echo", "repeated_output") and key and key not in self._accepted_keys:
                self._accepted_keys.append(key)
        return ""

    def _cancelled(self, generation: int) -> bool:
        stop = getattr(self, "_session_stop", None)
        return (generation != self._generation or self._closed or stopping.is_set()
                or (stop is not None and stop.is_set()))

    def _remaining(self, deadline: float, generation: int) -> float:
        if self._cancelled(generation):
            raise CancelledError("Qwen3-ASR request is stale or shutting down")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _UtteranceDeadline("Qwen3-ASR utterance budget exceeded")
        return remaining

    @contextmanager
    def _inference(self, deadline: float, generation: int):
        while not self._inference_lock.acquire(timeout=min(self._remaining(deadline, generation), 0.05)):
            pass
        try:
            self._remaining(deadline, generation)
            yield
        finally:
            self._inference_lock.release()

    def _decode(self, audio: np.ndarray, *, deadline: float, generation: int, **kwargs) -> str:
        timeout = self._remaining(deadline, generation)
        try:
            text = self._transcribe_impl(audio, timeout=timeout, generation=generation,
                                         deadline=deadline, **kwargs)
        except httpx.HTTPError as exc:
            # A real HTTP timeout can consume the deadline before the retry
            # handler regains control. Preserve its transport origin instead
            # of replacing it with an unclassified _UtteranceDeadline.
            try:
                self._remaining(deadline, generation)
            except _UtteranceDeadline:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                if (isinstance(exc, (httpx.TimeoutException, httpx.NetworkError,
                                     httpx.RemoteProtocolError))
                        or (status is not None and 500 <= status < 600)):
                    raise _UtteranceDeadline(
                        "Qwen3-ASR transport exhausted the utterance deadline",
                        transport=True,
                    ) from exc
                raise
            raise
        except Exception:
            self._remaining(deadline, generation)
            raise
        self._remaining(deadline, generation)
        return text

    def _wait_retry(self, seconds: float, deadline: float, generation: int) -> None:
        with self._state_lock:
            remaining = self._remaining(deadline, generation)
            event = self._cancel_event
        # The seek callback wakes this immediately. The short polling interval
        # also observes the session-wide stop events without waiting on them.
        end = time.monotonic() + min(remaining, seconds)
        while (left := end - time.monotonic()) > 0:
            event.wait(min(left, 0.05))
            self._remaining(deadline, generation)
        self._remaining(deadline, generation)

    def _transcribe_with_retries(
        self, audio: np.ndarray, probe: bool, deadline: float, generation: int,
    ) -> str:
        for attempt in range(4):
            try:
                return self._decode_with_recovery(audio, probe, deadline, generation)
            except httpx.HTTPError as exc:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                transient = (isinstance(exc, (httpx.TimeoutException, httpx.NetworkError,
                                               httpx.RemoteProtocolError))
                             or (status is not None and 500 <= status < 600))
                if not transient:
                    raise Qwen3ASRError(f"Qwen3-ASR permanent HTTP failure: {exc}") from exc
                log.warning("Qwen3-ASR transient HTTP failure (attempt %d/4): %s", attempt + 1, exc)
            if attempt < 3:
                try:
                    self._wait_retry(0.5 * (attempt + 1), deadline, generation)
                except _UtteranceDeadline as exc:
                    raise _UtteranceDeadline(
                        "Qwen3-ASR transport retries exhausted the utterance deadline",
                        transport=True,
                    ) from exc

        # Token-limit and malformed-response errors are utterance-local and
        # never reach this transport restart path.
        try:
            remaining = self._remaining(deadline, generation)
        except _UtteranceDeadline as exc:
            raise _UtteranceDeadline(
                "Qwen3-ASR transport retries exhausted the utterance deadline",
                transport=True,
            ) from exc
        if remaining <= 0.5:
            # The utterance budget was consumed by genuine transport failures
            # (HTTP timeouts, connection errors). Mark it so the caller counts
            # it toward the consecutive-transport-failure threshold.
            raise _UtteranceDeadline(
                "Insufficient utterance budget to restart Qwen3-ASR",
                transport=True,
            )
        try:
            self._restart(deadline=deadline, generation=generation)
        except (CancelledError, _UtteranceDeadline):
            raise
        except Exception as exc:
            self._remaining(deadline, generation)
            raise Qwen3ASRError(
                f"Qwen3-ASR transcription failed after retries; restart failed: {exc}"
            ) from exc
        if self._remaining(deadline, generation) <= 0.05:
            raise _UtteranceDeadline(
                "Insufficient utterance budget for a decode after restart",
                transport=True,
            )
        try:
            return self._decode_with_recovery(audio, probe, deadline, generation)
        except httpx.HTTPError as exc:
            raise Qwen3ASRTransportError(f"Qwen3-ASR transcription failed after restart: {exc}") from exc

    def _decode_with_recovery(
        self, audio: np.ndarray, probe: bool, deadline: float, generation: int,
    ) -> str:
        try:
            return self._decode(audio, probe=probe, deadline=deadline, generation=generation)
        except Qwen3ASRTruncationError:
            self._remaining(deadline, generation)
            recovered = self._recover_truncated_utterance(
                audio, probe=probe, deadline=deadline, generation=generation,
            )
            if recovered:
                return recovered
            if (getattr(self, "_prefer_split_on_truncation", False)
                    and len(audio) >= LONG_TRUNCATION_SPLIT_MIN_S * 16000
                    and self._remaining(deadline, generation) > 0.5):
                try:
                    fallback = self._decode(audio, probe=probe, deadline=deadline,
                                            generation=generation, max_tokens=512)
                except (Qwen3ASRTruncationError, Qwen3ASRResponseError, httpx.HTTPError) as exc:
                    log.warning("Qwen3-ASR full-window fallback failed: %s", exc)
                else:
                    if fallback:
                        return fallback
            raise

    def _reject_request(self, generation: int,
                        exc: _UtteranceDeadline | Qwen3ASRResponseError | Qwen3ASRTruncationError) -> None:
        with self._state_lock:
            if self._cancelled(generation):
                return
            self.last_rejection = exc.reason
            self.last_detected_language = "und"
            self.last_raw_transcript = ""
        log.warning("Qwen3-ASR skipped one utterance (%s): %s", exc.reason, exc)

    def _should_salvage(self, audio: np.ndarray, probe: bool, deadline: float,
                        reason: str | None) -> bool:
        return (reason in {"repeated_character", "repeated_kana_loop",
                           "implausible_length", "implausible_speech_rate"}
                and self.cfg.recover_rejected_speech and not probe
                and self.last_detected_language == "ja" and len(audio) >= 6 * 16000
                and deadline - time.monotonic() > 1.25)

    def _salvage(self, audio: np.ndarray, text: str, deadline: float, generation: int) -> str | None:
        raw, language = self.last_raw_transcript, self.last_detected_language
        try:
            candidate = self._recover_truncated_utterance(
                audio, probe=False, deadline=deadline, generation=generation,
            )
            self._remaining(deadline, generation)
        except CancelledError:
            # Discard partial recovery metadata before releasing inference
            # to the next generation.
            with self._state_lock:
                self.last_raw_transcript, self.last_detected_language = raw, language
                self.last_rejection = transcript_problem(text, len(audio) / 16000)
            raise
        except _UtteranceDeadline:
            raise
        except (RuntimeError, httpx.HTTPError, ValueError) as exc:
            log.warning("Qwen3-ASR quality recovery unavailable: %s", exc)
            candidate = None
        with self._state_lock:
            self._remaining(deadline, generation)
            if (candidate and self.last_detected_language == "ja"
                    and speech_key(candidate) != speech_key(text)
                    and transcript_problem(candidate, len(audio) / 16000) is None):
                return candidate
            self.last_raw_transcript, self.last_detected_language = raw, language
        return None

    def _recover_truncated_utterance(
        self, audio: np.ndarray, *, probe: bool, deadline: float, generation: int,
    ) -> str | None:
        """Recover a long token-truncated utterance without inventing text.

        A bounded two-window retry limits pathological repetitions. Do not
        accept a partial result when one window fails, languages disagree,
        or the original deadline expires. This avoids leaking incomplete
        transcripts and does not restart an otherwise healthy server.
        """
        if len(audio) < 4 * 16000 or self._remaining(deadline, generation) <= 0.5:
            return None

        # Production uses the exact former zero-overlap boundary. Test-only
        # overlap keeps words near the split audible to both requests, but
        # may duplicate text: never enable it blindly for subtitles.
        ranges = recovery_split_ranges(
            audio, overlap_s=getattr(self, "_truncation_recovery_overlap_s", 0.0),
        )
        parts = tuple(audio[start:end] for start, end in ranges)
        transcripts: list[str] = []
        languages: list[str] = []
        for part in parts:
            if self._remaining(deadline, generation) <= 0.5:
                return None
            try:
                candidate = self._decode(
                    part, probe=probe, deadline=deadline, generation=generation, max_tokens=512,
                )
            except _UtteranceDeadline:
                raise
            except (Qwen3ASRError, httpx.HTTPError, ValueError) as exc:
                log.warning("Qwen3-ASR shortened-audio recovery failed: %s", exc)
                return None
            lang = self.last_detected_language
            if not candidate or lang in ("und", "none") or transcript_problem(candidate, len(part) / 16000):
                return None
            transcripts.append(candidate)
            languages.append(lang)

        self._remaining(deadline, generation)
        if languages[0] != languages[1]:
            return None
        joined = ("" if languages[0] == "ja" else " ").join(transcripts)
        if transcript_problem(joined, len(audio) / 16000):
            return None
        self.last_detected_language = languages[0]
        self.last_raw_transcript = joined
        return joined

    def recover_japanese_from_mixed_audio(
        self, audio: np.ndarray, probe: bool = False, *,
        generation: int | None = None, deadline: float | None = None,
    ) -> list[tuple[float, float, str]]:
        """Recheck short windows of a foreign-dominant segment for Japanese.

        Auto language ID describes a dominant language for the whole audio
        request, so simultaneous English game voices can hide a Japanese
        streamer. Rechecking shorter slices uses the *audio* language label
        again, not forced-Japanese decoding (which hallucinates on English).
        Work is bounded and optional: only the pipeline's strong VAD-positive
        foreign segments call this method.
        """
        duration = len(audio) / 16000.0
        if self._closed or probe or duration < 2.2:
            return []
        generation = self._generation if generation is None else generation
        budget_s = min(float(self.cfg.response_timeout_s), 12.0)
        recovery_deadline = time.monotonic() + budget_s
        deadline = recovery_deadline if deadline is None else min(recovery_deadline, deadline)
        # At most four short automatic-language requests. Overlap boundaries
        # slightly to avoid cutting syllables; keep original media timestamps.
        num_windows = 2 if duration <= 4.5 else (3 if duration <= 9.0 else 4)
        step = duration / num_windows
        found: list[tuple[float, float, str]] = []
        try:
            with self._inference(deadline, generation):
                original_language = self.last_detected_language
                original_raw = self.last_raw_transcript
                try:
                    for i in range(num_windows):
                        try:
                            if self._remaining(deadline, generation) <= 0.15:
                                break
                        except _UtteranceDeadline:
                            break
                        start_s = max(0.0, i * step - (0.25 if i else 0.0))
                        end_s = min(duration, (i + 1) * step + (0.25 if i < num_windows - 1 else 0.0))
                        left = round(start_s * 16000)
                        right = round(end_s * 16000)
                        if right - left < 16000:
                            continue
                        try:
                            candidate = self._decode(
                                audio[left:right], deadline=deadline, generation=generation,
                                language="auto",
                            )
                        except _UtteranceDeadline:
                            break
                        except (httpx.HTTPError, Qwen3ASRError, ValueError) as exc:
                            log.debug("Qwen mixed-language window recovery unavailable: %s", exc)
                            break
                        if (self.last_detected_language != "ja"
                                or not has_japanese_recovery_text(candidate)
                                or transcript_problem(candidate, end_s - start_s)):
                            continue
                        key = speech_key(candidate)
                        if not key:
                            continue
                        if any(
                            speech_key(previous) == key and start_s < previous_end
                            for _, previous_end, previous in found
                        ):
                            continue
                        found.append((start_s, end_s, candidate))
                    if self._cancelled(generation):
                        raise CancelledError("Qwen3-ASR recovery became stale")
                finally:
                    # Supplemental decode must not overwrite primary cue metadata.
                    self.last_detected_language = original_language
                    self.last_raw_transcript = original_raw
        except (CancelledError, _UtteranceDeadline):
            return []
        return found

    def bind_session_stop(self, stop: threading.Event) -> None:
        self._session_stop = stop

    def cancel_pending(self, generation: int | None = None) -> None:
        """Invalidate earlier requests without stopping a healthy server.

        Active HTTP exchanges observe the generation change and disconnect.
        """
        with self._state_lock:
            if generation is None:
                generation = self._generation + 1
            if generation <= self._generation:
                return
            self._generation = generation
            self._cancel_event.set()
            self._cancel_event = threading.Event()

    def close(self) -> None:
        """Make shutdown visible before waiting for any server lifecycle work."""
        with self._state_lock:
            self._closed = True
            self._cancel_event.set()
        self.server.close()
        update_metrics(workers={"asr": "closed"})

    def remember(self, text: str) -> None:
        """Record an accepted transcript for context echo detection."""
        with self._state_lock:
            key = speech_key(text)
            if key:
                self._accepted_keys.append(key)
            self._recent.append(text)

    def forget(self, n: int | None = None) -> None:
        """Remove recent accepted transcripts (e.g. after seek).

        When called without arguments (n=None), clears all context to match
        the behavior of other ASR backends. When called with a specific count,
        removes only that many recent transcripts.
        """
        with self._state_lock:
            if n is None:
                self._recent.clear()
                accepted_keys = getattr(self, "_accepted_keys", None)
                if accepted_keys is not None:
                    accepted_keys.clear()
            else:
                for _ in range(n):
                    if self._recent:
                        self._recent.pop()
                    accepted_keys = getattr(self, "_accepted_keys", None)
                    if accepted_keys:
                        accepted_keys.pop()

    def set_vocabulary_hint(self, words: str) -> None:
        """Set a bounded *experimental* ASR vocabulary, or clear with ''.

        This is intentionally separate from stream context/name-fusion. A
        static hotword list must not become an instruction to invent dialogue.
        The production default is no system message at all.
        """
        if not isinstance(words, str):
            raise TypeError("Vocabulary hint must be a string")
        terms = [term.strip() for term in words.replace("\n", ",").split(",")]
        terms = [term for term in terms if term]
        if (len(terms) > 12 or any(len(term) > 32 for term in terms)
                or len(", ".join(terms)) > 220):
            raise ValueError("Vocabulary hint exceeds 12 terms or 220 characters")
        self._vocabulary_hint = ", ".join(terms)

    def recover_context_echo(self, audio: np.ndarray, probe: bool = False,
                             vad_confidence: float | None = None, timeout: float = 0.0, *,
                             generation: int | None = None, deadline: float | None = None) -> str:
        """Retry without optional vocabulary using the original utterance budget."""
        if not len(audio) or (deadline is None and timeout <= 0):
            return ""
        generation = self._generation if generation is None else generation
        deadline = time.monotonic() + timeout if deadline is None else deadline
        try:
            with self._inference(deadline, generation):
                text = self._decode(audio, probe=probe, no_context=True,
                                    deadline=deadline, generation=generation)
                with self._state_lock:
                    self._remaining(deadline, generation)
                    if not text:
                        self.last_rejection = "no_speech"
                    return text
        except CancelledError:
            return ""
        except (Qwen3ASRError, httpx.HTTPError) as exc:
            log.warning("Qwen3-ASR context echo recovery failed: %s", exc)
            return ""

    def _warmup(self, deadline: float, generation: int) -> None:
        """Warm up within the caller's budget, without a token-limit retry."""
        silent = np.zeros(2 * 16000, dtype=np.float32)
        try:
            self._decode(silent, deadline=deadline, generation=generation, max_tokens=512)
        except (CancelledError, _UtteranceDeadline):
            raise
        except (Qwen3ASRError, httpx.HTTPError) as exc:
            log.warning("Qwen3-ASR warmup failed (may still work): %s", exc)

    def _transcribe_impl(
        self,
        audio: np.ndarray,
        probe: bool = False,
        no_context: bool = False,
        timeout: float | None = None,
        max_tokens: int = 256,
        *,
        generation: int | None = None,
        deadline: float | None = None,
        language: str | None = None,
    ) -> str:
        """Send audio to the server and return transcribed text.

        ``language`` overrides ``cfg.language`` for this call ("auto" or "ja").

        ``timeout`` overrides the configured response timeout (used by context
        echo recovery to enforce the remaining time budget).

        The optional recognition vocabulary is conveyed through a system
        message only when explicitly enabled with set_vocabulary_hint().
        no_context=True omits it for independent echo recovery.
        """
        t0 = time.perf_counter()
        generation = self._generation if generation is None else generation
        if deadline is None:
            deadline = time.monotonic() + (self.cfg.response_timeout_s if timeout is None else timeout)
        self._remaining(deadline, generation)

        # Encode audio as WAV base64
        wav = wav_bytes(audio, normalize=True)
        b64 = base64.b64encode(wav).decode("ascii")

        # With asr.language = "auto", Qwen detects the language from audio.
        # The project defaults to "ja"; forcing Japanese can turn English
        # dialogue and game sounds into misleading Japanese hypotheses.
        messages = []
        hint = getattr(self, "_vocabulary_hint", "")
        if hint and not no_context and not probe:
            # Qwen3-ASR's chat template accepts vocabulary as a system turn.
            # Audio remains the only user message. This is *not* forced text.
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": "Vocabulary: " + hint}],
            })
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"},
                    }
                ],
            },
        )

        # Forced Japanese continues from the model's own language header.
        # Probes and the mixed-audio recovery pass stay automatic because
        # their purpose is to discover the language.
        mode = getattr(getattr(self, "cfg", None), "language", "auto") if language is None else language
        forced_prefix = "" if probe else _FORCED_LANGUAGE_PREFIX.get(mode, "")
        if forced_prefix:
            messages.append({"role": "assistant", "content": forced_prefix})

        payload = {
            "messages": messages,
            "temperature": 0.0 if not probe else 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        }

        http_s = 0.0
        try:
            client = self.server.client
            options = ({"check": lambda: self._remaining(deadline, generation)}
                       if isinstance(client, DeadlineHttpClient) else {})
            resp = client.post(
                f"{self.server.base_url}/v1/chat/completions",
                json=payload,
                timeout=self._remaining(deadline, generation),
                **options,
            )
            http_s = time.perf_counter() - t0
            self._remaining(deadline, generation)
            resp.raise_for_status()
        except httpx.DecodingError as exc:
            raise Qwen3ASRResponseError("Invalid HTTP encoding in llama-server response") from exc
        except httpx.TimeoutException:
            elapsed_on_timeout = time.perf_counter() - t0
            log.warning("Qwen3-ASR HTTP timeout after %.1fs", elapsed_on_timeout)
            update_metrics(counters={"asr_stall_count": 1})
            raise
        except httpx.HTTPError as exc:
            elapsed_on_error = time.perf_counter() - t0
            log.error("Qwen3-ASR HTTP error after %.1fs: %s", elapsed_on_error, exc)
            raise

        try:
            body = resp.json()
        except ValueError as exc:
            raise Qwen3ASRResponseError("Invalid JSON in llama-server response") from exc
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise Qwen3ASRResponseError("Invalid llama-server choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        usage = body.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        self._remaining(deadline, generation)
        # llama.cpp chat completions use "stop" for a completed sequence and
        # "length" for a limit. A missing finish reason is ambiguous, even
        # below the cap. Reaching the cap is enough to try bounded recovery.
        if finish == "length" or (finish is None and type(tokens) is int and tokens >= max_tokens):
            prefer_split = bool(getattr(self, "_prefer_split_on_truncation", False))
            if (prefer_split and max_tokens < 512
                    and len(audio) >= LONG_TRUNCATION_SPLIT_MIN_S * 16000):
                log.info(
                    "Qwen3-ASR experiment: %.1fs audio hit %d-token limit; "
                    "try bounded split recovery first", len(audio) / 16000, max_tokens,
                )
                raise Qwen3ASRTruncationError(
                    "Qwen3-ASR long audio truncated; experimental split-first recovery"
                )
            if max_tokens < 512 and self._remaining(deadline, generation) > 0.5:
                log.warning("Qwen3-ASR hit %d-token limit; retrying with 512", max_tokens)
                return self._transcribe_impl(audio, probe=probe, no_context=no_context,
                                             max_tokens=512, deadline=deadline, generation=generation,
                                             language=language)
            raise Qwen3ASRTruncationError("Qwen3-ASR output truncated at token limit")

        if finish != "stop":
            raise Qwen3ASRResponseError("Missing or unsupported llama-server finish_reason")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise Qwen3ASRResponseError("Invalid llama-server transcript content")
        # A prefilled turn normally returns only the continuation. Restore the
        # header so the existing parser sees a normal response; if the server
        # echoed the header already, leave it alone.
        if forced_prefix and "<asr_text>" not in content:
            content = forced_prefix + content
        # Keep model-detected language metadata separate from spoken text.
        language, text = parse_qwen_asr_response(content)
        with self._state_lock:
            self._remaining(deadline, generation)
            self.last_detected_language = language
            self.last_raw_transcript = text

        elapsed = time.perf_counter() - t0
        duration_s = len(audio) / 16000
        rtf = elapsed / max(duration_s, 1e-3)
        log.debug(
            "Qwen3-ASR: %.1fs audio -> %.1fs http, %.1fs total (%.2fx RTF): %s",
            duration_s, http_s, elapsed, rtf, text[:60],
        )
        update_metrics(
            stages={"asr_s": round(elapsed, 3), "asr_rtf": round(rtf, 3)},
            counters={"asr_http_total_s": round(http_s, 3)},
        )

        return text

    def _restart(self, *, deadline: float, generation: int) -> None:
        """Recycle one open server under the same utterance deadline.

        Uses a rolling five-minute window: at most two restarts are permitted
        within any preceding 300 seconds. Old timestamps are pruned on each
        call so the window is truly rolling rather than fixed.
        """
        restart_lock = getattr(self, "_restart_lock", None)
        if restart_lock is None:
            restart_lock = self._restart_lock = threading.Lock()
        remaining = self._remaining(deadline, generation)
        if not restart_lock.acquire(timeout=remaining):
            self._remaining(deadline, generation)
            raise _UtteranceDeadline("Qwen3-ASR restart lock exceeded utterance deadline")
        try:
            self._restart_locked(deadline=deadline, generation=generation)
        finally:
            restart_lock.release()

    def _restart_locked(self, *, deadline: float, generation: int) -> None:
        def check():
            self._remaining(deadline, generation)

        check()
        now = time.monotonic()
        # Prune timestamps older than 300 seconds
        while self._restart_timestamps and self._restart_timestamps[0] < now - 300:
            self._restart_timestamps.popleft()
        if len(self._restart_timestamps) >= 2:
            raise Qwen3ASRError("Qwen3-ASR restart limit reached within five minutes; server is unstable")
        self._restart_timestamps.append(now)
        self._restarts = len(self._restart_timestamps)
        log.warning("Restarting Qwen3-ASR server (attempt %d)", self._restarts)
        try:
            self.server.reset_for_restart(deadline=deadline, check=check)
            check()
            self.server.start(deadline=deadline, check=check)
            check()
            # A seek after startup cancels warmup work and leaves the healthy child.
            self._warmup(deadline, generation)
            with self._state_lock:
                check()
                self._accepted_keys.clear()
                self._recent.clear()
        except CancelledError:
            # A seek interrupted the attempt. Remove the timestamp so the
            # aborted restart does not consume a slot in the rolling window.
            if self._restart_timestamps and self._restart_timestamps[-1] == now:
                self._restart_timestamps.pop()
            self._restarts = len(self._restart_timestamps)
            raise

    def recover_stalled(self, deadline: float | None = None) -> bool:
        """Restart the owned server without invalidating the current session generation."""
        now = time.monotonic()
        while self._restart_timestamps and self._restart_timestamps[0] < now - 300:
            self._restart_timestamps.popleft()
        if len(self._restart_timestamps) >= 2:
            raise Qwen3ASRError("Qwen3-ASR watchdog restart limit reached within five minutes")
        if deadline is None:
            deadline = now + self.cfg.startup_timeout_s + 6.0
        with self._state_lock:
            generation = self._generation
        self._restart(deadline=deadline, generation=generation)
        return True
