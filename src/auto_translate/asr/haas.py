"""Haas-ja-v1.0 ASR served by an owned speech.cpp HTTP server.

Haas is a Japanese-only FastConformer model. Audio is sent as WAV to the
OpenAI-compatible ``/v1/audio/transcriptions`` endpoint. No language tagging
is needed — Haas transcribes only Japanese.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import time
import unicodedata
import wave
from collections import deque

import httpx
import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..speech_server import (
    SpeechCppServer,
    SpeechServerError,
    resolve_speech_executable,
)
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running
from .context_echo import speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_JAPANESE_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

# Haas is Japanese-only, yet engine revs can elicit short English noise
# words such as "ugh yeah", "phew", or elongated "yeaaah". These standalone
# phonetic guesses are not Japanese subtitles. Preserve mixed Japanese text.
_ENGINE_REV_LATIN = re.compile(
    r"(?:ye+a+h+|ye+a+|ya+h+|u+g+h+|u+h+|p+h+e+w+|w+h+e+w+|o+o+f+)",
    re.IGNORECASE,
)
# A standalone katakana cheer may be genuine speech. Use VAD evidence and
# repetition timing before rejecting it; do not blanket-ban イェー.
_ENGINE_REV_KATAKANA = re.compile(r"(?:イ[ェエ](?:ー{1,3}イ?|イ)){1,5}")


class HaasError(RuntimeError):
    """The Haas ASR server failed repeatedly; translation cannot continue."""


def wav_bytes(audio: np.ndarray, *, normalize: bool = True) -> bytes:
    """Convert float32 mono 16 kHz audio to WAV bytes."""
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("ASR requires finite mono float32 audio at 16000 Hz")
    value = np.clip(value, -1, 1)
    if normalize and value.size:
        power = value * value
        rms = float(np.sqrt(np.mean(power)))
        starts = np.arange(0, value.size, 160)
        frame_power = np.add.reduceat(power, starts) / np.minimum(160, value.size - starts)
        active_rms = float(np.sqrt(np.max(frame_power)))
        # Cap gain at 2.5× to avoid amplifying game engine noise into speech.
        # Require active_rms >= 0.006 (not 0.001) so that very quiet rumble is
        # not boosted.  Genuine Japanese speech typically has active_rms > 0.01.
        if active_rms >= 0.006 and rms < 0.02:
            gain = max(1.0, min(2.5, 0.02 / rms, 0.95 / float(np.max(np.abs(value)))))
            value = value * gain
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.ascontiguousarray(value * 32767, dtype="<i2").tobytes())
    return buffer.getvalue()


def is_english_engine_interjection(text: str) -> bool:
    """Only whole-utterance engine-like interjections, never Japanese dialogue."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[!！?？.,。、…〜～\s]+", " ", normalized).strip()
    words = normalized.split()
    return bool(words) and len(words) <= 8 and all(
        _ENGINE_REV_LATIN.fullmatch(word) for word in words
    )


def is_katakana_engine_interjection(text: str) -> bool:
    """Uncertain standalone イェー-type output, not normal dialogue."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[!！?？.,。、…〜～\s]+", "", normalized)
    return bool(normalized and _ENGINE_REV_KATAKANA.fullmatch(normalized))


def is_japanese_text(text: str) -> bool:
    """Check if text contains Japanese script characters."""
    return bool(_JAPANESE_SCRIPT.search(text))


# A clean decode this long after the last restart clears the restart budget.
_RESTART_STABLE_S = 120.0


class HaasASR:
    """Haas-ja-v1.0 ASR via speech.cpp HTTP server."""

    def __init__(self, cfg: AsrConfig, context: str = "") -> None:
        self.cfg = cfg
        # The speech.cpp Haas HTTP route accepts WAV only; context hints are unsupported.
        self.context = ""
        self.supports_context_biasing = False
        self._context_warning_logged = bool(context.strip())
        if self._context_warning_logged:
            log.warning("Haas ASR ignores configured context: speech.cpp has no verified Haas prompt support")
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
        self._restarts = 0
        self.last_timing: dict | None = None
        self._recent_engine_interjections: deque[float] = deque(maxlen=8)

        # --speech-server / asr.speech_server_path overrides the shared server_path.
        executable = resolve_speech_executable(cfg.speech_server_path or cfg.server_path or None)
        device = "cpu"
        if cfg.cuda_device is not None:
            device = "gpu"

        self.server = SpeechCppServer(
            model_name="haas-ja-v1.0",
            executable=executable,
            device=device,
            threads=cfg.threads,
            startup_timeout_s=cfg.startup_timeout_s,
        )
        try:
            self.server.start()
            # Warmup with silent audio to reserve encoder workspace (reduced for faster startup)
            warmup_s = 2.0
            self.server.transcribe(
                wav_bytes(np.zeros(int(16000 * warmup_s), np.float32)),
                timeout=cfg.startup_timeout_s,
            )
            check_running()
        except BaseException:
            self.close()
            raise
        update_metrics(workers={"asr": "ready (Haas-ja-v1.0)"})
        log.info("Haas-ja-v1.0 ready (pid=%s, device=%s, context biasing unsupported)",
                 self.server.process.pid, device)

    def set_context(self, context: str) -> None:
        """Explicitly report unsupported context hints instead of pretending to apply them."""
        with self._state_lock:
            if context.strip() and not self._context_warning_logged:
                self._context_warning_logged = True
                log.warning("Haas ASR name/context hints are unsupported by this speech.cpp endpoint")

    def cancel_pending(self) -> None:
        with self._state_lock:
            self._generation += 1
            self.last_raw_transcript = ""
            self._recent.clear()
            self._accepted_keys.clear()
            self._recent_engine_interjections.clear()

    def remember(self, text: str) -> None:
        """Bias the next decode toward continuity with recently accepted speech."""
        with self._state_lock:
            self._recent.append(text[-120:])
            self._accepted_keys.append(speech_key(text))

    def forget(self) -> None:
        with self._state_lock:
            self._recent.clear()
            self._recent_engine_interjections.clear()

    def reject_utterance(self, text: str, utt) -> str | None:
        """Acoustic-context gate for Haas's engine-rev-as-'yeah' false positives.

        Haas lacks calibrated word confidence: a text-only classifier cannot
        identify engine audio. Never suppress an ordinary Japanese sentence;
        reject only short isolated cheer-like hypotheses when evidence is
        speculative or when they recur abnormally within a few seconds.
        """
        if is_english_engine_interjection(text):
            return "haas_non_japanese_engine_interjection"
        if not is_katakana_engine_interjection(text):
            return None
        if utt.probe or (utt.vad_confidence is not None
                         and utt.vad_confidence < 0.65):
            return "haas_weak_katakana_interjection"
        now = float(utt.start_s)
        with self._state_lock:
            recent = self._recent_engine_interjections
            # A reset/seek clears history. Also clear on backward timestamps.
            if recent and now < recent[-1]:
                recent.clear()
            while recent and now - recent[0] > 7.0:
                recent.popleft()
            repeated = bool(recent and now - recent[-1] <= 7.0)
            recent.append(now)
        if repeated:
            return "haas_repeated_engine_interjection"
        return None

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        return self._decode(audio, probe, self.cfg.response_timeout_s)

    def recover_context_echo(self, audio: np.ndarray, probe: bool, vad_confidence: float | None,
                             timeout: float) -> str:
        """Decode identical PCM without hints/history. Retain a spoken list if corroborated."""
        with self._state_lock:
            self.last_raw_transcript = ""
            self.last_rejection = None
        duration_s = len(audio) / 16000
        if not len(audio):
            return ""
        wav = wav_bytes(audio, normalize=not probe)
        with self._inference_lock:
            if self._closed:
                return ""
            if not self.server.alive():
                self._restart("server exited")
            try:
                text, timing = self.server.transcribe_with_timing(wav, timeout=timeout)
            except (httpx.HTTPError, SpeechServerError, ValueError) as exc:
                if self._closed:
                    return ""
                return self._failed(exc)
        http_s = timing.get("http_request_s", 0.0)
        if http_s > 2.0:
            log.warning(
                "Haas stall (context echo): duration=%.1fs, http=%.3fs",
                duration_s, http_s
            )
            update_metrics(counters={"asr_stall_count": 1})
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
        if not text:
            self.last_rejection = "no_speech"
            return ""
        if is_japanese_text(text):
            return text
        self.last_rejection = "not_japanese"
        return ""

    def _decode(self, audio: np.ndarray, probe: bool, timeout: float) -> str:
        with self._state_lock:
            generation = self._generation
            self.last_raw_transcript = ""
            self.last_rejection = None
            self.last_match_score = 0.0
        duration_s = len(audio) / 16000
        if duration_s > 60:
            raise ValueError("ASR utterance exceeds 60 seconds; check VAD segmentation")
        if not len(audio):
            return ""
        prep_start = time.monotonic()
        wav = wav_bytes(audio, normalize=not probe)
        prep_time = time.monotonic() - prep_start
        started = time.monotonic()
        with self._inference_lock:
            if generation != self._generation or self._closed:
                return ""
            check_running()
            if not self.server.alive():
                self._restart("server exited")
            try:
                text, timing = self.server.transcribe_with_timing(wav, timeout=timeout)
            except (httpx.HTTPError, SpeechServerError, ValueError) as exc:
                if self._closed:
                    return ""
                return self._failed(exc)
        self._failures = 0
        self._note_stable_decode()
        with self._state_lock:
            if generation != self._generation:
                return ""
            self.last_raw_transcript = text
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))

        problem = (("haas_non_japanese_engine_interjection"
                    if is_english_engine_interjection(text) else None)
                   or ("no_speech" if not text else None)
                   or ("truncated" if not text and duration_s > 0.5 else None)
                   or transcript_problem(text, duration_s)
                   or ("blocklist" if any(word in text for word in self.cfg.blocklist) else None))

        if problem:
            self.last_rejection = problem
            log.debug("Discarding ASR output (%s): %s", problem, " ".join(text[:200].split()))
            return ""
        elapsed = time.monotonic() - started
        http_s = timing.get("http_request_s", 0.0)
        total_s = timing.get("total_s", 0.0)
        self.last_timing = timing
        if http_s > 2.0:
            log.warning(
                "Haas stall detected: duration=%.1fs, prep=%.3fs, http=%.3fs, total=%.3fs, rtf=%.2f",
                duration_s, prep_time, http_s, total_s, total_s / max(duration_s, 1e-3)
            )
            update_metrics(counters={"asr_stall_count": 1})
        update_metrics(
            stages={"asr_s": round(elapsed, 3), "asr_rtf": round(elapsed / max(duration_s, 1e-3), 3)},
            counters={"asr_http_total_s": round(http_s, 3)},
        )
        return text

    def _failed(self, exc: Exception) -> str:
        self._failures += 1
        self.last_rejection = "error"
        log.warning("ASR request failed (%d consecutive): %s", self._failures, exc)
        if not self.server.alive():
            self._restart(str(exc))
        elif self._failures >= 3:
            raise HaasError(f"Haas-ja-v1.0 failed three consecutive utterances: {exc}") from exc
        return ""

    def _note_stable_decode(self) -> None:
        """Forgive old restarts once the server has decoded cleanly for a while.

        The restart cap exists to stop a crash loop, not to make the third
        isolated crash hours into a stream fatal. A success only clears the
        counter when the last restart is at least _RESTART_STABLE_S old, so a
        server that dies right after each restart still hits the cap.
        """
        if self._restarts and (time.monotonic() - getattr(self, "_last_restart_at", 0.0)
                               >= _RESTART_STABLE_S):
            log.info("Haas-ja-v1.0 stable for %.0fs; restart counter reset (%d -> 0)",
                     _RESTART_STABLE_S, self._restarts)
            self._restarts = 0

    def _restart(self, reason: str) -> None:
        if self._restarts >= 2:
            raise HaasError(f"Haas-ja-v1.0 server stopped again ({reason}). {self.server.log_tail()}")
        self._restarts += 1
        self._last_restart_at = time.monotonic()
        log.warning("Restarting Haas-ja-v1.0 server (%s)", reason)
        update_metrics(workers={"asr": "restarting"})
        self.server.start()
        update_metrics(workers={"asr": "ready (Haas-ja-v1.0)"})

    def close(self) -> None:
        self._closed = True
        with self._state_lock:
            self._generation += 1
        server = getattr(self, "server", None)
        if server is not None:
            server.close()
