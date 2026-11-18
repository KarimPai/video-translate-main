"""Dual Japanese ASR adapter for the Qwen-era HY-MT2 pipeline.

Haas on speech.cpp/Vulkan is the primary recognizer; Whisper Turbo on a
killable CTranslate2 CUDA worker supplies independent acoustic evidence.
No stream title or unchecked history is passed to either decoder. Candidates
are validated independently. The deterministic selection favors agreement
and strong acoustic evidence rather than speculative LLM reconstruction.
"""
from __future__ import annotations

import logging
import time
import re
import threading
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher

import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import record_event, update_metrics
from .haas import HaasASR
from .quality_guard import transcript_problem
from .whisper_turbo import WhisperTurboASR

log = logging.getLogger(__name__)
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SPACES = re.compile(r"[\s。、！？!?.,…「」『』ー～〜]+")


def normalized_audio(audio: np.ndarray, *, quiet_gain: bool, probe: bool) -> np.ndarray:
    """Qwen-era bounded quiet-speech gain, shared by Haas and Whisper.

    A short active frame avoids treating VAD padding as the signal level.
    Silence, very faint noise and VAD-negative probes never get amplified.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Dual ASR requires finite mono 16 kHz audio")
    samples = np.clip(samples, -1.0, 1.0)
    if not samples.size or not quiet_gain or probe:
        return samples
    power = samples * samples
    rms = float(np.sqrt(np.mean(power)))
    peak = float(np.max(np.abs(samples)))
    if rms >= 0.02 or peak <= 0:
        return samples
    starts = np.arange(0, samples.size, 160)
    frames = np.add.reduceat(power, starts) / np.minimum(160, samples.size - starts)
    active_rms = float(np.sqrt(np.max(frames)))
    if active_rms < 0.001:
        return samples
    gain = min(8.0, 0.02 / max(rms, 1e-8), 0.95 / peak)
    return (samples * gain).astype(np.float32) if gain > 1 else samples


def canonical(text: str) -> str:
    return _SPACES.sub("", unicodedata.normalize("NFKC", text).casefold())


def plausible(text: str, duration_s: float, blocklist: list[str], *, probe=False) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    text = " ".join(text.split())
    if transcript_problem(text, duration_s):
        return False
    if any(token in text for token in blocklist):
        return False
    # Speech.cpp and Whisper operate as Japanese ASR, not speech translation.
    # Permit very short Latin reactions (e.g. OK); reject long English monologues.
    if not _JAPANESE.search(text) and len(text) > 6:
        return False
    if probe and not _JAPANESE.search(text):
        return False
    return True


def whisper_evidence(confidence: dict | None) -> bool:
    """Check model-internal evidence; NOT a calibrated cross-ASR confidence."""
    if not isinstance(confidence, dict) or not confidence.get("segments"):
        return False
    logprob = confidence.get("avg_logprob")
    no_speech = confidence.get("no_speech_prob")
    compression = confidence.get("compression_ratio")
    if logprob is not None and (not isinstance(logprob, (int, float)) or logprob < -0.65):
        return False
    if no_speech is not None and (not isinstance(no_speech, (int, float)) or no_speech > 0.5):
        return False
    if compression is not None and (not isinstance(compression, (int, float)) or compression > 2.4):
        return False
    return logprob is not None and logprob > -0.48


def select_candidate(primary: str, secondary: str, confidence: dict | None,
                     duration_s: float, *, probe=False) -> tuple[str, str]:
    """Do not compose words from two hypotheses or reconstruct unheard speech."""
    if not primary:
        # A Whisper-only probe needs acoustic support, not merely fluent text.
        if secondary and (not probe or whisper_evidence(confidence)):
            return secondary, "whisper_fallback"
        return "", "no_supported_speech"
    if not secondary:
        return primary, "haas_only"
    left, right = canonical(primary), canonical(secondary)
    if left == right:
        return primary, "agreement"
    similarity = SequenceMatcher(None, left, right).ratio()
    strong = whisper_evidence(confidence)
    # Minor lexical or punctuation differences: favor a strongly evidenced
    # Whisper hypothesis without treating its logprob as comparable to Haas.
    if strong and similarity >= 0.76:
        return secondary, "whisper_close_agreement"
    # Haas may capture only the start of a longer phrase; do not accept an
    # arbitrarily longer Whisper expansion without overlap and audio duration.
    if (strong and similarity >= 0.48 and duration_s >= 1.2
            and len(right) >= len(left) * 1.4 and len(right) <= duration_s * 15 + 24):
        return secondary, "whisper_supported_extension"
    return primary, "haas_conservative_disagreement"


class RecoveringHaas:
    """Retire a stuck speech.cpp worker before the following audio request."""
    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self.worker: HaasASR | None = HaasASR(cfg)
        self.failures = 0
        self.disabled = False

    def transcribe(self, audio: np.ndarray, probe=False) -> str:
        if self.disabled:
            return ""
        try:
            if self.worker is None:
                self.worker = HaasASR(self.cfg)
            result = self.worker.transcribe(audio, probe)
        except (TimeoutError, OSError, RuntimeError) as exc:
            self.failures += 1
            log.warning("Haas worker failed (count=%d): %s", self.failures, exc)
            worker, self.worker = self.worker, None
            if worker is not None:
                worker.close()
            if self.failures == 2 and self.cfg.haas_device.lower() != "cpu":
                self.cfg = replace(self.cfg, haas_device="cpu")
                log.warning("Switching Haas to CPU after repeated Vulkan failures")
            if self.failures >= 4:
                self.disabled = True
                log.error("Haas disabled after repeated failures; relying on Whisper if available")
            return ""
        self.failures = 0
        return result

    def close(self) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.close()


class DualASR:
    """Drop-in ASR satisfying the new pipeline's transcribe/remember/reset API."""

    def __init__(self, cfg: AsrConfig, *, live: bool = False):
        self.cfg = cfg
        self.live = live
        self.mode = cfg.secondary_live_mode if live else cfg.secondary_vod_mode
        self._lock = threading.RLock()
        self._decode_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self.last_raw_transcript = ""
        self.last_rejection: str | None = None
        self.last_language: str | None = None
        self.last_selection: str = ""
        self.secondary: WhisperTurboASR | None = None
        self.primary = RecoveringHaas(cfg)
        if self.mode != "off":
            try:
                self.secondary = WhisperTurboASR(
                    model=cfg.secondary_model,
                    device=cfg.secondary_device,
                    compute_type=cfg.secondary_compute_type,
                    beam_size=cfg.secondary_beam_size,
                    startup_timeout=cfg.secondary_startup_timeout_s,
                )
            except (Exception, OSError) as exc:
                log.warning("Whisper secondary not available: %s", exc)
                record_event("dual_asr_secondary_unavailable", reason=type(exc).__name__)
                self.secondary = None
        update_metrics(workers={"asr": self.status})
        log.info("Dual ASR ready: %s", self.status)

    @property
    def status(self) -> str:
        if self.secondary is None:
            return "Haas only (Whisper unavailable or disabled)"
        return "Haas + Whisper Turbo (" + self.mode + ")"

    def set_context(self, context: str) -> None:
        """Benchmark compatibility: no speculative ASR prompting in dual mode."""
        return None

    def remember(self, text: str) -> None:
        # Translation, not unconditional decoder prompts, maintains context.
        return None

    def forget(self) -> None:
        return None

    def cancel_pending(self) -> None:
        with self._lock:
            self._generation += 1
            self.last_raw_transcript = ""
            self.last_rejection = None

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        with self._decode_lock:
            with self._lock:
                if self._closed:
                    return ""
                generation = self._generation
                self.last_raw_transcript = ""
                self.last_rejection = None
                self.last_language = None
                self.last_selection = ""
            duration = len(audio) / 16000.0
            audio = normalized_audio(audio, quiet_gain=self.cfg.normalize_quiet_audio, probe=probe)
            if duration > 60:
                raise ValueError("Dual ASR utterance exceeds 60 seconds")
            primary_started = time.monotonic()
            primary = self.primary.transcribe(audio, probe)
            haas_elapsed = time.monotonic() - primary_started
            if self._invalidated(generation):
                return ""
            if not plausible(primary, duration, self.cfg.blocklist, probe=probe):
                primary = ""
            secondary = ""
            confidence = {}
            use_secondary = (self.secondary is not None and self.mode != "off"
                             and (self.mode == "all" or not primary or probe
                                  or len(canonical(primary)) <= self.cfg.secondary_selective_chars))
            whisper_elapsed = 0.0
            if use_secondary:
                secondary_started = time.monotonic()
                try:
                    timeout = (self.cfg.secondary_live_timeout_s if self.live
                               else self.cfg.secondary_vod_timeout_s)
                    secondary = self.secondary.transcribe(audio, timeout=timeout)
                    confidence = self.secondary.last_confidence
                except (TimeoutError, RuntimeError, OSError) as exc:
                    log.warning("Whisper secondary decode failed: %s", exc)
                    record_event("dual_asr_secondary_failed", reason=type(exc).__name__)
                    # A timed-out worker might still be decoding. Retire it,
                    # not only its reply, before attempting the next request.
                    self.secondary.close()
                    self.secondary = None
                    update_metrics(workers={"asr": self.status})
                whisper_elapsed = time.monotonic() - secondary_started
                if self._invalidated(generation):
                    return ""
                if not plausible(secondary, duration, self.cfg.blocklist, probe=probe):
                    secondary = ""
            text, reason = select_candidate(primary, secondary, confidence, duration, probe=probe)
            if self._invalidated(generation):
                return ""
            self.last_selection = reason
            update_metrics(stages={"dual_haas_s": round(haas_elapsed, 3),
                                   "dual_whisper_s": round(whisper_elapsed, 3)})
            self.last_raw_transcript = ("haas=" + primary[:160] + " | whisper=" + secondary[:160])
            self.last_rejection = None if text else "no_supported_speech"
            self.last_language = "Japanese" if text else None
            record_event("dual_asr_selection", selected=reason, duration_s=round(duration, 2),
                         primary_chars=len(primary), secondary_chars=len(secondary))
            return text

    def _invalidated(self, generation: int) -> bool:
        with self._lock:
            return self._closed or generation != self._generation

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
        self.primary.close()
        if self.secondary is not None:
            self.secondary.close()
            self.secondary = None
