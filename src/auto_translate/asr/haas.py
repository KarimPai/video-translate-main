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
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running
from .base import ASR
from .quality_guard import transcript_problem
from .context_echo import speech_key
from ..speech_server import SpeechCppServer, SpeechServerError, resolve_speech_executable

log = logging.getLogger(__name__)

_JAPANESE_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


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
        if active_rms >= 0.001 and rms < 0.02:
            gain = max(1.0, min(8.0, 0.02 / rms, 0.95 / float(np.max(np.abs(value)))))
            value = value * gain
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.ascontiguousarray(value * 32767, dtype="<i2").tobytes())
    return buffer.getvalue()


def is_japanese_text(text: str) -> bool:
    """Check if text contains Japanese script characters."""
    return bool(_JAPANESE_SCRIPT.search(text))


class HaasASR:
    """Haas-ja-v1.0 ASR via speech.cpp HTTP server."""

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
        self._restarts = 0

        executable = resolve_speech_executable(cfg.server_path if cfg.server_path != ".runtime/llama-b12318-cuda12/llama-server.exe" else None)
        device = "vulkan"
        if cfg.cuda_device is not None:
            device = "cuda"

        self.server = SpeechCppServer(
            model_name="haas-ja-v1.0",
            executable=executable,
            device=device,
            threads=cfg.threads,
            startup_timeout_s=cfg.startup_timeout_s,
        )
        try:
            self.server.start()
            # Warmup with silent audio to reserve encoder workspace
            warmup_s = 16.0
            self.server.transcribe(
                wav_bytes(np.zeros(int(16000 * warmup_s), np.float32)),
                timeout=cfg.startup_timeout_s,
            )
            check_running()
        except BaseException:
            self.close()
            raise
        update_metrics(workers={"asr": "ready (Haas-ja-v1.0)"})
        log.info("Haas-ja-v1.0 ready (pid=%s, device=%s, context %d chars)",
                 self.server.process.pid, device, len(self.context))

    def set_context(self, context: str) -> None:
        with self._state_lock:
            self.context = context.strip()[: self.cfg.max_context_chars]

    def cancel_pending(self) -> None:
        with self._state_lock:
            self._generation += 1
            self.last_raw_transcript = ""
            self._recent.clear()
            self._accepted_keys.clear()

    def remember(self, text: str) -> None:
        """Bias the next decode toward continuity with recently accepted speech."""
        with self._state_lock:
            self._recent.append(text[-120:])
            self._accepted_keys.append(speech_key(text))

    def forget(self) -> None:
        with self._state_lock:
            self._recent.clear()

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
                text = self.server.transcribe(wav, timeout=timeout)
            except (httpx.HTTPError, SpeechServerError, ValueError) as exc:
                if self._closed:
                    return ""
                return self._failed(exc)
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
        wav = wav_bytes(audio, normalize=not probe)
        started = time.monotonic()
        with self._inference_lock:
            if generation != self._generation or self._closed:
                return ""
            check_running()
            if not self.server.alive():
                self._restart("server exited")
            try:
                text = self.server.transcribe(wav, timeout=timeout)
            except (httpx.HTTPError, SpeechServerError, ValueError) as exc:
                if self._closed:
                    return ""
                return self._failed(exc)
        self._failures = 0
        with self._state_lock:
            if generation != self._generation:
                return ""
            self.last_raw_transcript = text
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))

        problem = (("no_speech" if not text else None)
                   or ("truncated" if not text and duration_s > 0.5 else None)
                   or transcript_problem(text, duration_s)
                   or ("blocklist" if any(word in text for word in self.cfg.blocklist) else None))

        if not problem and text and is_japanese_text(text):
            key = speech_key(text)
            with self._state_lock:
                repeated = len(key) >= 10 and key in self._accepted_keys
            if repeated:
                problem = "repeated_output"

        if problem:
            self.last_rejection = problem
            log.debug("Discarding ASR output (%s): %s", problem, " ".join(text[:200].split()))
            return ""
        elapsed = time.monotonic() - started
        update_metrics(stages={"asr_s": round(elapsed, 3), "asr_rtf": round(elapsed / max(duration_s, 1e-3), 3)})
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

    def _restart(self, reason: str) -> None:
        if self._restarts >= 2:
            raise HaasError(f"Haas-ja-v1.0 server stopped again ({reason}). {self.server.log_tail()}")
        self._restarts += 1
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
