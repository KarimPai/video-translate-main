"""Qwen3-ASR-1.7B served by an owned llama.cpp server, one VAD utterance at a time.

Audio is sent inline as a base64 WAV chat-completion part. The system message
carries free-form context (stream title, talent names, slang), which Qwen3-ASR
uses for contextual biasing. The model emits ``language X<asr_text>...``; the
language tag doubles as a free spoken-language check.
"""
from __future__ import annotations

import base64
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
from ..llama_server import LlamaServer, LlamaServerError, require_gguf, resolve_executable
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running
from .context_echo import active_seconds, context_match, speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_TAGGED = re.compile(r"^\s*language\s+([A-Za-z]+)\s*<asr_text>(.*)\Z", re.S)
_KANA = re.compile(r"[\u3040-\u30ff]")
_JAPANESE_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


class AsrError(RuntimeError):
    """The ASR server failed repeatedly; translation cannot continue."""


def parse_output(raw: str) -> tuple[str | None, str]:
    """Split Qwen3-ASR output into (language or None, transcript)."""
    match = _TAGGED.match(raw or "")
    if not match:
        return None, (raw or "").replace("<asr_text>", "").strip()
    language = match.group(1)
    return (None if language.lower() == "none" else language), match.group(2).strip()


def is_japanese(language: str | None, text: str) -> bool:
    if language == "Japanese":
        return True
    # Short kana-bearing lines are occasionally tagged Chinese; kana is decisive.
    if language in ("Chinese", "Cantonese"):
        return bool(_KANA.search(text))
    return language is None and bool(_JAPANESE_SCRIPT.search(text))


def wav_bytes(audio: np.ndarray, *, normalize: bool = True) -> bytes:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("ASR requires finite mono float32 audio at 16000 Hz")
    value = np.clip(value, -1, 1)
    if normalize and value.size:
        power = value * value
        rms = float(np.sqrt(np.mean(power)))
        # Silent VAD padding must not push a short reply below the gain floor.
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


class Qwen3ASR:
    def __init__(self, cfg: AsrConfig, context: str = "") -> None:
        self.cfg = cfg
        self.context = context.strip()[: cfg.max_context_chars]
        self._recent: deque[str] = deque(maxlen=cfg.recent_context_lines)
        self._accepted_keys: deque[str] = deque(maxlen=32)
        self.last_match_score = 0.0
        self._review_context = ""
        self.last_raw_transcript = ""
        self.last_rejection: str | None = None
        self.last_language: str | None = None
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._failures = 0
        self._restarts = 0
        model = require_gguf(cfg.model_path, "Qwen3-ASR model")
        mmproj = require_gguf(cfg.mmproj_path, "Qwen3-ASR audio projector (mmproj)")
        args = ["-m", str(model), "--mmproj", str(mmproj), "-ngl", "99", "-c", str(cfg.context_size),
                "-np", "1", "-fa", "on", "-fit", "off", "--jinja", "-t", str(cfg.threads)]
        if cfg.cuda_device is not None:
            args += ["--device", f"CUDA{cfg.cuda_device}"]
        self.server = LlamaServer("asr", resolve_executable(cfg.server_path), args,
                                  startup_timeout_s=cfg.startup_timeout_s)
        try:
            self.server.start()
            # Reserve encoder workspace for the default VOD hard cap before budgeting MT.
            warmup_s = 16.0
            self._request(wav_bytes(np.zeros(int(16000 * warmup_s), np.float32)), warmup_s,
                          timeout=cfg.startup_timeout_s, system="")
            check_running()
        except BaseException:
            self.close()
            raise
        update_metrics(workers={"asr": "ready (Qwen3-ASR-1.7B)"})
        log.info("Qwen3-ASR ready (pid=%s, context %d chars)", self.server.process.pid, len(self.context))

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

    def _system_prompt(self) -> str:
        with self._state_lock:
            recent = list(self._recent)
        if not recent:
            return self.context
        return "\n".join(part for part in (self.context, "直前の発話：", *recent) if part)

    def _request(self, wav: bytes, duration_s: float, *, timeout: float, system: str) -> dict:
        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
        }]})
        body = {"messages": messages, "temperature": 0.0, "stream": False,
                "max_tokens": min(512, 48 + int(duration_s * 16))}
        if not system:
            body["cache_prompt"] = False
        return self.server.chat(body, timeout=timeout)

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        return self._decode(audio, probe, self._system_prompt(), self.cfg.response_timeout_s)

    def recover_context_echo(self, audio: np.ndarray, probe: bool, vad_confidence: float | None,
                             timeout: float) -> str:
        """Decode identical PCM without hints/history. Retain a spoken list if corroborated."""
        original = speech_key(parse_output(self.last_raw_transcript)[1])
        context = self._review_context
        text = self._decode(audio, probe, "", timeout, recovery=True)
        if not text:
            return ""
        key = speech_key(text)
        active = active_seconds(audio)
        # A language tag and an energy floor are weak individually. Require both,
        # plausible duration, and substantially stronger support for a copied list.
        supported = self.last_language == "Japanese" and active >= .04 and len(key) <= 10 + 18 * active
        if probe or (vad_confidence is not None and vad_confidence < .35):
            # With little VAD support, an unrelated second guess is uncertain.
            supported = supported and len(key) >= 2 and key in original
        if context_match(text, context) or key == original:
            supported = (supported and key == original and active >= len(key) / 16
                         and not probe and (vad_confidence is None or vad_confidence >= .5))
        if re.match(r"^(用語|人名|直前の発話)[:：]", text) and text[:2] in context:
            supported = False
        if not supported:
            self.last_rejection = "unconfirmed_recovery"
            return ""
        return text

    def _decode(self, audio: np.ndarray, probe: bool, system: str, timeout: float,
                *, recovery: bool = False) -> str:
        with self._state_lock:
            generation = self._generation
            self.last_raw_transcript = ""
            self.last_rejection = None
            self.last_language = None
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
                payload = self._request(wav, duration_s, timeout=timeout, system=system)
            except (httpx.HTTPError, LlamaServerError, ValueError) as exc:
                if self._closed:
                    return ""
                return self._failed(exc)
        self._failures = 0
        choice = payload["choices"][0]
        raw = (choice.get("message") or {}).get("content") or ""
        with self._state_lock:
            if generation != self._generation:
                return ""
            self.last_raw_transcript = raw
        language, text = parse_output(raw)
        self.last_language = language
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
        problem = (("no_speech" if not text else None)
                   or (None if is_japanese(language, text) else f"language:{language}")
                   or ("probe_unconfirmed" if probe and language != "Japanese" else None)
                   or ("truncated" if choice.get("finish_reason") == "length" else None)
                   or transcript_problem(text, duration_s)
                   or ("blocklist" if any(word in text for word in self.cfg.blocklist) else None))
        if not recovery and text and is_japanese(language, text):
            score = context_match(text, system)
            key = speech_key(text)
            with self._state_lock:
                repeated = len(key) >= 10 and key in self._accepted_keys
            if score:
                problem = "context_echo"
            elif repeated and (probe or len(key) > 10 + 18 * active_seconds(audio)):
                problem, score = "repeated_output", 1.0
            if score:
                self.last_match_score = score
                self._review_context = system
        if problem:
            self.last_rejection = problem
            log.debug("Discarding ASR output (%s): %s", problem, " ".join(raw[:200].split()))
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
            raise AsrError(f"Qwen3-ASR failed three consecutive utterances: {exc}") from exc
        return ""

    def _restart(self, reason: str) -> None:
        if self._restarts >= 2:
            raise AsrError(f"Qwen3-ASR server stopped again ({reason}). {self.server.log_tail()}")
        self._restarts += 1
        log.warning("Restarting Qwen3-ASR server (%s)", reason)
        update_metrics(workers={"asr": "restarting"})
        self.server.start()
        update_metrics(workers={"asr": "ready (Qwen3-ASR-1.7B)"})

    def close(self) -> None:
        self._closed = True
        with self._state_lock:
            self._generation += 1
        server = getattr(self, "server", None)
        if server is not None:
            server.close()
