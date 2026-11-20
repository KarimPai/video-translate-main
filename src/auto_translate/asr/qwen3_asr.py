"""Qwen3-ASR speech recognition via llama.cpp HTTP server.

Qwen3-ASR is a multimodal model that transcribes audio to text. It requires
both a main model GGUF and an audio projector GGUF (mmproj). Audio is sent
as base64 WAV via the ``/v1/chat/completions`` endpoint with ``input_audio``
content type.

Supports 52 languages; for Japanese transcription, an assistant prefix with
``language Japanese<asr_text>`` guides the model.
"""
from __future__ import annotations

import base64
import logging
import re
import threading
import time
import unicodedata
from collections import deque

import httpx
import numpy as np

from ..asr.haas import wav_bytes
from ..config import AsrConfig
from ..llama_server import (
    LlamaServer,
    detect_device,
    require_gguf,
    resolve_executable,
)
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running
from .context_echo import speech_key
from .quality_guard import transcript_problem

log = logging.getLogger(__name__)

_QWEN3_SYSTEM = (
    "You are a speech recognition assistant. Transcribe the audio exactly as "
    "spoken. Do not add commentary, translations, or descriptions."
)

# Default llama-server path for Qwen3-ASR
LLAMA_SERVER_DEFAULT = ".runtime/llama-b12318-cuda12/llama-server.exe"


class Qwen3ASRError(RuntimeError):
    """The Qwen3-ASR server failed repeatedly; transcription cannot continue."""


class Qwen3ASR:
    """Qwen3-ASR speech recognition via llama.cpp HTTP server.

    Uses the multimodal chat completions API with input_audio content type.
    Requires both a model GGUF and an audio projector (mmproj) GGUF.
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
        self._restarts = 0
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
        )
        try:
            self.server.start()
            # Warmup with a short silent audio to reserve GPU memory
            self._warmup()
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

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        """Transcribe 16 kHz mono float32 audio to Japanese text."""
        gen = self._generation
        duration_s = len(audio) / 16000
        if duration_s < 0.05:
            return ""

        with self._inference_lock:
            try:
                text = self._transcribe_impl(audio, probe)
            except Exception as exc:
                self._failures += 1
                if self._failures >= 3 and self._restarts < 2:
                    log.warning("Qwen3-ASR failed %d times, attempting restart", self._failures)
                    try:
                        self._restart()
                        text = self._transcribe_impl(audio, probe)
                    except Exception:  # noqa: BLE001
                        raise Qwen3ASRError(f"Qwen3-ASR restart failed: {exc}") from exc
                else:
                    raise Qwen3ASRError(f"Qwen3-ASR transcription failed: {exc}") from exc

        if self._generation != gen:
            return ""

        # Quality guard
        duration_s = len(audio) / 16000
        reason = transcript_problem(text, duration_s)
        if reason:
            self.last_rejection = reason
            if reason in ("context_echo", "repeated_output"):
                with self._state_lock:
                    key = speech_key(text)
                    if key and key not in self._accepted_keys:
                        self._accepted_keys.append(key)
            return ""

        # Accept
        with self._state_lock:
            key = speech_key(text)
            if key:
                self._accepted_keys.append(key)
            self._recent.append(text)
            self.last_rejection = None

        return text

    def cancel_pending(self) -> None:
        """Invalidate in-flight transcription requests."""
        self._generation += 1

    def close(self) -> None:
        """Stop the server and release resources."""
        self._closed = True
        self.server.close()
        update_metrics(workers={"asr": "closed"})

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

    def recover_context_echo(self, audio: np.ndarray) -> str:
        """Retry transcription without context to recover from echo."""
        return self._transcribe_impl(audio, probe=False, no_context=True)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _warmup(self) -> None:
        """Warmup with silent audio to reserve GPU memory."""
        warmup_s = 2.0
        silent = np.zeros(int(16000 * warmup_s), dtype=np.float32)
        try:
            self._transcribe_impl(silent, probe=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("Qwen3-ASR warmup failed (may still work): %s", exc)

    def _transcribe_impl(
        self,
        audio: np.ndarray,
        probe: bool = False,
        no_context: bool = False,
    ) -> str:
        """Send audio to the server and return transcribed text."""
        t0 = time.perf_counter()

        # Encode audio as WAV base64
        wav = wav_bytes(audio, normalize=True)
        b64 = base64.b64encode(wav).decode("ascii")

        # Build messages for multimodal chat completions
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": "wav",
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "language Japanese<asr_text>",
                    }
                ],
            },
        ]

        payload = {
            "messages": messages,
            "temperature": 0.0 if not probe else 0.3,
            "max_tokens": 256,
            "stream": False,
        }

        timeout = self.cfg.response_timeout_s
        try:
            resp = self.server.client.post(
                f"{self.server.base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            http_s = time.perf_counter() - t0
            resp.raise_for_status()
            body = resp.json()
        except httpx.TimeoutException:
            log.warning("Qwen3-ASR HTTP timeout after %.1fs", http_s)
            update_metrics(counters={"asr_stall_count": 1})
            raise Qwen3ASRError(f"Qwen3-ASR timeout after {http_s:.1f}s")
        except httpx.HTTPError as exc:
            log.error("Qwen3-ASR HTTP error: %s", exc)
            raise Qwen3ASRError(str(exc)) from exc

        # Extract text from response
        try:
            content = body["choices"][0]["message"]["content"]
            text = content.strip()
        except (KeyError, IndexError) as exc:
            raise Qwen3ASRError(f"Unexpected response format: {body}") from exc

        # Clean up
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
        text = re.sub(r"<asr_text>", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()

        self.last_raw_transcript = text
        self._failures = 0

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

    def _restart(self) -> None:
        """Restart the server after a failure."""
        self._restarts += 1
        log.warning("Restarting Qwen3-ASR server (attempt %d)", self._restarts)
        self.server.close()
        self.server.start()
        self._warmup()
        self._failures = 0
        # Clear stale state to prevent context echo detection issues after restart
        with self._state_lock:
            self._accepted_keys.clear()
            self._recent.clear()
