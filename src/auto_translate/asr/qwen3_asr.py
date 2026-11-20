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
from ..windows_subprocess import check_running, stopping
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
        """Transcribe 16 kHz mono float32 audio to Japanese text.

        Recoverable failures (timeout, connection) are retried a bounded number
        of times before a server restart is attempted. Permanent failures and
        exhausted retries raise a clear terminal Qwen3ASRError so a single
        transient error cannot terminate the whole session.

        Uses a single monotonic deadline per utterance so the total time never
        exceeds the configured budget, even with retries.
        """
        gen = self._generation
        duration_s = len(audio) / 16000
        if duration_s < 0.05:
            return ""

        max_retries = 3
        restart_limit = 2
        backoff_s = 0.5

        # Single monotonic deadline per utterance
        budget = self.cfg.response_timeout_s
        deadline = time.monotonic() + budget

        with self._inference_lock:
            text: str | None = None
            last_exc: Exception | None = None
            attempts = 0
            while attempts <= max_retries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_exc = TimeoutError("utterance budget exceeded")
                    break
                try:
                    text = self._transcribe_impl(audio, probe, timeout=remaining)
                    self._failures = 0
                    break
                except httpx.TimeoutException as exc:
                    last_exc = exc
                    self._failures += 1
                    attempts += 1
                    log.warning("Qwen3-ASR timeout (attempt %d/%d)", attempts, max_retries + 1)
                    if attempts <= max_retries:
                        sleep_time = min(backoff_s * attempts, deadline - time.monotonic())
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_exc = exc
                    self._failures += 1
                    attempts += 1
                    log.warning("Qwen3-ASR connection failure (attempt %d/%d)", attempts, max_retries + 1)
                    if attempts <= max_retries:
                        sleep_time = min(backoff_s * attempts, deadline - time.monotonic())
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue
                    break
                except httpx.HTTPError as exc:
                    last_exc = exc
                    self._failures += 1
                    log.error("Qwen3-ASR invalid response: %s", exc)
                    # Permanent error — no retry
                    break
                except Qwen3ASRError as exc:
                    last_exc = exc
                    self._failures += 1
                    log.error("Qwen3-ASR permanent failure: %s", exc)
                    # Permanent error — no retry
                    break

            if (
                text is None
                and self._restarts < restart_limit
                and not self._closed
                and not stopping.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_exc = TimeoutError("utterance budget exceeded after restart")
                else:
                    try:
                        self._restart()
                        text = self._transcribe_impl(audio, probe, timeout=remaining)
                        self._failures = 0
                    except Exception as restart_exc:  # noqa: BLE001
                        raise Qwen3ASRError(
                            f"Qwen3-ASR restart failed after {attempts} attempt(s): {restart_exc}"
                        ) from last_exc

            if text is None:
                raise Qwen3ASRError(
                    f"Qwen3-ASR transcription failed after {attempts} attempt(s): {last_exc}"
                ) from last_exc

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

    def recover_context_echo(self, audio: np.ndarray, probe: bool = False,
                             vad_confidence: float | None = None,
                             timeout: float = 0.0) -> str:
        """Retry transcription to recover from context echo.

        NOTE: Qwen3-ASR does not send user-provided context in the request
        payload, so context removal is not applicable. This method retries
        the transcription with the remaining timeout budget.

        Enforces the supplied remaining timeout and never starts recovery when
        the budget has expired. Cancellation is propagated via the generation
        counter and the closed flag.
        """
        if timeout <= 0:
            return ""
        with self._state_lock:
            if self._closed:
                return ""
            gen = self._generation
        if not len(audio):
            return ""
        try:
            text = self._transcribe_impl(audio, probe=probe, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("Qwen3-ASR context echo recovery failed: %s", exc)
            return ""
        if self._generation != gen or self._closed:
            return ""
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            self.last_rejection = "no_speech"
            return ""
        return text

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
        timeout: float | None = None,
    ) -> str:
        """Send audio to the server and return transcribed text.

        ``timeout`` overrides the configured response timeout (used by context
        echo recovery to enforce the remaining time budget).

        NOTE: ``no_context`` is a no-op. Qwen3-ASR does not send user-provided
        context in the request payload, so there is nothing to remove. The
        parameter is kept for API compatibility with other ASR backends.
        """
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

        timeout = timeout if timeout is not None else self.cfg.response_timeout_s
        http_s = 0.0
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
            elapsed_on_timeout = time.perf_counter() - t0
            log.warning("Qwen3-ASR HTTP timeout after %.1fs", elapsed_on_timeout)
            update_metrics(counters={"asr_stall_count": 1})
            raise
        except httpx.HTTPError as exc:
            elapsed_on_error = time.perf_counter() - t0
            log.error("Qwen3-ASR HTTP error after %.1fs: %s", elapsed_on_error, exc)
            raise

        # Extract text from response
        try:
            content = body["choices"][0]["message"]["content"]
            text = content.strip()
        except (KeyError, IndexError) as exc:
            raise Qwen3ASRError(f"Unexpected response format: {body}") from exc

        # Clean up protocol metadata that must never reach subtitles
        # Remove "language Japanese" prefix (model protocol metadata, not spoken text)
        text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"<asr_text>", "", text).strip()
        text = " ".join(unicodedata.normalize("NFC", text).split())
        text = trim_repetition_tail("".join(c for c in text if unicodedata.category(c) != "Cc"))
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
        """Restart the server after a failure.

        Uses reset_for_restart() to clear the closed flag so start() can run
        again. Bounds restart attempts and never revives the server when the
        application is shutting down or the instance is closed.
        """
        if self._restarts >= 2:
            raise Qwen3ASRError("Qwen3-ASR restart limit reached (2 attempts); server is unstable")
        if stopping.is_set() or self._closed:
            raise Qwen3ASRError(
                "Qwen3-ASR restart skipped: application is shutting down or server is closed"
            )
        self._restarts += 1
        log.warning("Restarting Qwen3-ASR server (attempt %d)", self._restarts)
        # reset_for_restart stops the old process, clears _closed flag, and recreates the HTTP client
        self.server.reset_for_restart()
        self.server.start()
        self._warmup()
        self._failures = 0
        # Clear stale state to prevent context echo detection issues after restart
        with self._state_lock:
            self._accepted_keys.clear()
            self._recent.clear()
