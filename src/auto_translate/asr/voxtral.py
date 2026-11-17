"""Persistent audio.cpp Voxtral offline ASR, one complete VAD utterance at a time.

Interface verified against audio.cpp v0.9.5, commit
61c7838baf1f7f34769b5b7bd369c0d78edf9535. The offline HTTP route has no
backend cancellation API: seeking invalidates the result and lets the current
request drain. Stopping or a failed request terminates only our server process.
"""
from __future__ import annotations

import http.client
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

from ..config import AsrConfig
from ..runtime_diagnostics import update_metrics
from ..text_filter import trim_repetition_tail
from ..windows_subprocess import check_running, hidden_console_flags, owned_popen
from .quality_guard import potential_japanese_transcript, transcript_problem

log = logging.getLogger(__name__)
_MAX_RESPONSE_BYTES = 1_048_576
_MODEL_ID = "voxtral"


class VoxtralError(RuntimeError):
    """An actionable model/runtime failure; never silently switch ASR models."""


class VoxtralOOMError(VoxtralError):
    """CUDA could not allocate the model or inference workspace."""


def _runtime_error(message: str) -> VoxtralError:
    compact = message.lower()
    if any(word in compact for word in (
            "out of memory", "cuda_error_out_of_memory", "cudamalloc",
            "failed to allocate", "allocation failed", "insufficient_memory")):
        return VoxtralOOMError(
            "Voxtral CUDA memory allocation failed. Reduce PLaMo GPU offloading "
            "or close other GPU workloads, then restart translation. "
            f"The Q4_K ASR model and selected video resolution were preserved. {message[-1200:]}"
        )
    return VoxtralError(f"Voxtral audio.cpp failed: {message[-1600:]}")


def _audio16(audio: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Voxtral requires finite mono float32 audio at 16000 Hz")
    if value.size > 60 * 16000:
        raise ValueError("Voxtral utterance exceeds 60 seconds; check VAD segmentation")
    value = np.clip(value, -1, 1)
    if normalize and value.size:
        rms = float(np.sqrt(np.mean(value * value)))
        if 0.001 <= rms < 0.02:
            # Bound gain to 12 dB and leave silence and near-silence untouched.
            gain = max(1.0, min(4.0, 0.02 / rms, 0.95 / float(np.max(np.abs(value)))))
            value = value * gain
    return np.ascontiguousarray(value * 32767, dtype="<i2")


def _server_config(cfg: AsrConfig, model: Path, port: int) -> dict:
    return {
        "host": "127.0.0.1", "port": port,
        "backend": "cuda", "device": cfg.cuda_device, "threads": cfg.threads,
        "lazy_load": False, "max_loaded_models": 1, "idle_unload_ms": 0,
        "max_request_body_bytes": 2_097_152,
        "models": [{
            "id": _MODEL_ID, "family": "voxtral_realtime",
            "path": str(model), "task": "asr", "mode": "offline",
            "default_request_options": {
                "do_sample": False, "temperature": 1.0,
                "top_p": 1.0, "top_k": 50, "seed": 1234,
            },
        }],
    }


class VoxtralASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.process = None
        self.last_failed = False
        self.last_stop = None
        self.last_raw_transcript = ""
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._connection = None
        self._temporary = None
        self._log_stream = None

        executable = Path(cfg.voxtral_server_path).expanduser()
        if not executable.is_file():
            found = shutil.which(cfg.voxtral_server_path)
            if found:
                executable = Path(found)
            else:
                raise VoxtralError(
                    f"audio.cpp CUDA server not found: {cfg.voxtral_server_path}. "
                    "Run Setup.bat to install the pinned Windows CUDA runtime."
                )
        model = Path(cfg.voxtral_model_path).expanduser().resolve()
        if not model.is_file():
            raise VoxtralError(f"Voxtral Q4_K checkpoint not found: {model}. Run Setup.bat.")
        with model.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise VoxtralError(f"Invalid Voxtral GGUF checkpoint: {model}. Run Setup.bat to repair it.")

        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="video-translate-voxtral-")
            self._temp_path = Path(self._temporary.name)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", 0))
                self._port = reservation.getsockname()[1]
            config_path = self._temp_path / "server.json"
            config_path.write_text(json.dumps(_server_config(cfg, model, self._port)), encoding="utf-8")
            self._log_path = Path("logs/voxtral-server.log").resolve()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open("wb")
            self.process = owned_popen(
                [str(executable.resolve()), "--config", str(config_path), "--no-ui"],
                cwd=str(executable.resolve().parent), stdin=subprocess.DEVNULL,
                stdout=self._log_stream, stderr=subprocess.STDOUT,
                creationflags=hidden_console_flags(),
            )
            self._wait_ready()
        except BaseException:
            self.close()
            raise
        update_metrics(workers={"voxtral": "ready (CUDA Q4_K, offline)"})
        log.info("Voxtral ready: CUDA device %s, persistent offline Q4_K server pid=%s",
                 cfg.cuda_device, self.process.pid)

    def _log_tail(self) -> str:
        try:
            with self._log_path.open("rb") as stream:
                stream.seek(max(0, self._log_path.stat().st_size - 8192))
                return stream.read(8192).decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            return "See logs/voxtral-server.log for native loader diagnostics."

    def _request_json(self, method: str, path: str, body=None, *, timeout: float):
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        with self._state_lock:
            if self._closed:
                raise VoxtralError("Voxtral server is closed")
            self._connection = connection
        try:
            data = None if body is None else json.dumps(body).encode("utf-8")
            connection.request(method, path, body=data,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(payload) > _MAX_RESPONSE_BYTES:
                raise VoxtralError("Voxtral returned an oversized response")
            if response.status != 200:
                raise _runtime_error(f"HTTP {response.status}: {payload.decode('utf-8', errors='replace')}")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise VoxtralError("Voxtral returned a non-object response")
            return parsed
        finally:
            connection.close()
            with self._state_lock:
                if self._connection is connection:
                    self._connection = None

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            check_running()
            if self.process.poll() is not None:
                raise _runtime_error(
                    f"CUDA server exited during startup (code {self.process.returncode}). "
                    "Check NVIDIA driver and the matching cudart/cuBLAS/cuFFT DLL archive. "
                    + self._log_tail())
            try:
                status = self._request_json("GET", "/health", timeout=0.25)
            except (OSError, http.client.HTTPException, ValueError):
                time.sleep(0.05)
                continue
            if status.get("status") != "ok" or status.get("backend") != "cuda":
                raise VoxtralError(f"Unexpected Voxtral CUDA server readiness: {status}")
            models = self._request_json("GET", "/v1/models", timeout=1.0)
            if not any(entry.get("id") == _MODEL_ID for entry in models.get("data", [])
                       if isinstance(entry, dict)):
                raise VoxtralError("Voxtral server did not register its offline ASR model")
            if self.process.poll() is not None:
                raise _runtime_error("Voxtral exited while becoming ready. " + self._log_tail())
            return
        raise _runtime_error(
            f"CUDA model startup exceeded {self.cfg.startup_timeout_s:g}s. " + self._log_tail())

    def cancel_pending(self) -> None:
        """Invalidate seek/session results without unloading the model weights."""
        with self._state_lock:
            self._generation += 1
            self.last_raw_transcript = ""

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        with self._state_lock:
            generation = self._generation
            self.last_raw_transcript = ""
        pcm = _audio16(audio, normalize=not probe)
        if not pcm.size:
            return ""
        started = time.monotonic()
        with self._inference_lock:
            with self._state_lock:
                if generation != self._generation:
                    return ""
                if self._closed:
                    raise VoxtralError("Voxtral server is closed; restart translation")
                process = self.process
            check_running()
            if process is None or process.poll() is not None:
                error = _runtime_error("CUDA server exited. " + self._log_tail())
                self.close()
                raise error
            wav_path = self._temp_path / "utterance.wav"
            try:
                with wave.open(str(wav_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(pcm.tobytes())
                result = self._request_json(
                    "POST", "/v1/audio/transcriptions",
                    {"model": _MODEL_ID, "audio": str(wav_path)},
                    timeout=self.cfg.response_timeout_s,
                )
                text = result.get("text")
                if not isinstance(text, str):
                    raise VoxtralError("Voxtral returned no transcript text")
            except (OSError, http.client.HTTPException, ValueError, VoxtralError) as exc:
                with self._state_lock:
                    already_closed = self._closed
                self.last_failed = True
                native = self._log_tail()
                # Closing an HTTP request does not cancel native inference.
                # Even a seek-invalidated timeout must retire this server before
                # subsequent inference or GPU model-management operations.
                self.close()
                update_metrics(workers={"voxtral": "failed"})
                if already_closed:
                    return ""
                if isinstance(exc, TimeoutError):
                    raise TimeoutError(
                        f"Voxtral offline inference exceeded {self.cfg.response_timeout_s:g}s; "
                        "the owned server was stopped. Restart translation or increase the ASR response timeout."
                    ) from exc
                raise _runtime_error(f"{exc}. {native}") from exc
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._state_lock:
                if generation != self._generation or self._closed:
                    return ""
                self.last_raw_transcript = text
            self.last_failed = False
            text = trim_repetition_tail(unicodedata.normalize("NFC", text).strip())
            problem = transcript_problem(text, pcm.size / 16000)
            if problem or not potential_japanese_transcript(text) or any(word in text for word in self.cfg.blocklist):
                log.debug("Discarding Voxtral output (%s): %s", problem or "language/blocklist", text)
                return ""
            elapsed = time.monotonic() - started
            update_metrics(stages={"voxtral_s": round(elapsed, 3),
                                   "voxtral_rtf": round(elapsed / (pcm.size / 16000), 3)},
                           workers={"voxtral": "ready (CUDA Q4_K, offline)"})
            return text

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            process, self.process = self.process, None
            connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._log_stream is not None:
            self._log_stream.close()
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except OSError:
                log.debug("Voxtral temporary directory cleanup deferred", exc_info=True)
