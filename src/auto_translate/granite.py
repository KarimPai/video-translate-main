"""Granite Speech 5.0: local audio -> English via llama.cpp, no ASR/MT cascade.

IBM's documented llama-server endpoint:
    POST /v1/audio/transcriptions
    multipart fields: model, prompt, file (16 kHz mono WAV)
Model: ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

from .config import GraniteConfig
from .process_cleanup import _close_job, _create_kill_job
from .types import SAMPLE_RATE

log = logging.getLogger(__name__)


def encode_wav(audio: np.ndarray) -> bytes:
    """Encode 16kHz mono float32 PCM as signed 16-bit RIFF WAV."""
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not pcm.size:
        raise ValueError("cannot translate empty audio")
    if not np.all(np.isfinite(pcm)):
        raise ValueError("audio contains non-finite samples")
    samples = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def build_multipart(model: str, prompt: str, wav_bytes: bytes) -> tuple[bytes, str]:
    """Standards-compliant multipart body without requests/httpx dependencies."""
    boundary = "----granite-speech-" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in (("model", model), ("prompt", prompt)):
        parts.append(
            (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
             f"{value}\r\n").encode("utf-8")
        )
    parts.append(
        (f"--{boundary}\r\n"
         'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
         "Content-Type: audio/wav\r\n\r\n").encode("ascii")
        + wav_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _clean_result(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Granite returned a non-JSON-object reply")
    value = data.get("text")
    if not isinstance(value, str):
        raise ValueError(f"Granite did not return a text field: {str(data)[:250]}")
    cleaned = " ".join(value.strip().splitlines()).strip()
    for label in ("Translation:", "English:", "EN:"):
        if cleaned.casefold().startswith(label.casefold()):
            cleaned = cleaned[len(label):].strip()
    return cleaned


class GraniteSpeech5:
    """Own exactly one Q8 local llama-server process; never adopt a foreign server."""

    def __init__(self, cfg: GraniteConfig) -> None:
        self.cfg = cfg
        parsed = urllib.parse.urlsplit(cfg.base_url)
        if (parsed.scheme != "http"
                or parsed.hostname not in ("localhost", "127.0.0.1")
                or parsed.username or parsed.password
                or parsed.path not in ("", "/")):
            raise ValueError("Granite server must be a local http://127.0.0.1:PORT URL")
        self.host = "127.0.0.1"
        self.port = parsed.port or 9797
        self.base_url = f"http://{self.host}:{self.port}"
        self.process: subprocess.Popen | None = None
        self._job = None
        self._server_log = None
        self._closed = False
        self._registered = False

        self._start_server()
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def _start_server(self) -> None:
        # An unrelated Ollama/PLaMo/llama.cpp service must not be mistaken for
        # this model, and must never be terminated on shutdown.
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                raise RuntimeError(
                    f"Port {self.port} is already occupied. Close the other server "
                    "or set [granite] base_url to another localhost port."
                )
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            if isinstance(exc, RuntimeError):
                raise

        binary = self.cfg.llama_server_path or shutil.which("llama-server") or shutil.which("llama-server.exe")
        if not binary:
            raise RuntimeError(
                "Granite Speech 5 requires a recent llama.cpp llama-server.exe "
                "with multimodal audio support on PATH. Install/update llama.cpp."
            )
        command = [
            str(binary),
            "-hf", self.cfg.model_id,
            "-c", str(self.cfg.context_tokens),
            "-ngl", str(self.cfg.gpu_layers),
            "--host", self.host, "--port", str(self.port),
            "-np", "1",
        ]
        Path("logs").mkdir(exist_ok=True)
        self._server_log = Path("logs/granite5-server.log").open("a", encoding="utf-8")
        log.info("Starting Granite Speech 5.0 2B Q8 (local llama.cpp, port %d)", self.port)
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=self._server_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._job = _create_kill_job(self.process)
            atexit.register(self.close)
            self._registered = True
        except Exception:
            self.close()
            raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "Granite llama-server exited during startup. "
                    "See logs/granite5-server.log (check llama.cpp version and model download)."
                )
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=1.5) as res:
                    health = json.load(res)
                    if res.status == 200 and health.get("status") == "ok":
                        log.info("Granite Speech 5 ready at %s", self.base_url)
                        return
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(
            f"Granite model startup timed out after {self.cfg.startup_timeout_s}s; "
            "see logs/granite5-server.log"
        )

    def translate_audio(self, audio: np.ndarray) -> str:
        """Run exactly one direct audio->English decode for a VAD utterance."""
        wav_bytes = encode_wav(audio)
        body, content_type = build_multipart(
            self.cfg.model_id, self.cfg.prompt, wav_bytes,
        )
        request = urllib.request.Request(
            self.base_url + "/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.request_timeout_s) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(600).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Granite transcription endpoint HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Granite audio-to-English request failed: {exc}"
            ) from exc
        return _clean_result(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._registered:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
        if self._job is not None:
            _close_job(self._job)
            self._job = None
        if self.process is not None and self.process.poll() is None:
            # Closing a Windows Job Object may already have terminated it.
            try:
                self.process.terminate()
            except (OSError, ProcessLookupError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except (OSError, ProcessLookupError):
                    pass
                self.process.wait(timeout=3)
        self.process = None
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None
