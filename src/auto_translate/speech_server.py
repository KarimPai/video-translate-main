"""Owned local speech.cpp server process for Haas-ja-v1.0 ASR.

Manages ``speech.exe serve`` as a child process on a private loopback port,
mirroring the LlamaServer pattern used for llama.cpp.
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .runtime_diagnostics import rotate_log
from .windows_subprocess import check_running, hidden_console_flags, owned_popen, stopping

log = logging.getLogger(__name__)


class SpeechServerError(RuntimeError):
    """The owned speech-server could not start or stopped unexpectedly."""


def resolve_speech_executable(path: str | None = None) -> Path:
    """Find the speech.cpp binary, falling back to PATH."""
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which("speech") or shutil.which("speech.exe")
    if found:
        return Path(found)
    raise SpeechServerError(
        "speech.exe not found. Install speech.cpp: "
        "pip install speech.cpp or download from https://github.com/k2-fsa/speech.cpp"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


class SpeechCppServer:
    """One speech.cpp serve child on a private loopback port."""

    def __init__(
        self,
        model_name: str,
        executable: Path,
        *,
        device: str = "vulkan",
        threads: int = 8,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.model_name = model_name
        self.executable = executable
        self.device = device
        self.threads = threads
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen | None = None
        self.port = 0
        self._log_stream = None
        self._lock = threading.Lock()
        self._closed = False
        self._shutdown = threading.Event()
        self._cancel_thread = None
        self.client = httpx.Client(trust_env=False)
        self.log_path = Path("logs") / "haas-server.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _watch_cancellation(self) -> None:
        while not self._shutdown.wait(0.1):
            if stopping.is_set():
                self.close()
                return

    def start(self) -> None:
        check_running()
        with self._lock:
            if self._closed:
                raise SpeechServerError("Haas speech-server is closed")
            self._stop_process()
            self.port = _free_port()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            rotate_log(self.log_path)
            self._log_stream = self.log_path.open("ab")
            command = [
                str(self.executable), "serve", self.model_name,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--device", self.device,
                "-t", str(self.threads),
            ]
            log.info("Starting Haas speech-server: %s", " ".join(command[1:]))
            self.process = owned_popen(
                command,
                cwd=str(self.executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                creationflags=hidden_console_flags(),
            )
            process = self.process
            if self._cancel_thread is None:
                self._cancel_thread = threading.Thread(
                    target=self._watch_cancellation,
                    name="haas-cancel",
                    daemon=True,
                )
                self._cancel_thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            check_running()
            if process.poll() is not None:
                raise SpeechServerError(
                    f"Haas speech-server exited during startup "
                    f"(code {process.returncode}). {self.log_tail()}"
                )
            try:
                if self.client.get(self.base_url + "/health", timeout=0.5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.close()
        raise SpeechServerError(
            f"Haas speech-server did not start within {self.startup_timeout_s:g}s. "
            + self.log_tail()
        )

    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def transcribe(self, wav_bytes: bytes, timeout: float) -> str:
        """Send WAV audio to the speech.cpp transcription endpoint."""
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        response = self.client.post(
            self.base_url + "/v1/audio/transcriptions",
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return (payload.get("text") or "").strip()
        raise SpeechServerError(f"Haas returned invalid response: {payload}")

    def log_tail(self, size: int = 4096) -> str:
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, self.log_path.stat().st_size - size))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return f"See {self.log_path}."

    def _stop_process(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown.set()
            self._stop_process()
        self.client.close()
