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
from .windows_subprocess import (
    check_running,
    hidden_console_flags,
    owned_popen,
    stopping,
)

log = logging.getLogger(__name__)


class SpeechServerError(RuntimeError):
    """The owned speech-server could not start or stopped unexpectedly."""


def resolve_speech_executable(path: str | None = None) -> Path:
    """Find the speech.cpp binary, falling back to PATH only when no explicit path is given.
    
    If an explicit path is provided but does not exist, raise a clear error
    instead of silently falling back to PATH — the user configured a specific
    location and should know if it is wrong.
    """
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise SpeechServerError(
            f"Configured speech.cpp path does not exist: {candidate}"
        )
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
        device: str = "gpu",
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
            try:
                self.port = _free_port()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_log(self.log_path)
                self._log_stream = self.log_path.open("ab")
                command = [
                    str(self.executable), "serve", self.model_name,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--device", self.device,
                    "--threads", str(self.threads),
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
            except BaseException:
                # Already holding _lock: clean up inline, not via _abort_startup.
                try:
                    self._stop_process()
                except Exception:
                    log.exception("Haas failed to clean up after an interrupted spawn")
                raise
        deadline = time.monotonic() + self.startup_timeout_s
        try:
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
            # A health timeout is recoverable: stop this child but keep the
            # server restartable so HaasASR._restart() can try again.
            raise SpeechServerError(
                f"Haas speech-server did not start within {self.startup_timeout_s:g}s. "
                + self.log_tail()
            )
        except BaseException:
            # Cancellation, KeyboardInterrupt or any unexpected error while
            # waiting for /health must never leave a loading child on the GPU.
            self._abort_startup(process)
            raise

    def _abort_startup(self, process: subprocess.Popen) -> None:
        with self._lock:
            if self.process is process:
                try:
                    self._stop_process()
                except Exception:
                    log.exception("Haas speech-server cleanup after failed startup raised")

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

    def transcribe_with_timing(self, wav_bytes: bytes, timeout: float) -> tuple[str, dict]:
        """Send WAV audio to the speech.cpp transcription endpoint with client-side timing.
        
        Returns (text, timing_dict) where timing_dict contains:
        - audio_prep_s: time to prepare the request payload (already done by caller)
        - http_request_s: total HTTP round-trip time (request sent to response received)
        - total_s: total client-side time including prep and HTTP
        """
        t0 = time.monotonic()
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        t1 = time.monotonic()
        response = self.client.post(
            self.base_url + "/v1/audio/transcriptions",
            files=files,
            timeout=timeout,
        )
        t2 = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            text = (payload.get("text") or "").strip()
        else:
            raise SpeechServerError(f"Haas returned invalid response: {payload}")
        timing = {
            "audio_prep_s": round(t1 - t0, 4),
            "http_request_s": round(t2 - t1, 4),
            "total_s": round(t2 - t0, 4),
        }
        return (text, timing)

    def log_tail(self, size: int = 4096) -> str:
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, self.log_path.stat().st_size - size))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return f"See {self.log_path}."

    def _stop_process(self) -> None:
        # Never lose ownership of a process until its exit is observed.
        # If termination fails, a later close()/start() can retry safely.
        process = self.process
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=3)
                    except (subprocess.TimeoutExpired, OSError):
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=3)
                if process.poll() is None:
                    raise SpeechServerError("Haas speech-server did not exit after termination")
                self.process = None
        finally:
            # A failed terminate/kill must not leave our log stream open.
            if self._log_stream is not None:
                try:
                    self._log_stream.close()
                finally:
                    self._log_stream = None

    def close(self) -> None:
        with self._lock:
            # Retained process references must remain terminable even after
            # a previous close() attempt failed.
            if self._closed and self.process is None and self._log_stream is None:
                return
            self._closed = True
            self._shutdown.set()
            try:
                self._stop_process()
            finally:
                self.client.close()
