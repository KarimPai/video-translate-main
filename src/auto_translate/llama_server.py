"""Owned local llama.cpp server processes shared by the ASR and translation models."""
from __future__ import annotations

import logging
import shutil
import socket
import struct
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


class LlamaServerError(RuntimeError):
    """The owned llama-server could not start or stopped unexpectedly."""


def resolve_executable(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(path) or shutil.which("llama-server")
    if found:
        return Path(found)
    raise LlamaServerError(f"llama-server not found: {path}. Install llama.cpp via scripts/Setup-Windows.ps1.")


def detect_device(executable: Path | None = None, cuda_device: int | None = None) -> str:
    """Detect the best available GPU device for llama-server.

    Honors explicit cuda_device configuration:
    - cuda_device=-1: return "CPU" for CPU-only inference
    - cuda_device>=0: return the specific CUDA device (e.g. "CUDA2")
    - cuda_device=None: auto-detect best available device

    Prefers CUDA for NVIDIA GPUs (better performance), falls back to Vulkan.
    Returns a device string like "CUDA0" or "Vulkan0" suitable for --device flag.
    If executable is provided, uses it directly; otherwise resolves from config/PATH.
    """
    import re

    # Honor explicit CPU request
    if cuda_device == -1:
        return "CPU"

    # Honor explicit CUDA device index
    if cuda_device is not None and cuda_device >= 0:
        requested = f"CUDA{cuda_device}"
        # Verify the device exists if we can query
        exe = executable
        if exe is None:
            try:
                exe = resolve_executable("")
            except LlamaServerError:
                pass
        verified = False
        if exe is not None:
            try:
                result = subprocess.run(
                    [str(exe), "--list-devices"],
                    capture_output=True, text=True, timeout=10,
                    check=False,
                    creationflags=hidden_console_flags(),
                )
                output = result.stdout + result.stderr
                devices = re.findall(r"^\s+(\w+\d+):", output, re.MULTILINE)
                if requested in devices:
                    return requested
                # Requested device not found — warn and fall through to auto-detect
                log.warning(
                    "Requested CUDA device %s not found in available devices %s; "
                    "falling back to auto-detection",
                    requested, devices,
                )
                verified = True
            except (subprocess.TimeoutExpired, OSError):
                pass
        # If we could verify and the device was not found, fall through to auto-detect.
        # If we could not verify (no executable or timeout), return the requested device
        # anyway — llama-server will fail with a clear error if it doesn't exist.
        if not verified:
            return requested
        # Fall through to auto-detect below

    # Auto-detect: find the best available device
    exe = executable
    if exe is None:
        try:
            exe = resolve_executable("")
        except LlamaServerError:
            pass

    if exe is None:
        # No executable found, default to CUDA0 (most common on Windows NVIDIA)
        return "CUDA0"

    try:
        result = subprocess.run(
            [str(exe), "--list-devices"],
            capture_output=True, text=True, timeout=10,
            check=False,
            creationflags=hidden_console_flags(),
        )
        output = result.stdout + result.stderr
        # Parse output like:
        #   "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 3070 ..."
        # or "Available device: CUDA0 NVIDIA GeForce RTX 3070 ..."
        # Match indented device names: "  Vulkan0:" or "  CUDA0:"
        devices = re.findall(r"^\s+(\w+\d+):", output, re.MULTILINE)
        if devices:
            # Prefer CUDA over Vulkan for NVIDIA GPUs
            for dev in devices:
                if dev.startswith("CUDA"):
                    return dev
            # Fall back to first available GPU device
            return devices[0]
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: assume CUDA0 (works with CUDA builds)
    return "CUDA0"


def require_gguf(path: str, label: str) -> Path:
    model = Path(path).expanduser().resolve()
    if not model.is_file():
        from .models import cached_model, model_at_default_path

        spec = model_at_default_path(path)
        cached = cached_model(spec) if spec is not None else None
        if cached is None:
            raise LlamaServerError(f"{label} not found: {model}. Configure an existing model path or install via scripts/Setup-Windows.ps1.")
        model = cached
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise LlamaServerError(f"{label} is not a GGUF file: {model}. Reinstall via scripts/Setup-Windows.ps1.")
    return model


_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_block_count(path: Path) -> int | None:
    """Transformer layer count from GGUF metadata, or None if unreadable."""
    try:
        with path.open("rb") as f:
            magic, _version, _tensors, kv_count = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF":
                return None

            def string() -> str:
                (length,) = struct.unpack("<Q", f.read(8))
                return f.read(length).decode("utf-8", errors="replace")

            def skip(kind: int) -> None:
                if kind == 8:
                    (length,) = struct.unpack("<Q", f.read(8))
                    f.seek(length, 1)
                elif kind == 9:
                    inner, count = struct.unpack("<IQ", f.read(12))
                    if inner in _GGUF_SCALARS:
                        f.seek(struct.calcsize(_GGUF_SCALARS[inner]) * count, 1)
                    else:
                        for _ in range(count):
                            skip(inner)
                else:
                    f.seek(struct.calcsize(_GGUF_SCALARS[kind]), 1)

            for _ in range(kv_count):
                key = string()
                (kind,) = struct.unpack("<I", f.read(4))
                if key.endswith(".block_count") and kind in _GGUF_SCALARS:
                    (value,) = struct.unpack(_GGUF_SCALARS[kind], f.read(struct.calcsize(_GGUF_SCALARS[kind])))
                    return int(value)
                skip(kind)
    except (OSError, struct.error, KeyError):
        return None
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


class LlamaServer:
    """One llama-server child on a private loopback port."""

    def __init__(self, name: str, executable: Path, args: list[str], *, startup_timeout_s: float) -> None:
        self.name = name
        self.executable = executable
        self.args = list(args)
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen | None = None
        self.port = 0
        self._log_stream = None
        self._lock = threading.Lock()
        self._closed = False
        self._shutdown = threading.Event()
        self._cancel_thread = None
        self._watcher_generation = 0
        self.client = httpx.Client(trust_env=False)
        self.log_path = Path("logs") / f"{name}-server.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _watch_cancellation(self, generation: int) -> None:
        while not self._shutdown.wait(0.1):
            if self._watcher_generation != generation:
                # Superseded by a newer watcher; exit cleanly.
                return
            if stopping.is_set():
                self.close()
                return

    def start(self) -> None:
        check_running()
        with self._lock:
            if self._closed:
                raise LlamaServerError(f"{self.name} server is closed")
            self._stop_process()
            self.port = _free_port()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            rotate_log(self.log_path)
            self._log_stream = self.log_path.open("ab")
            command = [str(self.executable), *self.args, "--host", "127.0.0.1", "--port", str(self.port)]
            log.info("Starting %s llama-server: %s", self.name, " ".join(command[1:]))
            self.process = owned_popen(command, cwd=str(self.executable.parent), stdin=subprocess.DEVNULL,
                                       stdout=self._log_stream, stderr=subprocess.STDOUT,
                                       creationflags=hidden_console_flags())
            process = self.process
            # Join any old watcher before creating a new one
            if self._cancel_thread is not None and self._cancel_thread.is_alive():
                self._shutdown.set()
                self._cancel_thread.join(timeout=2)
                self._cancel_thread = None
                self._shutdown.clear()
            self._watcher_generation += 1
            self._cancel_thread = threading.Thread(
                target=self._watch_cancellation,
                args=(self._watcher_generation,),
                name=f"{self.name}-cancel", daemon=True,
            )
            self._cancel_thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            check_running()
            if process.poll() is not None:
                raise LlamaServerError(f"{self.name} llama-server exited during startup "
                                       f"(code {process.returncode}). {self.log_tail()}")
            try:
                if self.client.get(self.base_url + "/health", timeout=0.5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.close()
        raise LlamaServerError(f"{self.name} llama-server did not load within {self.startup_timeout_s:g}s. "
                               + self.log_tail())

    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def chat(self, body: dict, timeout: float) -> dict:
        response = self.client.post(self.base_url + "/v1/chat/completions", json=body, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaServerError(f"{self.name} returned invalid choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LlamaServerError(f"{self.name} returned invalid message")
        content = message.get("content")
        if content is None:
            raise LlamaServerError(f"{self.name} returned empty message (no content)")
        return payload

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

    def reset_for_restart(self) -> None:
        """Clear the closed flag so start() can be called again.

        Stops the old process, clears the cancellation watcher (start()
        recreates a fresh one), and recreates the HTTP client to release
        sockets bound to the old process/port. Does not revive the server
        while the application is shutting down.
        """
        with self._lock:
            if stopping.is_set():
                self._closed = True
                self._shutdown.set()
                self._stop_process()
                self.client.close()
                return
            self._stop_process()
            # Wake the previous cancellation watcher and wait for it to exit
            self._shutdown.set()
            if self._cancel_thread is not None and self._cancel_thread.is_alive():
                self._cancel_thread.join(timeout=2)
            self._cancel_thread = None
            self._closed = False
            self._shutdown.clear()
            # Increment generation so the new watcher knows it is the current one
            self._watcher_generation += 1
            # Close and recreate the HTTP client to release sockets.
            self.client.close()
            self.client = httpx.Client(trust_env=False)
        log.info("%s server reset for restart", self.name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown.set()
            self._stop_process()
        self.client.close()
