"""Owned local llama.cpp server processes shared by the ASR and translation models."""
from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from contextlib import contextmanager
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
    - cuda_device=-1: return "none" for CPU-only inference
    - cuda_device>=0: return the specific CUDA device (e.g. "CUDA2")
    - cuda_device=None: auto-detect best available device

    Prefers CUDA for NVIDIA GPUs (better performance), falls back to Vulkan.
    Returns a device string like "CUDA0" or "Vulkan0" suitable for --device flag.
    If executable is provided, uses it directly; otherwise resolves from config/PATH.
    """
    import re

    # llama.cpp uses --device none to disable GPU offload.
    if cuda_device == -1:
        return "none"
    if cuda_device is not None and cuda_device < -1:
        raise ValueError("cuda_device must be -1 (CPU) or a nonnegative CUDA index")

    # Explicit CUDA indices must never silently select another GPU.
    if cuda_device is not None:
        requested = f"CUDA{cuda_device}"
        exe = executable
        if exe is None:
            try:
                exe = resolve_executable("")
            except LlamaServerError:
                pass
        if exe is not None:
            try:
                result = subprocess.run(
                    [str(exe), "--list-devices"],
                    capture_output=True, text=True, timeout=10,
                    check=False,
                    creationflags=hidden_console_flags(),
                )
                if result.returncode == 0:
                    devices = re.findall(r"^\s+(\w+\d+):", result.stdout + result.stderr, re.MULTILINE)
                    if devices and requested not in devices:
                        raise LlamaServerError(
                            f"Requested CUDA device {requested} is unavailable; "
                            f"llama-server reports: {', '.join(devices)}. "
                            "Change cuda_device or install a CUDA-enabled llama-server."
                        )
                else:
                    log.warning(
                        "Cannot verify %s: llama-server --list-devices exited with code %s; "
                        "the server will validate the requested GPU on startup",
                        requested, result.returncode,
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                log.warning("Could not enumerate CUDA devices (%s); requesting %s explicitly", exc, requested)
        return requested

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


class DeadlineHttpClient:
    """Synchronous interface for cancellable, wall-clock-bounded HTTP exchanges."""

    def __init__(self) -> None:
        self._closed = threading.Event()

    def get(self, url: str, *, timeout: float, check: Callable[[], None] | None = None) -> httpx.Response:
        return self._request("GET", url, timeout, check)

    def post(self, url: str, *, timeout: float, json: dict | None = None,
             files: dict | None = None, check: Callable[[], None] | None = None) -> httpx.Response:
        return self._request("POST", url, timeout, check, json=json, files=files)

    def _request(self, method: str, url: str, timeout: float,
                 check: Callable[[], None] | None, **kwargs) -> httpx.Response:
        deadline = time.monotonic() + timeout

        def check_request() -> None:
            if self._closed.is_set():
                raise CancelledError("model server HTTP client is closed")
            if check is not None:
                check()

        async def exchange() -> httpx.Response:
            check_request()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            # HTTPX timeouts apply per phase. Bound the entire exchange and
            # cancel its socket when the calling utterance becomes stale.
            async with asyncio.timeout(remaining):
                async with httpx.AsyncClient(trust_env=False, timeout=remaining) as client:
                    async def send() -> httpx.Response:
                        check_request()
                        return await client.request(method, url, **kwargs)

                    task = asyncio.create_task(send())
                    try:
                        while True:
                            done, _ = await asyncio.wait((task,), timeout=0.05)
                            check_request()
                            if done:
                                return task.result()
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        try:
            return asyncio.run(exchange())
        except TimeoutError as exc:
            raise httpx.TimeoutException("model server HTTP wall-clock deadline exceeded") from exc

    def close(self) -> None:
        # Each exchange observes this signal and closes its own connection.
        self._closed.set()


class LlamaServer:
    """One llama-server child on a private loopback port."""

    def __init__(self, name: str, executable: Path, args: list[str], *,
                 startup_timeout_s: float, strict_http_deadlines: bool = False) -> None:
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
        self._strict_http_deadlines = strict_http_deadlines
        self.client = self._new_client()
        self.log_path = Path("logs") / f"{name}-server.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _new_client(self) -> httpx.Client | DeadlineHttpClient:
        return DeadlineHttpClient() if self._strict_http_deadlines else httpx.Client(trust_env=False)

    @contextmanager
    def _lifecycle_lock(self, remaining: Callable[[], float] | None):
        if remaining is None:
            self._lock.acquire()
        else:
            while not self._lock.acquire(timeout=min(0.05, remaining())):
                pass
        try:
            yield
        finally:
            self._lock.release()

    def _watch_cancellation(self) -> None:
        while not self._shutdown.wait(0.1):
            if stopping.is_set():
                self.close()
                return

    def start(self, *, deadline: float | None = None,
              check: Callable[[], None] | None = None) -> None:
        if self._closed:
            raise LlamaServerError(f"{self.name} server is closed")
        check_running()
        startup_deadline = time.monotonic() + self.startup_timeout_s
        if deadline is not None:
            startup_deadline = min(startup_deadline, deadline)

        def remaining() -> float:
            check_running()
            if self._shutdown.is_set():
                raise CancelledError(f"{self.name} server is closing")
            if check is not None:
                check()
            seconds = startup_deadline - time.monotonic()
            if seconds <= 0:
                raise LlamaServerError(f"{self.name} llama-server did not load before its startup deadline. "
                                       + self.log_tail())
            return seconds

        with self._lifecycle_lock(remaining):
            if self._closed or self._shutdown.is_set():
                raise LlamaServerError(f"{self.name} server is closed")
            if deadline is not None:
                remaining()
            elif check is not None:
                check()
            self._stop_process(deadline=startup_deadline)
            # Process creation, log setup and watcher startup all need cleanup.
            # The health-polling exception handler below cannot catch failures here.
            try:
                self.port = _free_port()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_log(self.log_path)
                self._log_stream = self.log_path.open("ab")
                command = [str(self.executable), *self.args, "--host", "127.0.0.1", "--port", str(self.port)]
                log.info("Starting %s llama-server: %s", self.name, " ".join(command[1:]))
                remaining()
                self.process = owned_popen(command, cwd=str(self.executable.parent), stdin=subprocess.DEVNULL,
                                           stdout=self._log_stream, stderr=subprocess.STDOUT,
                                           creationflags=hidden_console_flags())
                process = self.process
                client, health_url = self.client, self.base_url + "/health"
                if self._cancel_thread is None:
                    watcher = threading.Thread(target=self._watch_cancellation,
                                               name=f"{self.name}-cancel", daemon=True)
                    watcher.start()
                    self._cancel_thread = watcher
            except BaseException:
                # Already holding _lock: do not call _abort_startup here.
                try:
                    self._stop_process(deadline=startup_deadline)
                except Exception:
                    log.exception("%s failed to clean up after an interrupted spawn", self.name)
                raise
        try:
            while True:
                timeout = min(remaining(), 0.2 if check is not None else 0.5)
                if process.poll() is not None:
                    raise LlamaServerError(f"{self.name} llama-server exited during startup "
                                           f"(code {process.returncode}). {self.log_tail()}")
                try:
                    options = {"check": remaining} if isinstance(client, DeadlineHttpClient) else {}
                    ready = client.get(health_url, timeout=timeout, **options).status_code == 200
                except httpx.HTTPError:
                    ready = False
                delay = remaining()
                if ready:
                    if self.process is not process or process.poll() is not None:
                        raise LlamaServerError(f"{self.name} llama-server exited during startup")
                    return
                self._shutdown.wait(min(0.05 if check is not None else 0.1, delay))
        except BaseException:
            # Cancellation (CancelledError), KeyboardInterrupt or any unexpected
            # error while waiting for /health must never leave a loading child
            # holding VRAM. Only reap the child this call started.
            self._abort_startup(process, startup_deadline)
            raise

    def _abort_startup(self, process: subprocess.Popen, deadline: float | None = None) -> None:
        # Close takes responsibility for the child as soon as it signals
        # shutdown. A cancelled startup should not wait for close's reap.
        if self._shutdown.is_set():
            return
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        acquired = self._lock.acquire() if timeout is None else self._lock.acquire(timeout=timeout)
        if not acquired:
            return
        try:
            if self.process is process:
                try:
                    self._stop_process(deadline=deadline)
                except Exception:
                    log.exception("%s llama-server cleanup after failed startup raised", self.name)
        finally:
            self._lock.release()

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

    def _stop_process(self, *, deadline: float | None = None) -> None:
        # Never lose ownership of a process until its exit is observed.
        # If termination fails, a later close()/start() can retry safely.
        process = self.process
        termination_error: OSError | None = None

        def wait_remaining() -> float:
            if deadline is None:
                return 3.0
            return max(0.0, min(3.0, deadline - time.monotonic()))

        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except OSError as exc:
                        termination_error = exc
                        log.warning("Could not terminate %s llama-server", self.name)
                    timeout = wait_remaining()
                    if timeout > 0:
                        try:
                            process.wait(timeout=timeout)
                        except (subprocess.TimeoutExpired, OSError):
                            pass
                    if process.poll() is None:
                        try:
                            process.kill()
                        except OSError as exc:
                            termination_error = exc
                            log.warning("Could not kill %s llama-server", self.name)
                        timeout = wait_remaining()
                        if timeout > 0:
                            try:
                                process.wait(timeout=timeout)
                            except (subprocess.TimeoutExpired, OSError):
                                pass
                if process.poll() is None:
                    if termination_error is not None:
                        raise termination_error
                    raise LlamaServerError(
                        f"{self.name} llama-server did not exit before its teardown deadline"
                        if deadline is not None else
                        f"{self.name} llama-server did not exit after termination"
                    )
                self.process = None
        finally:
            # A failed terminate/kill must not leave our log stream open.
            if self._log_stream is not None:
                try:
                    self._log_stream.close()
                finally:
                    self._log_stream = None

    def reset_for_restart(self, *, deadline: float | None = None,
                          check: Callable[[], None] | None = None) -> None:
        """Recycle the process and client of an open server. Close is permanent."""
        if self._closed:
            raise LlamaServerError(f"{self.name} server is closed")
        if deadline is None:
            deadline = time.monotonic() + self.startup_timeout_s + 6.0

        def remaining() -> float:
            check_running()
            if self._shutdown.is_set():
                raise CancelledError(f"{self.name} server is closing")
            if check is not None:
                check()
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise LlamaServerError(f"{self.name} restart deadline exceeded")
            return seconds

        with self._lifecycle_lock(remaining):
            check_running()
            if self._closed or self._shutdown.is_set():
                raise LlamaServerError(f"{self.name} server is closed")
            remaining()
            self._stop_process(deadline=deadline)
            remaining()
            old_client = self.client
            self.client = self._new_client()
            old_client.close()
        log.info("%s server reset for restart", self.name)

    def close(self) -> None:
        self._shutdown.set()
        with self._lock:
            # Retained process references must remain terminable even after
            # a previous close() attempt failed.
            if self._closed and self.process is None and self._log_stream is None:
                return
            self._closed = True
            try:
                self._stop_process()
            finally:
                self.client.close()
