"""Cooperative inference admission; MPV's decoder never waits on this gate.

This bounds simultaneous model requests, not physical VRAM allocations. Models
remain resident and no CUDA synchronization or decoder reconfiguration is used.
"""

import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager

from .runtime_diagnostics import record_event
from .windows_subprocess import stopping


class SafeInferenceGate:
    def __init__(self, cfg, stop=None):
        self.enabled = cfg.enabled
        self.stop = stop if stop is not None else threading.Event()
        self._slots = threading.BoundedSemaphore(cfg.max_inference_concurrency)

    @contextmanager
    def operation(self, name, cancel_event=None, current=None, generation=None, deadline=None):
        def check():
            if (self.stop.is_set() or stopping.is_set()
                    or (cancel_event is not None and cancel_event.is_set())
                    or (current is not None and generation is not None and current() != generation)):
                raise CancelledError("inference cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise CancelledError("inference playback deadline expired")

        check()
        acquired = False
        started = time.monotonic()
        try:
            if self.enabled:
                while not acquired:
                    check()
                    acquired = self._slots.acquire(timeout=.05)
                check()
                if time.monotonic() - started >= 1:
                    record_event("inference_backpressure", stage=name,
                                 wait_s=round(time.monotonic() - started, 3))
            yield
        finally:
            if acquired:
                self._slots.release()
