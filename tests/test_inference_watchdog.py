"""CPU-only watchdog tests for stalled, silent, and stale-generation workers."""
from __future__ import annotations

import queue
import time

from auto_translate.pipeline import (
    WorkerState,
    _current_inference_stall,
    _start_watchdog_recovery,
)


def test_watchdog_only_targets_active_current_generation_inference():
    state = WorkerState()
    state.enter_inference(17, 4)
    state.inference_start = time.monotonic() - 10.0

    assert _current_inference_stall(state, 1.0, 4)
    assert not _current_inference_stall(state, 1.0, 5)

    state.leave_inference()
    assert not _current_inference_stall(state, 1.0, 4)

    state.enter_inference(18, 4, probe=True)
    state.inference_start = time.monotonic() - 10.0
    assert not _current_inference_stall(state, 1.0, 4)


def test_watchdog_executes_backend_recovery_without_mpv_pause_state():
    class Backend:
        recovered = False

        def recover_stalled(self):
            self.recovered = True
            return True

    backend = Backend()
    failures = queue.Queue()
    status = {"key": (4, 1), "utterance_id": 17, "done": False, "error": None}
    thread = _start_watchdog_recovery("asr", backend, failures, status)
    thread.join(2.0)

    assert backend.recovered
    assert status["done"]
    assert status["error"] is None
    assert failures.empty()