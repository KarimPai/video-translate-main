"""CPU-only watchdog tests for stalled, silent, and stale-generation workers."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from auto_translate.asr.whisper_turbo import WhisperASR
from auto_translate.pipeline import (
    WorkerState,
    _active_inference_watchdog_timeout,
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

        def recover_stalled(self, deadline=None):
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


def test_active_watchdog_timeout_does_not_include_model_startup_timeout():
    assert _active_inference_watchdog_timeout(30.0) == 35.0


def test_watchdog_reports_recovery_failure_before_blocking_cleanup():
    cleanup_started = threading.Event()
    finish_cleanup = threading.Event()

    class Backend:
        def recover_stalled(self, deadline=None):
            return False

        def close(self):
            cleanup_started.set()
            finish_cleanup.wait(2.0)

    failures = queue.Queue()
    status = {"key": (4, 1), "utterance_id": 17, "done": False, "error": None}
    thread = _start_watchdog_recovery("asr", Backend(), failures, status)
    try:
        assert cleanup_started.wait(1.0)
        name, error = failures.get(timeout=0.2)
        assert name == "watchdog"
        assert "recovery failed" in str(error)
    finally:
        finish_cleanup.set()
        thread.join(2.0)
    assert not thread.is_alive()


def test_whisper_watchdog_invalidates_output_without_releasing_active_model():
    backend = object.__new__(WhisperASR)
    backend._state_lock = threading.Lock()
    backend._inference_lock = threading.Lock()
    backend._generation = 3
    backend._closed = False
    backend._recent = []
    backend._accepted_keys = []
    backend.last_raw_transcript = "late transcript"
    backend.last_language_spans = [(0.0, 1.0, "ja")]
    backend.last_detected_language = "ja"
    backend.last_detected_language_probability = 0.9
    backend._model = object()
    active_model = backend._model
    backend._inference_lock.acquire()
    try:
        assert backend.recover_stalled(deadline=time.monotonic() + 1.0) is False
        assert backend._generation == 4
        assert backend._model is active_model
        assert backend.last_raw_transcript == ""
    finally:
        backend._inference_lock.release()


def test_whisper_discards_native_result_returning_after_absolute_deadline():
    backend = object.__new__(WhisperASR)
    backend._state_lock = threading.Lock()
    backend._inference_lock = threading.Lock()
    backend._generation = 0
    backend._closed = False
    backend._recent = []
    backend._accepted_keys = []
    backend.last_raw_transcript = ""
    backend.last_rejection = None
    backend.last_detected_language = "und"
    backend.last_detected_language_probability = 0.0
    backend.last_language_spans = []
    backend.last_match_score = 0.0
    backend._failures = 0
    backend.cfg = SimpleNamespace(response_timeout_s=0.02, live_response_timeout_s=0.02)

    def slow_inference(*_args, **_kwargs):
        time.sleep(0.05)
        return "こんにちは"

    backend._transcribe_impl = slow_inference
    result = backend.transcribe(np.zeros(1600, dtype=np.float32))

    assert result == ""
    assert backend.last_rejection == "error"
    assert backend.last_raw_transcript == ""