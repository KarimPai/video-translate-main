"""Whisper resource-release synchronization tests; no model is initialized."""
from __future__ import annotations

import threading

from auto_translate.asr.whisper_turbo import WhisperASR


def test_close_waits_for_active_inference_before_releasing_model():
    backend = object.__new__(WhisperASR)
    backend._closed = False
    backend._generation = 0
    backend._state_lock = threading.Lock()
    backend._inference_lock = threading.Lock()
    backend._model = object()
    model = backend._model
    backend._inference_lock.acquire()
    started, finished = threading.Event(), threading.Event()

    def close_backend():
        started.set()
        backend.close()
        finished.set()

    thread = threading.Thread(target=close_backend)
    thread.start()
    try:
        assert started.wait(1.0)
        assert not finished.wait(0.05)
        assert backend._model is model
    finally:
        backend._inference_lock.release()
    assert finished.wait(2.0)
    thread.join(2.0)
    assert not thread.is_alive()
    assert backend._model is None


def test_close_defers_release_when_inference_exceeds_bounded_wait():
    backend = object.__new__(WhisperASR)
    backend._closed = False
    backend._generation = 0
    backend._state_lock = threading.Lock()
    backend._inference_lock = threading.Lock()
    backend._model = object()
    backend._release_thread = None
    backend.CLOSE_WAIT_S = 0.01
    model = backend._model
    backend._inference_lock.acquire()

    backend.close()

    assert backend._model is model
    backend._inference_lock.release()
    backend._release_thread.join(2.0)
    assert not backend._release_thread.is_alive()
    assert backend._model is None