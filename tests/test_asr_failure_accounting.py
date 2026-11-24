"""CPU-only regressions for ASR request retries and utterance failure limits."""
from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from auto_translate.asr.haas import HaasASR, HaasError
from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
from auto_translate.pipeline import run_asr
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Utterance


class AlwaysFailingASR:
    def __init__(self):
        self.cfg = SimpleNamespace(response_timeout_s=1.0, context_retry_limit=0)
        self.last_rejection = None
        self.request_attempts = 0
        self.failed_utterances = 0

    def transcribe(self, _audio, _probe=False):
        self.request_attempts += 1
        self.last_rejection = "error"
        return ""

    def note_utterance_failure(self):
        self.failed_utterances += 1


def test_three_request_attempts_count_as_one_failed_utterance(monkeypatch):
    from auto_translate import pipeline

    monkeypatch.setattr(pipeline.time, "sleep", lambda _delay: None)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(Utterance(0.0, 1.0, np.zeros(16000, dtype=np.float32)))
    incoming.put(None)
    asr = AlwaysFailingASR()

    run_asr(asr, incoming, outgoing)

    assert asr.request_attempts == 3
    assert asr.failed_utterances == 1


def test_failed_asr_keeps_audio_out_of_reusable_subtitle_coverage(monkeypatch):
    from auto_translate import pipeline

    monkeypatch.setattr(pipeline.time, "sleep", lambda _delay: None)
    cache = SessionSubtitleCache()
    cache.record_processed(0.0, 10.0)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(Utterance(4.0, 5.0, np.zeros(16000, dtype=np.float32)))
    incoming.put(None)

    run_asr(AlwaysFailingASR(), incoming, outgoing, session_cache=cache)

    assert cache.covered_until(2.0) == 4.0


@pytest.mark.parametrize("backend_type", [HaasASR, WhisperASR])
def test_asr_retries_share_one_absolute_utterance_deadline(monkeypatch, backend_type):
    from auto_translate import pipeline

    class DeadlineASR(backend_type):
        def __init__(self):
            self.cfg = SimpleNamespace(response_timeout_s=1.0, context_retry_limit=0)
            self.last_rejection = None
            self.deadlines = []

        def transcribe(self, _audio, _probe=False, *, deadline=None):
            self.deadlines.append(deadline)
            self.last_rejection = "error"
            return ""

        def note_utterance_failure(self):
            pass

        def cancel_pending(self):
            pass

        def forget(self):
            pass

    monkeypatch.setattr(pipeline.time, "sleep", lambda _delay: None)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(Utterance(0.0, 1.0, np.zeros(16000, dtype=np.float32)))
    incoming.put(None)
    asr = DeadlineASR()

    run_asr(asr, incoming, outgoing)

    assert len(asr.deadlines) == 3
    assert asr.deadlines[0] is not None
    assert asr.deadlines.count(asr.deadlines[0]) == 3


@pytest.mark.parametrize(
    ("backend_type", "error_type"),
    [(HaasASR, HaasError), (WhisperASR, WhisperError)],
)
def test_consecutive_failure_limit_counts_utterances(backend_type, error_type):
    backend = object.__new__(backend_type)
    backend._failures = 0

    backend.note_utterance_failure()
    backend.note_utterance_failure()
    with pytest.raises(error_type):
        backend.note_utterance_failure()