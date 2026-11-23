"""Sample-level regressions for the Silero segmenter without model inference."""

import queue

import numpy as np
import pytest

from auto_translate.config import VadConfig
from auto_translate.types import SAMPLE_RATE, AudioChunk, AudioProgress, SourceEOF, Utterance
from auto_translate.vad import FRAME, Segmenter, run_segmenter


@pytest.fixture
def scripted_segmenter(monkeypatch):
    def make(probabilities, **settings):
        class Vad:
            def __init__(self, *args):
                self.probs = iter(probabilities)
                self.last_input = None

            def reset(self):
                pass

            def __call__(self, pcm):
                assert len(pcm) % FRAME == 0
                self.last_input = pcm.copy()
                return [next(self.probs) for _ in range(len(pcm) // FRAME)]

        monkeypatch.setattr("auto_translate.vad.StreamingSileroVAD", Vad)
        return Segmenter(VadConfig(fallback_s=0.0, **settings))

    return make


def marked_pcm(length):
    # A varying sample sequence with constant RMS in each full frame.
    return np.random.default_rng(42).choice(
        np.array([-.02, .02], dtype=np.float32), size=length,
    )


def assert_exact_audio(cues, pcm, start_sample):
    actual = np.concatenate([cue.audio for cue in cues])
    np.testing.assert_array_equal(actual, pcm)
    for cue in cues:
        assert cue.start_s == start_sample / SAMPLE_RATE
        start_sample += len(cue.audio)
        assert cue.end_s == start_sample / SAMPLE_RATE
        assert cue.start_s <= cue.speech_start_s < cue.end_s


def test_forced_split_preserves_audit_weak_continuation(scripted_segmenter):
    seg = scripted_segmenter([.95] * 12 + [.20] * 550 + [0] * 17)
    pcm = marked_pcm(579 * FRAME)

    cues = seg.feed(AudioChunk(0, pcm, 30.0)) + seg.flush()

    assert len(cues) == 2
    assert cues[0].end_s == 14.016
    assert cues[1].speech_start_s == cues[0].end_s
    assert cues[1].vad_confidence == pytest.approx(.20)
    # Keep all 562 speech frames and the configured nine frames of trailing pad.
    assert_exact_audio(cues, pcm[:571 * FRAME], 0)


@pytest.mark.parametrize("tail", [0, 1, 400, FRAME - 1])
def test_flush_keeps_exact_pcm_after_unaligned_initial_chunk(scripted_segmenter, tail):
    seg = scripted_segmenter([.95] * (10 + bool(tail)))
    pcm = marked_pcm(10 * FRAME + tail)

    assert seg.feed(AudioChunk(0, pcm[:100], 1.0)) == []
    assert seg.feed(AudioChunk(100, pcm[100:], 2.0)) == []
    cues = seg.flush()

    assert len(cues) == 1
    assert_exact_audio(cues, pcm, 0)
    if tail:
        np.testing.assert_array_equal(seg.vad.last_input[:tail], pcm[-tail:])
        assert not np.any(seg.vad.last_input[tail:])
    assert seg.flush() == []


def test_eof_partial_frame_keeps_forced_continuation(scripted_segmenter):
    seg = scripted_segmenter([.95] * 12 + [.20] * 551)
    start = 16000
    pcm = marked_pcm(562 * FRAME + 400)

    cues = seg.feed(AudioChunk(start, pcm, 30.0)) + seg.flush()

    assert len(cues) == 2
    assert_exact_audio(cues, pcm, start)
    assert cues[1].speech_start_s == cues[0].end_s
    assert cues[1].timings.audio_read == 30.0


def test_pause_ends_continuation_admission(scripted_segmenter):
    seg = scripted_segmenter(
        [.95] * 40 + [0] * 20 + [.20] * 20 + [0],
        prob_smooth_frames=1, adaptive_split=False,
        max_utterance_s=1.0, max_split_extension_s=0.0,
    )
    pcm = marked_pcm(80 * FRAME + 400)

    cues = seg.feed(AudioChunk(0, pcm, 30.0)) + seg.flush()

    assert len(cues) > 1  # There is at least one forced cut before the pause.
    assert_exact_audio(cues, pcm[:40 * FRAME], 0)


def test_silent_remainder_after_a_forced_cut_does_not_create_a_cue(scripted_segmenter):
    seg = scripted_segmenter(
        [.95] * 22 + [0] * 10,
        prob_smooth_frames=1, adaptive_split=False, split_context_ms=32,
        max_utterance_s=1.0, max_split_extension_s=0.0,
    )
    speech = marked_pcm(22 * FRAME)
    pcm = np.concatenate([speech, np.zeros(9 * FRAME + 400, dtype=np.float32)])

    cues = seg.feed(AudioChunk(0, pcm, 1.0))

    assert len(cues) == 1  # The hard limit emitted the cue before EOF.
    assert len(cues[0].audio) >= len(speech)
    assert_exact_audio(cues, pcm[:len(cues[0].audio)], 0)
    assert seg.flush() == []
    assert seg.flush() == []


def test_reset_discards_stale_partial_pcm_and_eof_reports_exact_progress(scripted_segmenter):
    seg = scripted_segmenter([.95] * 11)
    pcm = marked_pcm(10 * FRAME + 400)
    start = 12 * SAMPLE_RATE
    original_tail = AudioChunk(0, np.full(400, .5, dtype=np.float32), 1.0)
    inbox, outbox = queue.Queue(), queue.Queue()
    assert seg.feed(original_tail) == []
    for item in (
        AudioChunk(start, pcm[:0], 2.0, reset=True, generation=1),
        original_tail,
        SourceEOF(0),
        AudioChunk(start, pcm, 3.0, generation=1),
        SourceEOF(1),
        None,
    ):
        inbox.put(item)

    run_segmenter(seg, inbox, outbox, get_generation=lambda: 1)

    reset, cue, progress, eof, done = [outbox.get_nowait() for _ in range(5)]
    assert reset == AudioProgress(start, reset=True, generation=1)
    assert isinstance(cue, Utterance)
    assert cue.generation == 1
    assert_exact_audio([cue], pcm, start)
    assert progress == AudioProgress(start + len(pcm), generation=1)
    assert eof == SourceEOF(1)
    assert done is None and outbox.empty()
