"""Regression tests for cautious adjacent-speaker reference context."""
import queue
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.config import SpeakerConfig
from auto_translate.pipeline import RecentSpeakerJapanese, run_translate
from auto_translate.speaker import AdjacentSpeakerContinuity
from auto_translate.types import AudioProgress, Transcript, Utterance


class MockEmbeddingStream:
    def accept_waveform(self, *, sample_rate, waveform):
        self.waveform = waveform

    def input_finished(self):
        pass


class MockExtractor:
    def __init__(self, vectors):
        self.vectors = iter(vectors)

    def create_stream(self):
        return MockEmbeddingStream()

    def is_ready(self, stream):
        return True

    def compute(self, stream):
        return next(self.vectors)


def utterance(start, end, *, handoff=False):
    return Utterance(
        float(start), float(end),
        np.ones(round((end - start) * 16000), dtype=np.float32),
        speaker_handoff=handoff,
    )


class SpeakerEmbeddingContinuityTests(unittest.TestCase):
    @staticmethod
    def tracker(*vectors):
        tracker = object.__new__(AdjacentSpeakerContinuity)
        tracker.cfg = SpeakerConfig()
        tracker.extractor = MockExtractor(vectors)
        tracker._counter = 0
        tracker.reset()
        return tracker

    def test_consecutive_same_voice_reuses_opaque_key(self):
        tr = self.tracker([1.0, 0], [.99, .01])
        a = tr.tag(utterance(0, 1.9))
        b = tr.tag(utterance(2, 3.8))
        self.assertIsNotNone(a.speaker_context_key)
        self.assertEqual(a.speaker_context_key, b.speaker_context_key)

    def test_different_voice_never_shares_key(self):
        tr = self.tracker([1.0, 0], [0, 1.0])
        a = tr.tag(utterance(0, 1.9))
        b = tr.tag(utterance(2, 3.8))
        self.assertNotEqual(a.speaker_context_key, b.speaker_context_key)

    def test_handoff_and_long_gap_break_identity(self):
        tr = self.tracker([1, 0], [1, 0], [1, 0])
        a = tr.tag(utterance(0, 1.9))
        b = tr.tag(utterance(2, 3.9, handoff=True))
        c = tr.tag(utterance(6, 7.9))
        self.assertNotEqual(a.speaker_context_key, b.speaker_context_key)
        self.assertNotEqual(b.speaker_context_key, c.speaker_context_key)

    def test_short_uncertain_audio_breaks_continuity(self):
        tr = self.tracker([1, 0], [1, 0])
        a = tr.tag(utterance(0, 1.9))
        short = tr.tag(utterance(2.0, 2.6))
        c = tr.tag(utterance(2.7, 4.3))
        self.assertIsNone(short.speaker_context_key)
        self.assertNotEqual(a.speaker_context_key, c.speaker_context_key)

    def test_reset_on_seek_does_not_reuse_key(self):
        tr = self.tracker([1, 0], [1, 0])
        a = tr.tag(utterance(0, 1.9))
        tr.reset()
        b = tr.tag(utterance(2, 3.9))
        self.assertNotEqual(a.speaker_context_key, b.speaker_context_key)


class RecentContextWindowTests(unittest.TestCase):
    @staticmethod
    def tr(start, end, text, key=None, handoff=False, generation=0):
        return Transcript(
            float(start), float(end), text, generation=generation,
            speaker_context_key=key, speaker_handoff=handoff,
        )

    def test_verified_adjacent_speaker_can_reference_two_japanese_lines(self):
        window = RecentSpeakerJapanese()
        self.assertIsNone(window.context_for(self.tr(0, 2, "最初", 1)))
        window.observe(self.tr(0, 2, "最初", 1))
        self.assertEqual(window.context_for(self.tr(2.1, 4, "次", 1)), "最初")
        window.observe(self.tr(2.1, 4, "次", 1))
        self.assertEqual(window.context_for(self.tr(4.1, 5.9, "続き", 1)), "最初。次")

    def test_never_crosses_handoff_missing_id_or_silence_gap(self):
        window = RecentSpeakerJapanese()
        window.observe(self.tr(0, 2, "first", 1))
        self.assertIsNone(window.context_for(self.tr(2.1, 3, "second", 2)))
        self.assertIsNone(window.context_for(self.tr(2.1, 3, "second", 1, handoff=True)))
        self.assertIsNone(window.context_for(self.tr(5, 6, "second", 1)))
        self.assertIsNone(window.context_for(self.tr(2.1, 3, "second", None)))

    def test_translation_pipeline_separates_speakers_and_seek_generations(self):
        class SpyTranslator:
            memory_lines = 0

            def __init__(self):
                self.reference = None
                self.calls = []

            def set_reference_context(self, text):
                self.reference = text

            def translate(self, ja, history):
                self.calls.append((ja, self.reference))
                return "English " + str(len(self.calls))

        spy = SpyTranslator()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(self.tr(0, 2, "これは最初", 7))
        incoming.put(self.tr(2.1, 4, "どうしたの", 7))
        incoming.put(self.tr(4.1, 6, "別人です", 8, handoff=True))
        incoming.put(self.tr(6.1, 8, "次です", 8))
        incoming.put(AudioProgress(0, reset=True, generation=1))
        incoming.put(self.tr(0, 2, "シーク後", 8, generation=1))
        incoming.put(None)
        run_translate(spy, 0, incoming, outgoing)
        self.assertEqual(spy.calls, [
            ("これは最初", None),
            ("どうしたの", "これは最初"),
            ("別人です", None),
            ("次です", "別人です"),
            ("シーク後", None),
        ])

    def test_context_is_cleared_on_failed_translation(self):
        class FailTranslator:
            memory_lines = 0

            def __init__(self):
                self.reference = None
                self.references = []

            def set_reference_context(self, text):
                self.reference = text

            def translate(self, ja, history):
                self.references.append(self.reference)
                if ja == "invalid":
                    return "[SKIP]"
                return "ok"

        spy = FailTranslator()
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(self.tr(0, 2, "first", 1))
        incoming.put(self.tr(2, 4, "invalid", 1))
        incoming.put(self.tr(4, 6, "third", 1))
        incoming.put(None)
        run_translate(spy, 0, incoming, outgoing)
        self.assertEqual(spy.references, [None, "first", None])


if __name__ == "__main__":
    unittest.main()
