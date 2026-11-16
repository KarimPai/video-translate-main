"""Offline alignment behavior using a fake WhisperX; no models are downloaded."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.refinement import RefinementCancelled, refine_subtitle_timings
from auto_translate.types import SAMPLE_RATE, Subtitle


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.media = Path(self.directory.name) / "video.wav"
        self.media.touch()
        self.whisperx = Mock()
        self.whisperx.load_audio.return_value = np.zeros(SAMPLE_RATE * 5)
        self.whisperx.load_align_model.return_value = (Mock(), {"language": "ja"})
        self.whisperx.align.return_value = {"segments": [{"words": [
            {"word": "こん", "start": 1.1, "end": 1.3},
            {"word": "にちは", "start": 1.4, "end": 1.8},
        ]}]}
        self.cue = Subtitle(1, 2, "こんにちは", "Hello", generation=6)
        self.patcher = patch.dict("sys.modules", {"whisperx": self.whisperx})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_aligns_existing_japanese_cpu_default_and_never_mutates_input(self):
        actual = refine_subtitle_timings(self.media, [self.cue])[0]
        self.assertEqual((actual.start_s, actual.end_s, actual.speech_start_s), (1.1, 1.8, 1.1))
        self.assertEqual((actual.ja, actual.en, actual.generation), ("こんにちは", "Hello", 6))
        self.assertEqual((self.cue.start_s, self.cue.end_s), (1, 2))
        self.whisperx.load_align_model.assert_called_once_with(language_code="ja", device="cpu")
        segment = self.whisperx.align.call_args.args[0][0]
        self.assertEqual(segment["text"], self.cue.ja)

    def test_invalid_missing_or_incomplete_alignment_keeps_original(self):
        for words in [[], [{"word": "こんにちは"}],
                      [{"word": "こんにちは", "start": float("nan"), "end": 1.8}],
                      [{"word": "こんにちは", "start": 0, "end": 1.8}],
                      [{"word": "こん", "start": 1.1, "end": 1.8}]]:
            with self.subTest(words=words):
                self.whisperx.align.return_value = {"segments": [{"words": words}]}
                actual = refine_subtitle_timings(self.media, [self.cue])[0]
                self.assertEqual((actual.start_s, actual.end_s), (1, 2))

    def test_cancellation_before_and_after_native_call_discards_result(self):
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(RefinementCancelled):
            refine_subtitle_timings(self.media, [self.cue], cancel=cancelled)
        self.whisperx.load_audio.assert_not_called()
        cancelled.clear()
        def align(*args, **kwargs):
            cancelled.set()
            return {"segments": []}
        self.whisperx.align.side_effect = align
        with self.assertRaises(RefinementCancelled):
            refine_subtitle_timings(self.media, [self.cue], cancel=cancelled)
        self.assertEqual((self.cue.start_s, self.cue.end_s), (1, 2))

    def test_rejects_live_invalid_media_or_wrong_duration_before_alignment(self):
        with self.assertRaisesRegex(ValueError, "VOD"):
            refine_subtitle_timings(self.media, [self.cue], is_live=True)
        with self.assertRaisesRegex(ValueError, "existing local"):
            refine_subtitle_timings(self.media.with_name("missing.wav"), [self.cue])
        with self.assertRaisesRegex(ValueError, "exceed"):
            refine_subtitle_timings(self.media, [Subtitle(4, 6, "こんにちは", "Hello")])
        self.whisperx.load_align_model.assert_not_called()

    def test_missing_dependency_explains_optional_install(self):
        with (patch.dict("sys.modules", {"whisperx": None}),
              self.assertRaisesRegex(RuntimeError, "refinement.*extra")):
            refine_subtitle_timings(self.media, [self.cue])

    def test_native_failure_propagates_without_partial_updates(self):
        self.whisperx.align.side_effect = RuntimeError("alignment failed")
        with self.assertRaisesRegex(RuntimeError, "alignment failed"):
            refine_subtitle_timings(self.media, [self.cue])
        self.assertEqual(self.cue.start_s, 1)


if __name__ == "__main__":
    unittest.main()
