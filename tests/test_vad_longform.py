"""Offline long-form VAD split regressions; no FireRed model download."""
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig
from auto_translate.types import SAMPLE_RATE, AudioChunk
from auto_translate.vad import FRAME, Segmenter, _select_longform_cut


class LongformBoundaryTests(unittest.TestCase):
    @staticmethod
    def _process(pcm, probs, cfg=None):
        class FakeVAD:
            def __call__(self, audio):
                assert len(probs) == len(audio) // FRAME
                return probs
            def reset(self):
                pass
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=FakeVAD()):
            segmenter = Segmenter(cfg or VadConfig())
            emitted = segmenter.feed(AudioChunk(0, pcm, 0.0))
            emitted += segmenter.flush()
        return [x for x in emitted if not x.probe]

    def test_real_short_pause_selected_near_six_second_soft_limit(self):
        frames = 600
        pcm = np.full(frames * FRAME, .3, dtype=np.float32)
        probs = np.full(frames, .9, dtype=np.float32)
        pcm[545 * FRAME:558 * FRAME] = 0.0
        probs[545:558] = 0.05
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0].end_s, 5.45)
        self.assertLess(result[0].end_s, 5.65)
        self.assertAlmostEqual(result[0].end_s, result[1].start_s, places=6)
        self.assertEqual(sum(len(u.audio) for u in result), len(pcm))
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in result]), pcm))

    def test_continuous_speech_waits_for_hard_cap_without_losing_audio(self):
        frames = 720
        pcm = np.full(frames * FRAME, .3, dtype=np.float32)
        probs = np.full(frames, .9, dtype=np.float32)
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0].end_s, 4.0)
        self.assertLess(result[0].end_s, 7.0)
        self.assertAlmostEqual(result[0].end_s, result[1].start_s, places=6)
        self.assertEqual(sum(len(u.audio) for u in result), len(pcm))
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in result]), pcm))

    def test_one_frame_dip_does_not_count_as_a_real_pause(self):
        frames = [np.full(FRAME, .3, dtype=np.float32) for _ in range(600)]
        probs = [.9] * 600
        frames[560][:] = 0
        probs[560] = 0.01
        selected = _select_longform_cut(
            frames, probs, search_frames=200, pause_frames=8,
            context_frames=12, negative_threshold=.25, hard=False,
        )
        self.assertIsNone(selected)
        fallback = _select_longform_cut(
            frames, probs, search_frames=200, pause_frames=8,
            context_frames=12, negative_threshold=.25, hard=True,
        )
        self.assertIsInstance(fallback, int)
        self.assertGreaterEqual(fallback, 400)
        self.assertLessEqual(fallback, 588)


if __name__ == "__main__":
    unittest.main()
