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

    def test_short_engine_like_confidence_spike_is_not_speech(self):
        frames = 300
        pcm = np.full(frames * FRAME, 0.25, dtype=np.float32)
        probs = np.full(frames, 0.01, dtype=np.float32)
        probs[120:126] = 0.98  # 60ms spike, too short for a speech onset
        self.assertEqual(self._process(pcm, probs), [])

    def test_very_short_confident_sound_is_rejected_on_emit(self):
        frames = 300
        pcm = np.full(frames * FRAME, 0.25, dtype=np.float32)
        probs = np.full(frames, 0.01, dtype=np.float32)
        probs[120:134] = 0.98  # 140ms, should not meet 250ms voiced minimum
        self.assertEqual(self._process(pcm, probs), [])

    def test_sustained_speech_survives_onset_filter(self):
        frames = 300
        pcm = np.full(frames * FRAME, 0.25, dtype=np.float32)
        probs = np.full(frames, 0.01, dtype=np.float32)
        probs[120:190] = 0.98  # sustained 700ms voiced speech
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].speech_start_s, 1.2, delta=0.05)
        self.assertIsNotNone(result[0].vad_confidence)
        self.assertGreater(result[0].vad_confidence, .9)

    def test_real_short_pause_selected_near_seven_second_soft_limit(self):
        # The current soft cap is 7s and the hard cap is 8s; a sustained
        # 200ms quiet gap just before 7s is preferred over a forced split.
        frames = 760
        pcm = np.full(frames * FRAME, .3, dtype=np.float32)
        probs = np.full(frames, .9, dtype=np.float32)
        pcm[690 * FRAME:710 * FRAME] = 0.0
        probs[690:710] = 0.05
        result = self._process(pcm, probs, VadConfig(max_utterance_s=7.0))
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0].end_s, 6.85)
        self.assertLess(result[0].end_s, 7.15)
        self.assertAlmostEqual(result[0].end_s, result[1].start_s, places=6)
        self.assertEqual(sum(len(u.audio) for u in result), len(pcm))
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in result]), pcm))

    def test_continuous_speech_waits_for_hard_cap_without_losing_audio(self):
        # 9s of continuous speech passes the 7s soft cap and reaches the
        # 8s hard cap before fallback cutting a low-energy neighborhood.
        frames = 900
        pcm = np.full(frames * FRAME, .3, dtype=np.float32)
        probs = np.full(frames, .9, dtype=np.float32)
        result = self._process(pcm, probs, VadConfig(max_utterance_s=7.0))
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0].end_s, 5.5)
        self.assertLess(result[0].end_s, 8.0)
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
