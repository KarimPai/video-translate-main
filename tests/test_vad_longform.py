"""Offline long-form VAD split regressions; no FireRed model download."""
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig
from auto_translate.types import SAMPLE_RATE, AudioChunk
from auto_translate.vad import FRAME, Segmenter, _select_longform_cut


class LongformBoundaryTests(unittest.TestCase):
    @staticmethod
    def _process(pcm, probs, cfg=None, generation=0):
        class FakeVAD:
            def __call__(self, audio):
                assert len(probs) == len(audio) // FRAME
                return probs
            def reset(self):
                pass
        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=FakeVAD()):
            segmenter = Segmenter(cfg or VadConfig())
            if generation:
                segmenter.feed(AudioChunk(0, pcm[:0], 0.0, reset=True, generation=generation))
            emitted = segmenter.feed(AudioChunk(0, pcm, 0.0, generation=generation))
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
        self.assertEqual(self._process(pcm, probs, VadConfig(min_speech_ms=250)), [])

    def test_quiet_sustained_speech_survives_recall_threshold(self):
        frames = 300
        pcm = np.full(frames * FRAME, .05, dtype=np.float32)
        probs = np.full(frames, .01, dtype=np.float32)
        probs[100:170] = .36  # above new .34 threshold, below old .38
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].speech_start_s, 1.0, delta=.07)
        self.assertGreater(result[0].vad_confidence, .34)

    def test_short_low_confidence_game_noise_is_still_rejected(self):
        frames = 300
        pcm = np.full(frames * FRAME, .2, dtype=np.float32)
        probs = np.full(frames, .01, dtype=np.float32)
        probs[100:114] = .36  # 140ms, below retained 200ms minimum
        self.assertEqual(self._process(pcm, probs), [])

    def test_high_confidence_short_japanese_reply_survives(self):
        frames = 300
        pcm = np.full(frames * FRAME, 0.25, dtype=np.float32)
        probs = np.full(frames, 0.01, dtype=np.float32)
        # 180ms voice: under the ordinary 200ms limit but strong enough
        # to qualify for the 120ms high-confidence short-reply exception.
        probs[120:138] = 0.99
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].vad_confidence, 0.45)

    def test_weak_short_noise_does_not_use_short_reply_exception(self):
        frames = 300
        pcm = np.full(frames * FRAME, 0.25, dtype=np.float32)
        probs = np.full(frames, 0.01, dtype=np.float32)
        probs[120:138] = 0.60
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

    def test_forced_cut_during_pause_uses_next_speech_onset_not_silent_remainder(self):
        frames = 170
        pcm = np.full(frames * FRAME, .3, dtype=np.float32)
        probs = np.full(frames, .95, dtype=np.float32)
        pcm[65 * FRAME:110 * FRAME] = 0.
        probs[65:110] = .01
        result = self._process(pcm, probs, VadConfig(
            max_utterance_s=1., max_split_extension_s=0., prob_smooth_frames=1,
            adaptive_split=False,
        ))
        self.assertGreaterEqual(len(result), 2)
        self.assertLess(result[1].start_s, 1.1)  # retained silent ASR context
        self.assertAlmostEqual(result[1].speech_start_s, 1.1, places=6)
        np.testing.assert_array_equal(np.concatenate([u.audio for u in result]), pcm)

    def test_pre_roll_keeps_name_audio_but_does_not_advance_speech_timestamp(self):
        pcm = np.zeros(250 * FRAME, dtype=np.float32)
        pcm[100 * FRAME:170 * FRAME] = .15
        probs = np.full(250, .01, dtype=np.float32)
        probs[100:170] = .95
        result = self._process(pcm, probs, VadConfig(prob_smooth_frames=1))
        self.assertEqual(len(result), 1)
        cue = result[0]
        self.assertAlmostEqual(cue.speech_start_s, 1., places=6)
        self.assertAlmostEqual(cue.start_s, .6, places=6)
        voice = cue.audio[round((1. - cue.start_s) * SAMPLE_RATE):
                          round((1.7 - cue.start_s) * SAMPLE_RATE)]
        np.testing.assert_array_equal(voice, pcm[100 * FRAME:170 * FRAME])

    def test_accuracy_first_vod_hard_cap_does_not_lose_pcm(self):
        # A 17-second uninterrupted utterance is segmented by the 16s cap,
        # without audio gaps or double-processing at the boundary.
        frames = 1700
        pcm = np.full(frames * FRAME, 0.3, dtype=np.float32)
        probs = np.full(frames, 0.95, dtype=np.float32)
        result = self._process(pcm, probs)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0].end_s, 12.0)
        self.assertLessEqual(result[0].end_s, 16.0)
        self.assertAlmostEqual(result[0].end_s, result[1].start_s, places=6)
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in result]), pcm))

    def test_first_seek_cue_is_short_then_normal_context_returns(self):
        pcm = np.full(2400 * FRAME, .3, dtype=np.float32)
        result = self._process(pcm, np.full(2400, .95), generation=1)
        self.assertLessEqual(result[0].end_s, 6.0)
        self.assertGreater(result[1].end_s - result[1].start_s, 12.0)
        np.testing.assert_array_equal(np.concatenate([u.audio for u in result]), pcm)

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
