"""CPU-only coverage for adaptive speech, safe retries and quality telemetry."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import MpvConfig, VadConfig, vad_for_source
from auto_translate.output.mpv import MpvOutput
from auto_translate.pipeline import Stats
from auto_translate.types import AudioChunk, Subtitle, Timings
from auto_translate.vad import FRAME, Segmenter


class AdaptiveVadTests(unittest.TestCase):
    def test_live_uses_shorter_natural_pause_target(self):
        cfg = VadConfig()
        live = vad_for_source(cfg, live=True)
        self.assertEqual(live.adaptive_split_s, 4.0)
        self.assertEqual(cfg.adaptive_split_s, 5.0)
        self.assertEqual(live.max_utterance_s, 7.0)

    def test_earlier_clear_pause_splits_without_losing_samples(self):
        # 8s; a 300ms pause at 5.8s must split after the new 5s target,
        # without waiting for the old 6.5s target or the unchanged 14s cap.
        frames = 800
        pcm = np.full(frames * FRAME, .25, dtype=np.float32)
        probs = np.full(frames, .98, dtype=np.float32)
        pcm[580 * FRAME:610 * FRAME] = 0
        probs[580:610] = .01

        class FakeVAD:
            def reset(self):
                pass
            def __call__(self, audio):
                return probs[:len(audio) // FRAME]

        with patch("auto_translate.vad.StreamingFireRedVAD", return_value=FakeVAD()):
            segmenter = Segmenter(VadConfig())
            utterances = segmenter.feed(AudioChunk(0, pcm, 0.0))
            utterances += segmenter.flush()
        self.assertGreaterEqual(len(utterances), 2)
        self.assertLess(utterances[0].end_s, 6.5)
        self.assertAlmostEqual(utterances[0].end_s, utterances[1].start_s, places=5)
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in utterances]), pcm))




class PlaybackQualityTests(unittest.TestCase):
    def test_live_stale_window(self):
        player = object.__new__(MpvOutput)
        player.cfg = MpvConfig(live_late_window_s=7.0)
        player.ingest = SimpleNamespace(source=SimpleNamespace(live=True))
        # The production constructor initializes this; __new__ bypasses it.
        player._session_start = 0.0
        player._now = 17.0
        player._lines = []
        player._lock = threading.Lock()
        player._subtitle_ready = threading.Event()
        player.show(Subtitle(2.0, 7.0, "待って", "Wait!"))
        self.assertEqual(player._lines, [])

    def test_runtime_p95_and_deadline_misses(self):
        stats = Stats()
        stats.add(Subtitle(0, 1, "はい", "Yes", Timings(
            audio_read=10, asr_start=10, asr_end=11, mt_start=11, mt_end=20)),
            deadline_s=9.0)
        self.assertEqual(stats.deadline_misses, 1)
        self.assertIn("p95", stats.summary())


if __name__ == "__main__":
    unittest.main()
