"""CPU-only coverage for adaptive speech, safe retries and quality telemetry."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import MpvConfig, TranslateConfig, VadConfig, vad_for_source
from auto_translate.glossary import Glossary
from auto_translate.output.mpv import MpvOutput
from auto_translate.pipeline import Stats
from auto_translate.translate.plamo import PlamoTranslator, _needs_fragment_context
from auto_translate.types import AudioChunk, Subtitle, Timings
from auto_translate.vad import FRAME, Segmenter


class AdaptiveVadTests(unittest.TestCase):
    def test_live_uses_shorter_natural_pause_target(self):
        cfg = VadConfig()
        live = vad_for_source(cfg, live=True)
        self.assertEqual(live.adaptive_split_s, 4.0)
        self.assertEqual(cfg.adaptive_split_s, 6.5)
        self.assertEqual(live.max_utterance_s, 7.0)

    def test_earlier_clear_pause_splits_without_losing_samples(self):
        # 12s; clear 300ms pause at 9.2s, well before the VOD 14s cap.
        frames = 1200
        pcm = np.full(frames * FRAME, .25, dtype=np.float32)
        probs = np.full(frames, .98, dtype=np.float32)
        pcm[920 * FRAME:950 * FRAME] = 0
        probs[920:950] = .01

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
        self.assertLess(utterances[0].end_s, 10.0)
        self.assertAlmostEqual(utterances[0].end_s, utterances[1].start_s, places=5)
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in utterances]), pcm))




class PLaMoQualityTests(unittest.TestCase):
    def test_only_short_dependant_fragments_trigger_context(self):
        self.assertTrue(_needs_fragment_context("それは"))
        self.assertTrue(_needs_fragment_context("行きたいけど"))
        self.assertFalse(_needs_fragment_context("ありがとうございます"))
        self.assertFalse(_needs_fragment_context("明日は車でゲームをしながらゆっくり帰ろうと思います"))

    def test_context_uses_previous_japanese_only_when_needed(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig()
        translator.glossary = Glossary({})
        translator.vod_title = "Example"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        translator.memory_lines = 0
        from unittest.mock import MagicMock
        translator.client = MagicMock()
        translator.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "That one."}}]}
        translator._chat("それは", [("赤い車がある", "There's a red car")], 2.0)
        body = translator.client.post.call_args.kwargs["json"]
        self.assertIn("Immediately preceding Japanese", body["chat_template_kwargs"]["context"])
        self.assertIn("赤い車がある", body["chat_template_kwargs"]["context"])
        self.assertTrue(body["messages"][-1]["content"] == "それは")

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
