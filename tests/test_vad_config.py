"""Regression coverage for FireRedVAD defaults without loading its model."""

import tempfile
import unittest
from pathlib import Path

from auto_translate.config import VadConfig, load_config


class VadConfigurationTests(unittest.TestCase):
    def test_recommended_defaults(self):
        cfg = VadConfig()
        self.assertEqual(cfg.threshold, 0.30)
        self.assertEqual(cfg.neg_threshold, 0.20)
        self.assertEqual(cfg.speech_pad_ms, 200)
        self.assertEqual(cfg.min_silence_ms, 400)
        self.assertEqual(cfg.min_speech_ms, 100)
        self.assertEqual(cfg.max_utterance_s, 7.0)
        self.assertEqual(cfg.max_split_extension_s, 1.0)
        self.assertEqual(cfg.split_pause_min_ms, 80)
        self.assertEqual(cfg.split_context_ms, 120)
        self.assertEqual(cfg.fallback_s, 5.0)
        self.assertEqual(cfg.fallback_min_db, -50.0)

    def test_partial_override_preserves_other_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[vad]\nthreshold = 0.40\n", encoding="utf-8")
            cfg = load_config(path).vad
        self.assertEqual(cfg.threshold, 0.40)
        self.assertEqual(cfg.speech_pad_ms, 200)
        self.assertEqual(cfg.max_utterance_s, 7.0)

    def test_local_overrides_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                "[vad]\nspeech_pad_ms = 80\nmax_utterance_s = 5.0\n",
                encoding="utf-8",
            )
            cfg = load_config(path).vad
        self.assertEqual(cfg.speech_pad_ms, 80)
        self.assertEqual(cfg.max_utterance_s, 5.0)


if __name__ == "__main__":
    unittest.main()
