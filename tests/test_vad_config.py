"""Regression coverage for FireRedVAD defaults without loading its model."""

import tempfile
import unittest
from pathlib import Path

from auto_translate.config import VadConfig, load_config, vad_for_source


class VadConfigurationTests(unittest.TestCase):
    def test_recommended_defaults(self):
        cfg = VadConfig()
        self.assertEqual(cfg.threshold, 0.34)
        self.assertEqual(cfg.neg_threshold, 0.25)
        self.assertEqual(cfg.speech_pad_ms, 400)
        self.assertEqual(cfg.min_silence_ms, 550)
        self.assertEqual(cfg.min_speech_ms, 200)
        self.assertEqual(cfg.onset_ms, 50)
        self.assertEqual(cfg.short_reply_ms, 120)
        self.assertEqual(cfg.short_reply_probability, 0.85)
        self.assertEqual(cfg.prob_smooth_frames, 5)
        self.assertEqual(cfg.max_utterance_s, 14.0)
        self.assertEqual(cfg.max_split_extension_s, 2.0)
        self.assertEqual(cfg.live_max_utterance_s, 7.0)
        self.assertEqual(cfg.live_max_split_extension_s, 1.0)
        self.assertEqual(cfg.split_pause_min_ms, 120)
        self.assertEqual(cfg.split_context_ms, 160)
        self.assertEqual(cfg.fallback_s, 0.0)
        self.assertEqual(cfg.fallback_min_db, -50.0)

    def test_partial_override_preserves_other_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[vad]\nthreshold = 0.40\n", encoding="utf-8")
            cfg = load_config(path).vad
        self.assertEqual(cfg.threshold, 0.40)
        self.assertEqual(cfg.speech_pad_ms, 400)
        self.assertEqual(cfg.min_silence_ms, 550)
        self.assertEqual(cfg.min_speech_ms, 200)
        self.assertEqual(cfg.onset_ms, 50)
        self.assertEqual(cfg.max_utterance_s, 14.0)

    def test_vod_vs_live_chunk_limits(self):
        configured = VadConfig(threshold=0.41, live_max_utterance_s=6.5)
        vod = vad_for_source(configured, live=False)
        live = vad_for_source(configured, live=True)
        self.assertIs(vod, configured)
        self.assertIsNot(live, configured)
        self.assertEqual((vod.max_utterance_s, vod.max_split_extension_s), (14.0, 2.0))
        self.assertEqual((live.max_utterance_s, live.max_split_extension_s), (6.5, 1.0))
        self.assertEqual((live.threshold, live.speech_pad_ms), (0.41, 400))
        self.assertEqual(configured.max_utterance_s, 14.0)

    def test_opt_in_short_vod_segments_do_not_affect_live(self):
        cfg = VadConfig(vod_short_segments=True)
        vod = vad_for_source(cfg, live=False)
        live = vad_for_source(cfg, live=True)
        self.assertIsNot(vod, cfg)
        self.assertEqual((vod.max_utterance_s, vod.max_split_extension_s,
                          vod.adaptive_split_s), (11.0, 2.0, 7.5))
        self.assertEqual((live.max_utterance_s, live.adaptive_split_s),
                         (7.0, 5.5))
        self.assertEqual((cfg.max_utterance_s, cfg.adaptive_split_s),
                         (14.0, 9.0))

    def test_short_vod_profile_is_optional_in_config_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[vad]\nvod_short_segments = true\n",
                            encoding="utf-8")
            cfg = load_config(path).vad
        self.assertTrue(cfg.vod_short_segments)
        self.assertEqual(vad_for_source(cfg, live=False).max_utterance_s, 11.0)

    def test_noise_probe_can_be_opted_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[vad]\nfallback_s = 5.0\n", encoding="utf-8")
            self.assertEqual(load_config(path).vad.fallback_s, 5.0)

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
