"""Japanese-only Whisper Turbo: legacy routing fields are ignored."""
import tempfile
import unittest
from pathlib import Path

from auto_translate.asr.dual import DualASR
from auto_translate.asr.whisper_turbo import WhisperTurboASR
from auto_translate.config import Config, load_config


class JapaneseOnlyRoutingTests(unittest.TestCase):
    def test_legacy_routing_option_is_absent(self):
        self.assertFalse(hasattr(Config().dual_asr, "language_routing"))
        self.assertFalse(hasattr(DualASR, "preflight_language"))

    def test_whisper_turbo_is_transcription_only(self):
        self.assertFalse(hasattr(WhisperTurboASR, "transcribe_with_language"))

    def test_old_configs_still_load_without_reintroducing_language_routing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.toml"
            path.write_text("[dual_asr]\nlanguage_routing = true\n", encoding="utf-8")
            cfg = load_config(path)
        self.assertTrue(cfg.dual_asr.enabled)
        self.assertFalse(hasattr(cfg.dual_asr, "language_routing"))


if __name__ == "__main__":
    unittest.main()
