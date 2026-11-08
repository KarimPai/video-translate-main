"""Qwen-only configuration and legacy-config regression checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.config import TranslateConfig, load_config
from auto_translate.translate import create_translator


class TranslationConfigurationTests(unittest.TestCase):
    def test_defaults_are_local_qwen_only(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.qwen_hf_model, "unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
        self.assertEqual(cfg.qwen_model, "qwen3.5-4b")
        self.assertEqual(cfg.qwen_context, 4096)
        self.assertEqual(cfg.qwen_base_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(cfg.glossary, "glossary.toml")
        self.assertEqual(cfg.subtitle_style, "gist")
        self.assertTrue(cfg.quality_recheck_enabled)
        self.assertFalse(hasattr(cfg, "backend"))

    def test_legacy_local_config_does_not_reenable_old_engine(self):
        legacy = """
[translate]
backend = "legacy"
model = "ignored"
plamo_model = "unused"
plamo_hf_model = "unused"
plamo_base_url = "http://127.0.0.1:8000/v1"
plamo_gpu_layers = 66
context_lines = 4
memory_lines = 12
qwen_context = 3072
[translate.presets.old]
model = "ignored"
"""
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.toml"
            file.write_text(legacy, encoding="utf-8")
            cfg = load_config(file)
        self.assertEqual(cfg.translate.qwen_model, "qwen3.5-4b")
        self.assertEqual(cfg.translate.qwen_context, 3072)
        self.assertEqual(cfg.translate.context_lines, 4)
        self.assertFalse(hasattr(cfg.translate, "plamo_model"))

    def test_factory_uses_qwen_and_preserves_glossary(self):
        cfg, glossary = TranslateConfig(), object()
        with patch("auto_translate.translate.QwenTranslator") as constructor:
            actual = create_translator(cfg, glossary, vod_title="Hololive ARK")
        self.assertIs(actual, constructor.return_value)
        constructor.assert_called_once_with(cfg, glossary, vod_title="Hololive ARK")

    def test_cpu_speaker_diarization_setting_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.toml"
            self.assertTrue(load_config(file).speaker.enabled)
            file.write_text("[speaker]\nenabled = false\nnum_threads = 1\n", encoding="utf-8")
            cfg = load_config(file)
            self.assertFalse(cfg.speaker.enabled)
            self.assertEqual(cfg.speaker.num_threads, 1)

    def test_missing_config_is_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp) / "missing.toml")
        self.assertEqual(cfg.translate.qwen_model, "qwen3.5-4b")


if __name__ == "__main__":
    unittest.main()
