"""PLaMo-only translation regression tests.

No model or llama-server is started; the translator constructor is mocked.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.config import TranslateConfig, load_config
from auto_translate.translate import create_translator


class TranslationConfigurationTests(unittest.TestCase):
    def test_defaults_select_plamo_without_backend_switch(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.plamo_hf_model, "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M")
        self.assertEqual(cfg.context_lines, 8)
        self.assertEqual(cfg.memory_lines, 24)
        for obsolete in ("backend", "model", "base_url", "extra_body", "min_request_interval_s"):
            self.assertFalse(hasattr(cfg, obsolete), obsolete)

    def test_old_local_engine_keys_are_ignored_but_plamo_options_survive(self):
        text = """
[translate]
backend = "previous-engine"
model = "unused-model"
base_url = "http://localhost:11434/v1"
temperature = 0.3
min_request_interval_s = 3
extra_body = { unused = true }
plamo_model = "plamo-3-translate"
plamo_gpu_layers = 66
context_lines = 4
memory_lines = 12
[translate.presets.previous]
model = "unused-preset"
"""
        with tempfile.TemporaryDirectory() as folder:
            cfg_path = Path(folder) / "config.toml"
            cfg_path.write_text(text, encoding="utf-8")
            cfg = load_config(cfg_path)
        self.assertEqual(cfg.translate.plamo_model, "plamo-3-translate")
        self.assertEqual(cfg.translate.plamo_gpu_layers, 66)
        self.assertEqual(cfg.translate.context_lines, 4)
        self.assertEqual(cfg.translate.memory_lines, 12)
        self.assertFalse(hasattr(cfg.translate, "backend"))

    def test_factory_always_creates_plamo_with_glossary_and_video_title(self):
        cfg = TranslateConfig()
        glossary = object()
        with patch("auto_translate.translate.PlamoTranslator") as constructor:
            instance = create_translator(cfg, glossary, vod_title="Hololive VOD")
        self.assertIs(instance, constructor.return_value)
        constructor.assert_called_once_with(
            cfg, glossary, vod_title="Hololive VOD"
        )

    def test_no_config_file_still_uses_plamo_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config(Path(folder) / "missing.toml")
        self.assertEqual(cfg.translate.plamo_model, "plamo-3-translate")


if __name__ == "__main__":
    unittest.main()
