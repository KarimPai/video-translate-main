"""CAT-only translation configuration and preserved Haas/Sherpa settings."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.config import AsrConfig, TranslateConfig, load_config
from auto_translate.translate import create_translator


class TranslationConfigurationTests(unittest.TestCase):
    def test_only_cat_translator_is_configured(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.cat_model, "cat-translate-3.3b")
        self.assertEqual(cfg.cat_hf_repo, "Sundowner123/cat-translate-3.3b-gguf")
        self.assertEqual(cfg.cat_hf_file, "cat-translate-3.3b-Q4_K_M.gguf")
        self.assertEqual(cfg.cat_base_url, "http://127.0.0.1:8003/v1")
        self.assertEqual(cfg.cat_context, 2048)
        self.assertEqual(cfg.cat_gpu_layers, 99)
        self.assertEqual(cfg.max_tokens, 192)
        self.assertEqual(cfg.context_lines, 0)
        self.assertEqual(cfg.memory_lines, 12)
        self.assertEqual(cfg.glossary, "glossary.toml")
        for obsolete in (
            "index_model", "index_hf_model", "index_base_url", "index_context",
            "qwen_model", "plamo_model", "quality_retry_enabled",
            "quality_retry_timeout_s", "temperature", "top_p",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertFalse(hasattr(cfg, obsolete))

    def test_cat_settings_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\ncat_context = 3072\ncat_gpu_layers = 50\n'
                'max_tokens = 96\ncat_base_url = "http://127.0.0.1:8004/v1"\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.cat_context, 3072)
        self.assertEqual(cfg.cat_gpu_layers, 50)
        self.assertEqual(cfg.max_tokens, 96)
        self.assertEqual(cfg.cat_base_url, "http://127.0.0.1:8004/v1")

    def test_retired_index_and_qwen_settings_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\nindex_model = "index-translate-2b"\n'
                'index_hf_model = "IndexTeam/Index-Translate-2B-GGUF:Q4_K_M"\n'
                'index_base_url = "http://127.0.0.1:8002/v1"\n'
                'quality_retry_enabled = true\nquality_retry_timeout_s = 4\n'
                'qwen_model = "qwen3.5-4b"\nplamo_model = "plamo"\n'
                'temperature = 0.7\ncontext_lines = 3\n'
                'cat_context = 1024\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.cat_model, "cat-translate-3.3b")
        self.assertEqual(cfg.cat_context, 1024)
        self.assertEqual(cfg.cat_base_url, "http://127.0.0.1:8003/v1")
        self.assertFalse(hasattr(cfg, "index_model"))
        self.assertFalse(hasattr(cfg, "quality_retry_enabled"))

    def test_factory_creates_cat_only(self):
        cfg, glossary = TranslateConfig(), object()
        with patch("auto_translate.translate.CatTranslator") as construct:
            result = create_translator(cfg, glossary, vod_title="Hololive GTA")
        self.assertIs(result, construct.return_value)
        construct.assert_called_once_with(cfg, glossary, vod_title="Hololive GTA")

    def test_haas_q8_backend_is_preserved(self):
        cfg = AsrConfig()
        self.assertEqual(cfg.haas_model, "haas-ja-v1.0:q8_0")
        self.assertEqual(cfg.haas_device, "Vulkan0")
        self.assertTrue(cfg.haas_decode_on_cpu)
        self.assertTrue(cfg.haas_keywords_enabled)

    def test_speaker_interruptions_remain_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "config.toml"
            cfg = load_config(file)
            self.assertTrue(cfg.speaker.enabled)
            self.assertTrue(cfg.speaker.interruption_handoffs)
            file.write_text("[speaker]\nenabled = false\n", encoding="utf-8")
            self.assertFalse(load_config(file).speaker.enabled)

    def test_missing_config_uses_cat(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_config(Path(temp) / "missing.toml")
        self.assertEqual(cfg.translate.cat_model, "cat-translate-3.3b")


if __name__ == "__main__":
    unittest.main()
