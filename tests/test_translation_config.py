"""PLaMo-only translation configuration and preserved Haas/Sherpa settings."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.config import AsrConfig, TranslateConfig, load_config
from auto_translate.translate import create_translator


class TranslationConfigurationTests(unittest.TestCase):
    def test_only_plamo8b_translator_is_configured(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.plamo_model, "plamo-3-translate-8b")
        self.assertEqual(cfg.plamo_hf_repo, "pfnet/plamo-3-translate-8b-GGUF")
        self.assertEqual(cfg.plamo_hf_file, "plamo-3-translate-8b-Q3_K_M.gguf")
        self.assertEqual(cfg.plamo_base_url, "http://127.0.0.1:8005/v1")
        self.assertEqual(cfg.plamo_context, 1024)
        self.assertEqual(cfg.plamo_gpu_layers, 99)
        self.assertEqual(cfg.max_tokens, 128)
        self.assertEqual(cfg.context_lines, 2)
        self.assertTrue(cfg.context_title_enabled)
        self.assertEqual(cfg.memory_lines, 12)
        self.assertEqual(cfg.glossary, "glossary.toml")
        for obsolete in (
            "index_model", "index_hf_model", "index_base_url", "index_context",
            "qwen_model", "cat_model", "cat_context", "cat_base_url", "quality_retry_enabled",
            "quality_retry_timeout_s", "temperature", "top_p",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertFalse(hasattr(cfg, obsolete))

    def test_plamo_settings_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\nplamo_context = 3072\nplamo_gpu_layers = 50\n'
                'max_tokens = 96\nplamo_base_url = "http://127.0.0.1:8004/v1"\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.plamo_context, 3072)
        self.assertEqual(cfg.plamo_gpu_layers, 50)
        self.assertEqual(cfg.max_tokens, 96)
        self.assertEqual(cfg.plamo_base_url, "http://127.0.0.1:8004/v1")

    def test_retired_index_and_qwen_settings_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\nindex_model = "index-translate-2b"\n'
                'index_hf_model = "IndexTeam/Index-Translate-2B-GGUF:Q4_K_M"\n'
                'index_base_url = "http://127.0.0.1:8002/v1"\n'
                'quality_retry_enabled = true\nquality_retry_timeout_s = 4\n'
                'qwen_model = "qwen3.5-4b"\nlegacy_plamo_model = "plamo"\n'
                'temperature = 0.7\ncontext_lines = 3\n'
                'plamo_context = 1536\ncat_context = 4096\ncat_model = "retired"\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.plamo_model, "plamo-3-translate-8b")
        self.assertEqual(cfg.plamo_context, 1536)
        self.assertEqual(cfg.plamo_base_url, "http://127.0.0.1:8005/v1")
        self.assertFalse(hasattr(cfg, "index_model"))
        self.assertFalse(hasattr(cfg, "quality_retry_enabled"))

    def test_factory_creates_plamo8b_only(self):
        cfg, glossary = TranslateConfig(), object()
        with patch("auto_translate.translate.PlamoTranslator") as construct:
            result = create_translator(cfg, glossary, vod_title="Hololive GTA")
        self.assertIs(result, construct.return_value)
        construct.assert_called_once_with(cfg, glossary, vod_title="Hololive GTA")

    def test_haas_q8_backend_is_preserved(self):
        cfg = AsrConfig()
        self.assertEqual(cfg.haas_model, "haas-ja-v1.0:q8_0")
        self.assertEqual(cfg.haas_device, "Vulkan0")
        self.assertTrue(cfg.haas_decode_on_cpu)
        self.assertTrue(cfg.haas_keywords_enabled)
        self.assertTrue(cfg.youtube_captions_first)

    def test_youtube_caption_priority_is_local_configurable(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "config.toml"
            file.write_text("[asr]\\nyoutube_captions_first = false\\n".replace("\\\\n", "\\n"), encoding="utf-8")
            self.assertFalse(load_config(file).asr.youtube_captions_first)

    def test_speaker_interruptions_remain_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "config.toml"
            cfg = load_config(file)
            self.assertTrue(cfg.speaker.enabled)
            self.assertTrue(cfg.speaker.interruption_handoffs)
            file.write_text("[speaker]\nenabled = false\n", encoding="utf-8")
            self.assertFalse(load_config(file).speaker.enabled)

    def test_missing_config_uses_plamo8b(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_config(Path(temp) / "missing.toml")
        self.assertEqual(cfg.translate.plamo_model, "plamo-3-translate-8b")


if __name__ == "__main__":
    unittest.main()
