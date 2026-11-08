"""Index-only translation configuration and existing Haas/Sherpa settings."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.config import AsrConfig, TranslateConfig, load_config
from auto_translate.translate import create_translator


class TranslationConfigurationTests(unittest.TestCase):
    def test_only_index_translator_is_configured(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.index_model, "index-translate-2b")
        self.assertEqual(cfg.index_hf_model, "IndexTeam/Index-Translate-2B-GGUF:Q4_K_M")
        self.assertEqual(cfg.index_base_url, "http://127.0.0.1:8002/v1")
        self.assertEqual(cfg.index_context, 4096)
        self.assertEqual(cfg.index_gpu_layers, 99)
        self.assertEqual(cfg.max_tokens, 160)
        self.assertEqual(cfg.context_lines, 2)
        self.assertTrue(cfg.quality_retry_enabled)
        self.assertEqual(cfg.quality_retry_timeout_s, 8.0)
        self.assertEqual(cfg.glossary, "glossary.toml")
        self.assertFalse(hasattr(cfg, "qwen_model"))
        self.assertFalse(hasattr(cfg, "plamo_model"))
        self.assertFalse(hasattr(cfg, "temperature"))
        self.assertFalse(hasattr(cfg, "quality_recheck_enabled"))

    def test_index_specific_settings_are_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\nindex_context = 2048\nindex_gpu_layers = 50\n'
                'context_lines = 6\nmax_tokens = 192\n'
                'quality_retry_enabled = false\nquality_retry_timeout_s = 5.0\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.index_context, 2048)
        self.assertEqual(cfg.index_gpu_layers, 50)
        self.assertEqual(cfg.context_lines, 6)
        self.assertEqual(cfg.max_tokens, 192)
        self.assertFalse(cfg.quality_retry_enabled)
        self.assertEqual(cfg.quality_retry_timeout_s, 5.0)

    def test_old_qwen_and_other_translator_settings_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[translate]\nqwen_model = "old-model"\n'
                'qwen_hf_model = "unsloth/legacy"\nqwen_context = 8192\n'
                'qwen_base_url = "http://127.0.0.1:8001/v1"\n'
                'temperature = 0.7\ntop_p = 0.8\n'
                'quality_recheck_enabled = true\n'
                'plamo_model = "legacy"\nbackend = "qwen"\n'
                'index_context = 3072\n'
                '[translate.presets.old]\nmodel = "ignored"\n',
                encoding="utf-8",
            )
            cfg = load_config(path).translate
        self.assertEqual(cfg.index_model, "index-translate-2b")
        self.assertEqual(cfg.index_context, 3072)
        self.assertEqual(cfg.index_base_url, "http://127.0.0.1:8002/v1")
        self.assertFalse(hasattr(cfg, "qwen_model"))
        self.assertFalse(hasattr(cfg, "plamo_model"))

    def test_factory_creates_index_with_glossary(self):
        cfg, glossary = TranslateConfig(), object()
        with patch("auto_translate.translate.IndexTranslator") as construct:
            result = create_translator(cfg, glossary, vod_title="Hololive GTA")
        self.assertIs(result, construct.return_value)
        construct.assert_called_once_with(cfg, glossary, vod_title="Hololive GTA")

    def test_speaker_diarization_stays_available(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "config.toml"
            self.assertTrue(load_config(file).speaker.enabled)
            file.write_text("[speaker]\nenabled = false\nnum_threads = 1\n", encoding="utf-8")
            cfg = load_config(file)
        self.assertFalse(cfg.speaker.enabled)
        self.assertEqual(cfg.speaker.num_threads, 1)

    def test_haas_q8_backend_is_unchanged(self):
        cfg = AsrConfig()
        self.assertEqual(cfg.haas_model, "haas-ja-v1.0:q8_0")
        self.assertEqual(cfg.haas_device, "Vulkan0")
        self.assertTrue(cfg.haas_decode_on_cpu)
        self.assertTrue(cfg.haas_keywords_enabled)
        self.assertFalse(hasattr(cfg, "model_path"))
        self.assertFalse(hasattr(cfg, "prompt"))

    def test_old_asr_config_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[asr]\nmodel_path = "old.gguf"\ndevice = "cpu"\n'
                'haas_device = "Vulkan0"\n',
                encoding="utf-8",
            )
            cfg = load_config(path).asr
        self.assertEqual(cfg.haas_model, "haas-ja-v1.0:q8_0")
        self.assertEqual(cfg.haas_device, "Vulkan0")

    def test_missing_config_uses_index(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_config(Path(temp) / "missing.toml")
        self.assertEqual(cfg.translate.index_model, "index-translate-2b")


if __name__ == "__main__":
    unittest.main()
