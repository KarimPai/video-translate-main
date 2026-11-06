import tempfile
import unittest
from pathlib import Path

from auto_translate.config import load_config


class PlamoConfigTests(unittest.TestCase):
    def test_defaults_select_plamo_3_translate_2b(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory) / "missing.toml")

        self.assertEqual(
            config.translate.plamo_hf_model,
            "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M",
        )
        self.assertEqual(config.translate.plamo_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.translate.plamo_context, 8192)
        self.assertEqual(config.translate.plamo_gpu_layers, 99)

    def test_legacy_ollama_settings_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[translate]\n'
                'model = "kaelri/hy-mt2:1.8b-q4_K_M"\n'
                'base_url = "http://127.0.0.1:11434/v1"\n'
                'temperature = 0.2\n'
                'context_lines = 6\n',
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(
            config.translate.plamo_hf_model,
            "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M",
        )
        self.assertEqual(config.translate.context_lines, 6)
        self.assertFalse(hasattr(config.translate, "model"))
        self.assertFalse(hasattr(config.translate, "base_url"))

    def test_plamo_settings_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[translate]\n'
                'plamo_context = 4096\n'
                'plamo_gpu_layers = 42\n'
                'plamo_base_url = "http://127.0.0.1:9000/v1"\n',
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.translate.plamo_context, 4096)
        self.assertEqual(config.translate.plamo_gpu_layers, 42)
        self.assertEqual(config.translate.plamo_base_url, "http://127.0.0.1:9000/v1")


if __name__ == "__main__":
    unittest.main()
