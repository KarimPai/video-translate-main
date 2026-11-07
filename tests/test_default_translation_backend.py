"""Translator defaults must not depend on an external HY-MT2 endpoint."""
import tempfile
import unittest
from pathlib import Path

from auto_translate.config import TranslateConfig, load_config


class TranslationBackendDefaultsTests(unittest.TestCase):
    def test_default_backend_is_managed_plamo(self):
        self.assertEqual(TranslateConfig().backend, "plamo")

    def test_missing_local_config_selects_plamo(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_config(Path(temp) / "config.toml")
        self.assertEqual(cfg.translate.backend, "plamo")
        self.assertEqual(cfg.translate.plamo_base_url, "http://127.0.0.1:8000/v1")

    def test_explicit_legacy_backend_remains_respected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('[translate]\nbackend = "hymt2"\n', encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg.translate.backend, "hymt2")


if __name__ == "__main__":
    unittest.main()
