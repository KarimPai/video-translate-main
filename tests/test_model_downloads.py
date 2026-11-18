import hashlib
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "model installer dependency unavailable")
class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("auto_translate.models")

    def model(self, data):
        return self.module.ModelFile("test/repo", "revision", "model.gguf", "models/test",
                                     len(data), hashlib.sha256(data).hexdigest())

    def test_main_reuses_configured_paths_without_download(self):
        from auto_translate.config import Config
        cfg = Config()
        cfg.asr.model_path = "shared/asr"
        cfg.translate.model_path = "shared/translation.gguf"
        with patch("auto_translate.config.load_config", return_value=cfg), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(self.module, "verify_file", return_value=True) as verify, \
                patch.object(self.module, "snapshot_download") as snapshot, \
                patch.object(self.module, "hf_hub_download") as download:
            self.module.main()
        verify.assert_called_once_with(Path(cfg.translate.model_path), self.module.MODELS[0])
        snapshot.assert_not_called()
        download.assert_not_called()

    def test_valid_existing_model_is_reused_without_download(self):
        data = b"GGUFtest-model"
        model = self.model(data)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / model.directory / model.filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(data)
            with patch.object(self.module, "hf_hub_download") as download:
                self.assertEqual(self.module.ensure_model(model, root), destination)
            download.assert_not_called()

    def test_corrupt_existing_model_is_replaced_only_after_cache_validation(self):
        data = b"GGUFtest-model"
        model = self.model(data)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / model.directory / model.filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"bad")
            cached = root / "cached.gguf"
            cached.write_bytes(data)
            with patch.object(self.module, "hf_hub_download", return_value=str(cached)) as download:
                self.module.ensure_model(model, root)
            self.assertEqual(destination.read_bytes(), data)
            download.assert_called_once_with(repo_id="test/repo", revision="revision", filename="model.gguf")
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_corrupt_download_preserves_existing_file(self):
        model = self.model(b"GGUFtest-model")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / model.directory / model.filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"existing")
            cached = root / "cached.gguf"
            cached.write_bytes(b"corrupt")
            with patch.object(self.module, "hf_hub_download", return_value=str(cached)):
                with self.assertRaisesRegex(ValueError, "verification"):
                    self.module.ensure_model(model, root)
            self.assertEqual(destination.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
