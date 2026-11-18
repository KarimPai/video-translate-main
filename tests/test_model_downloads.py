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

    def test_verified_download_is_used_directly_without_touching_old_file(self):
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
                self.assertEqual(self.module.ensure_model(model, root), cached)
            self.assertEqual(destination.read_bytes(), b"bad")
            download.assert_called_once_with(repo_id="test/repo", revision="revision", filename="model.gguf")
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_shared_cache_is_reused_without_copying_or_downloading(self):
        data = b"GGUFtest-model"
        model = self.model(data)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cached = root / "shared.gguf"
            cached.write_bytes(data)
            with patch.object(self.module, "try_to_load_from_cache", return_value=str(cached)) as lookup, patch.object(
                    self.module, "hf_hub_download") as download:
                self.assertEqual(self.module.ensure_model(model, root), cached)
            lookup.assert_called_once_with(model.repo_id, model.filename, revision=model.revision)
            download.assert_not_called()
            self.assertFalse((root / model.directory).exists())

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
