"""Checkpoint reuse and corruption handling, without downloading model weights."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auto_translate import voxtral_model


class VoxtralCheckpointTests(unittest.TestCase):
    def test_integrity_rejects_missing_truncated_and_same_size_corruption(self):
        content = b"GGUF reviewed checkpoint"
        with TemporaryDirectory() as folder, patch.object(voxtral_model, "SIZE", len(content)), patch.object(
                voxtral_model, "SHA256", hashlib.sha256(content).hexdigest()):
            path = Path(folder) / voxtral_model.FILENAME
            self.assertFalse(voxtral_model.verify(path))
            path.write_bytes(content[:-1])
            self.assertFalse(voxtral_model.verify(path))
            path.write_bytes(b"X" + content[1:])
            self.assertFalse(voxtral_model.verify(path))
            path.write_bytes(content)
            self.assertTrue(voxtral_model.verify(path))

    def test_valid_checkpoint_reused_without_network(self):
        content = b"GGUF verified"
        fetch = Mock(side_effect=AssertionError("valid checkpoint must be reused"))
        with TemporaryDirectory() as folder, patch.object(voxtral_model, "SIZE", len(content)), patch.object(
                voxtral_model, "SHA256", hashlib.sha256(content).hexdigest()), patch.dict(
                "sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=fetch)}):
            path = Path(folder) / voxtral_model.FILENAME
            path.write_bytes(content)
            self.assertEqual(voxtral_model.download(folder), path)
            fetch.assert_not_called()

    def test_corrupt_file_forces_pinned_replacement(self):
        content = b"GGUF verified"
        with TemporaryDirectory() as folder, patch.object(voxtral_model, "SIZE", len(content)), patch.object(
                voxtral_model, "SHA256", hashlib.sha256(content).hexdigest()):
            path = Path(folder) / voxtral_model.FILENAME
            path.write_bytes(b"broken")

            def replace(*args, **kwargs):
                self.assertEqual(args, (voxtral_model.REPO, voxtral_model.FILENAME))
                self.assertEqual(kwargs["revision"], voxtral_model.REVISION)
                self.assertTrue(kwargs["force_download"])
                path.write_bytes(content)
                return str(path)

            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=replace)}):
                self.assertEqual(voxtral_model.download(folder), path)

    def test_download_with_wrong_digest_fails_explicitly(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / voxtral_model.FILENAME
            path.write_bytes(b"corrupt")
            fetch = Mock(return_value=str(path))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=fetch)}):
                with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                    voxtral_model.download(folder)


if __name__ == "__main__":
    unittest.main()
