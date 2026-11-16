import json
import tempfile
import unittest
from pathlib import Path

from auto_translate.gui_build import build_identity, verify_build


class BuildIdentityTests(unittest.TestCase):
    def test_embedded_manifest_proves_commit_and_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "gui-build.json"
            expected = dict(schema_version=1, commit="a" * 40, gui_sha256="b" * 64, protocol=1)
            manifest.write_text(json.dumps(expected), encoding="utf-8-sig")
            data = build_identity(Path(folder), frozen=True, bundle_dir=folder)
            verify_build(data, "a" * 40, "B" * 64)
            self.assertTrue(data["packaged"])
            with self.assertRaises(ValueError):
                verify_build(data, "c" * 40, "b" * 64)
            with self.assertRaises(ValueError):
                verify_build(data, "a" * 40, "c" * 64)

    def test_old_packages_are_identified_and_cannot_pass_verified_install(self):
        with tempfile.TemporaryDirectory() as folder:
            data = build_identity(Path(folder), frozen=True, bundle_dir=folder)
            self.assertEqual(data["commit"], "unversioned")
            with self.assertRaises(ValueError):
                verify_build(data, "a" * 40, "b" * 64)

    def test_malformed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "gui-build.json").write_text('{"schema_version":1,"commit":"fake"}')
            with self.assertRaises(ValueError):
                build_identity(Path(folder), frozen=True, bundle_dir=folder)


if __name__ == "__main__":
    unittest.main()
