"""DeepFilterNet3 default route: configuration and FFmpeg input rate.

These tests need no downloaded model and do not import the optional enhancer.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from auto_translate.config import (
    Config, DeepFilterConfig, IngestConfig, SeparationConfig, load_config,
)
from auto_translate.ingest import Ingest


class DeepFilterConfigurationTests(unittest.TestCase):
    def test_deepfilter_is_enabled_by_default_and_uvr_remains_disabled(self):
        cfg = Config()
        self.assertTrue(cfg.deepfilter.enabled)
        self.assertFalse(cfg.separation.enabled)
        with tempfile.TemporaryDirectory() as temp:
            resolved = load_config(Path(temp) / "missing.toml")
            self.assertTrue(resolved.deepfilter.enabled)
            self.assertFalse(resolved.separation.enabled)

    def test_user_override_can_disable_deepfilter(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.toml"
            p.write_text("[deepfilter]\nenabled = false\n", encoding="utf-8")
            self.assertFalse(load_config(p).deepfilter.enabled)

    def test_input_audio_is_48khz_mono_when_deepfilter_is_enabled(self):
        source = SimpleNamespace(
            start_s=0.0, live=False, paced=False, location=__file__,
        )
        ingest = Ingest(
            source, IngestConfig(),
            separation_cfg=SeparationConfig(),
            deepfilter_cfg=DeepFilterConfig(),
        )
        self.assertTrue(ingest.deepfilter_enabled)
        self.assertFalse(ingest.separation_enabled)
        command = ingest._audio_commands()[0]
        graph = command[command.index("-af") + 1]
        self.assertIn("aresample=48000", graph)
        self.assertIn("aformat=sample_fmts=s16:channel_layouts=mono", graph)
        self.assertEqual(command[command.index("-ar") + 1], "48000")

    def test_uvr_and_deepfilter_cannot_run_together(self):
        source = SimpleNamespace(start_s=0.0, live=False, location=__file__)
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            Ingest(
                source, IngestConfig(),
                separation_cfg=SeparationConfig(enabled=True),
                deepfilter_cfg=DeepFilterConfig(enabled=True),
            )


if __name__ == "__main__":
    unittest.main()
