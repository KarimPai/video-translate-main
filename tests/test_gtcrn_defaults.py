"""GTCRN defaults, 16 kHz FFmpeg routing, and local overrides."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from auto_translate.config import (
    Config, GtcrnConfig, IngestConfig, SeparationConfig, load_config,
)
from auto_translate.ingest import Ingest


class GtcrnConfigurationTests(unittest.TestCase):
    def test_gtcrn_is_enabled_by_default_and_uvr_is_off(self):
        cfg = Config()
        self.assertTrue(cfg.gtcrn.enabled)
        self.assertEqual(cfg.gtcrn.num_threads, 1)
        self.assertFalse(cfg.separation.enabled)
        with tempfile.TemporaryDirectory() as temp:
            resolved = load_config(Path(temp) / "missing.toml")
            self.assertTrue(resolved.gtcrn.enabled)

    def test_local_override_can_disable_gtcrn(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.toml"
            p.write_text("[gtcrn]\nenabled = false\n", encoding="utf-8")
            self.assertTrue(load_config(p).gtcrn.enabled)

    def test_retired_deepfilter_option_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.toml"
            p.write_text("[deepfilter]\nenabled = false\n", encoding="utf-8")
            self.assertFalse(load_config(p).gtcrn.enabled)

    def test_ffmpeg_outputs_16khz_mono_to_gtcrn(self):
        source = SimpleNamespace(
            start_s=0.0, live=False, paced=False, location=__file__,
        )
        ingest = Ingest(
            source, IngestConfig(),
            separation_cfg=SeparationConfig(),
            gtcrn_cfg=GtcrnConfig(enabled=True),
        )
        self.assertTrue(ingest.gtcrn_enabled)
        self.assertFalse(ingest.separation_enabled)
        command = ingest._audio_commands()[0]
        graph = command[command.index("-af") + 1]
        self.assertIn("aresample=16000", graph)
        self.assertIn("aformat=sample_fmts=s16:channel_layouts=mono", graph)
        self.assertEqual(command[command.index("-ar") + 1], "16000")

    def test_uvr_and_gtcrn_are_mutually_exclusive(self):
        source = SimpleNamespace(start_s=0.0, live=False, location=__file__)
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            Ingest(
                source, IngestConfig(),
                separation_cfg=SeparationConfig(enabled=True),
                gtcrn_cfg=GtcrnConfig(enabled=True),
            )


if __name__ == "__main__":
    unittest.main()
