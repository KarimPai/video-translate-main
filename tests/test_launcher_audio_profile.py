"""Launcher contract for GTCRN and Sherpa opt-in CPU processing."""
import unittest
from pathlib import Path

from auto_translate.config import Config


class LauncherAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (Path(__file__).resolve().parents[1] / "Start translate.bat").read_text(
            encoding="utf-8-sig"
        )

    def test_launcher_keeps_sherpa_and_disables_gtcrn(self):
        self.assertIn("--audio-profile speaker", self.script)
        self.assertNotIn("--audio-profile both", self.script)
        self.assertNotIn("--audio-profile baseline --no-web", self.script)
        self.assertIn("uv sync --extra diarization", self.script)
        self.assertIn("m.version('sherpa-onnx')", self.script)
        self.assertIn('python -c "import sherpa_onnx"', self.script)

    def test_video_and_whisper_baseline_preserved(self):
        self.assertIn("--video-quality %VIDEO_QUALITY%", self.script)
        self.assertIn('set "VIDEO_QUALITY=720"', self.script)
        self.assertIn("--hwdec auto-safe", self.script)
        self.assertIn("--whisper-compute int8_float16", self.script)

    def test_default_models_and_vocal_separation_flags(self):
        cfg = Config()
        # The CLI profile sets GTCRN off regardless of older config.toml values.
        self.assertTrue(cfg.gtcrn.enabled)
        self.assertEqual(cfg.gtcrn.num_threads, 1)
        self.assertTrue(cfg.speaker.enabled)
        self.assertEqual(cfg.speaker.num_threads, 2)
        self.assertFalse(cfg.separation.enabled)


if __name__ == "__main__":
    unittest.main()
