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

    def test_launcher_uses_raw_audio(self):
        self.assertNotIn("sherpa", self.script.lower())
        self.assertNotIn("--extra diarization", self.script)
        self.assertNotIn("--audio-profile", self.script)

    def test_video_and_whisper_baseline_preserved(self):
        self.assertIn("--video-quality %VIDEO_QUALITY%", self.script)
        self.assertIn('if not defined QUALITY_CHOICE set "QUALITY_CHOICE=3"', self.script)
        for choice, quality in ((1, 360), (2, 480), (3, 720), (4, 1080)):
            self.assertIn(
                f'if "%QUALITY_CHOICE%"=="{choice}" set "VIDEO_QUALITY={quality}"',
                self.script,
            )
        self.assertIn("Enter 1, 2, 3, or 4", self.script)
        self.assertIn("--hwdec auto-safe", self.script)
        self.assertIn("--whisper-compute int8_float16", self.script)

    def test_default_models_and_vocal_separation_flags(self):
        cfg = Config()
        self.assertFalse(hasattr(cfg, "gtcrn"))
        self.assertFalse(hasattr(cfg, "speaker"))
        self.assertFalse(cfg.separation.enabled)


if __name__ == "__main__":
    unittest.main()
