"""Offline GUI contract tests; Qt/Windows/video hardware not required."""
import sys
import tempfile
import os
import subprocess
from unittest.mock import patch
import unittest
from pathlib import Path

from auto_translate.config import MpvConfig
from auto_translate.windows_subprocess import hidden_console_flags
from auto_translate.gui_runner import (
    QUALITY_OPTIONS, READY_MARKER, backend_python, build_backend_command,
    discover_project_root, prepare_project, subprocess_environment, valid_youtube_url,
)


class GuiRunnerTests(unittest.TestCase):
    def test_youtube_links(self):
        for url in (
            "https://www.youtube.com/watch?v=jd7Bp0prwiI",
            "https://www.youtube.com/live/jd7Bp0prwiI?si=x",
            "https://youtu.be/jd7Bp0prwiI",
            "https://m.youtube.com/shorts/jd7Bp0prwiI",
        ):
            with self.subTest(url=url):
                self.assertTrue(valid_youtube_url(url))
        for url in (
            "", "not a link", "https://example.com/",
            "https://youtube.com.evil.test/watch?v=a",
            "https://evil.youtube.com/watch?v=a",
            "file:///C:/Windows", "https://www.youtube.com/watch",
            "https://www.youtube.com/live/", "https://u:p@youtube.com/watch?v=a",
            "javascript:alert(1)",
        ):
            with self.subTest(url=url):
                self.assertFalse(valid_youtube_url(url))

    def test_build_command_is_argument_array_and_retains_all_existing_backend_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            venv = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            venv.parent.mkdir(parents=True)
            venv.touch()
            if sys.platform == "win32":
                venv.with_name("pythonw.exe").touch()
            argv0, args = build_backend_command(
                root, " https://www.youtube.com/live/jd7Bp0prwiI?si=x ", 720, 24680,
                frozen=True,
            )
            self.assertEqual(argv0, str(venv.with_name("pythonw.exe") if sys.platform == "win32" else venv))
            self.assertEqual(args[args.index("--mpv-wid") + 1], "24680")
            self.assertEqual(args[args.index("--video-quality") + 1], "720")
            self.assertIn("--no-rtx-vsr", args)
            self.assertNotIn("--rtx-vsr", args)
            _, vsr_args = build_backend_command(
                root, "https://www.youtube.com/watch?v=jd7Bp0prwiI",
                480, 24680, rtx_vsr=True, frozen=True,
            )
            self.assertIn("--rtx-vsr", vsr_args)
            self.assertNotIn("--no-rtx-vsr", vsr_args)
            _, off_args = build_backend_command(
                root, "https://www.youtube.com/watch?v=jd7Bp0prwiI",
                480, 24680, rtx_vsr=False, frozen=True,
            )
            self.assertIn("--no-rtx-vsr", off_args)
            for unavailable in (360, 720, 1080):
                with self.assertRaisesRegex(ValueError, "480p"):
                    build_backend_command(
                        root, "https://www.youtube.com/watch?v=jd7Bp0prwiI",
                        unavailable, 24680, rtx_vsr=True, frozen=True,
                    )
            for flag in ("-u", "--mpv", "--no-web", "--shared-gpu-playback",
                         "--simple-terminal", "--hwdec", "--gui-control"):
                self.assertIn(flag, args)
            self.assertEqual(args[3], "https://www.youtube.com/live/jd7Bp0prwiI?si=x")
            for bad in (0, -1):
                with self.assertRaises(ValueError):
                    build_backend_command(root, "https://youtu.be/abc", 720, bad)
            with self.assertRaises(ValueError):
                build_backend_command(root, "https://youtu.be/abc", 1440, 42)

    def test_start_config_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.dual-asr.example.toml").write_text("[mpv]\n", encoding="utf8")
            prepare_project(root)
            self.assertEqual((root / "config.toml").read_text(), "[mpv]\n")
            (root / "config.toml").write_text("custom = true\n")
            prepare_project(root)
            self.assertEqual((root / "config.toml").read_text(), "custom = true\n")
            self.assertTrue((root / "logs").is_dir())

    def test_project_discovery_from_dist_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src/auto_translate").mkdir(parents=True)
            (root / "pyproject.toml").touch()
            (root / "src/auto_translate/__main__.py").touch()
            exe = root / "dist/VideoTranslate/VideoTranslate.exe"
            exe.parent.mkdir(parents=True)
            exe.touch()
            self.assertEqual(discover_project_root(executable=str(exe), cwd=exe.parent), root)

    def test_gui_child_creationflags_only_when_gui_env_is_set(self):
        with patch.object(os, "name", "nt"), patch.object(
                subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True):
            with patch.dict(os.environ, {"VIDEO_TRANSLATE_GUI": "1"}):
                self.assertEqual(hidden_console_flags(), 0x08000000)
            with patch.dict(os.environ, {"VIDEO_TRANSLATE_GUI": "0"}):
                self.assertEqual(hidden_console_flags(), 0)
        if sys.platform != "win32":
            with patch.dict(os.environ, {"VIDEO_TRANSLATE_GUI": "1"}):
                self.assertEqual(hidden_console_flags(), 0)

    def test_environment_and_config_embedding_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = subprocess_environment(root, {"PATH": "test-path"})
            self.assertIn(str(root / "src"), env["PYTHONPATH"])
            self.assertEqual(env["PYTHONUNBUFFERED"], "1")
            self.assertEqual(env["PYTHONUTF8"], "1")
            self.assertEqual(env["VIDEO_TRANSLATE_GUI"], "1")
            self.assertEqual(MpvConfig().window_id, None)
            self.assertEqual(QUALITY_OPTIONS, (360, 480, 720, 1080))
            self.assertEqual(READY_MARKER, "VIDEO_TRANSLATE_GUI_READY")


if __name__ == "__main__":
    unittest.main()
