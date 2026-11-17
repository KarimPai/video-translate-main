"""Prevent external MPV console windows while preserving Qt-embedded playback."""

import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvOutput


class MpvConsoleVisibilityTests(unittest.TestCase):
    def _launch(self, *, embedded: bool):
        cfg = MpvConfig(window_id=24680 if embedded else None)
        ingest = SimpleNamespace(
            source=SimpleNamespace(
                live=False, start_s=0.0,
                location="https://www.youtube.com/watch?v=example",
            ),
            cfg=SimpleNamespace(vod_max_audio_lead_s=75.0),
            video_out=None,
            enable_vod_playback_pacing=MagicMock(),
        )
        with (
            patch("auto_translate.output.mpv.find_mpv", return_value="mpv.exe"),
            patch("auto_translate.output.mpv.subprocess.Popen") as popen,
            patch("auto_translate.output.mpv.MpvIpc") as ipc,
            patch("auto_translate.output.mpv.threading.Thread"),
            patch("auto_translate.output.mpv.hidden_console_flags",
                  return_value=0x08000000 if embedded else 0),
            patch("auto_translate.output.mpv.time.sleep"),
        ):
            popen.return_value.poll.return_value = None
            ipc.return_value.get.return_value = False
            MpvOutput(cfg, ingest, threading.Event())
            return popen.call_args

    def test_embedded_mpv_is_console_free_with_ipc_and_video_intact(self):
        call = self._launch(embedded=True)
        command = call.args[0]
        self.assertIn("--wid=24680", command)
        self.assertIn("--no-terminal", command)
        self.assertIn("--hwdec=auto-safe", command)
        self.assertIn("--osc=no", command)
        self.assertFalse(any(x.startswith("--vf=") for x in command))
        self.assertTrue(any(x.startswith("--input-ipc-server=") for x in command))
        self.assertTrue(any(x.startswith("--log-file=") for x in command))
        self.assertEqual(call.kwargs["creationflags"], 0x08000000)
        self.assertEqual(call.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(call.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(call.kwargs["stderr"], subprocess.DEVNULL)

    def test_original_mpv_cli_mode_keeps_console(self):
        call = self._launch(embedded=False)
        command = call.args[0]
        self.assertNotIn("--no-terminal", command)
        self.assertIn("--hwdec=auto-safe", command)
        self.assertIn("--osc=yes", command)
        self.assertFalse(any(x.startswith("--wid=") for x in command))
        self.assertEqual(call.kwargs["creationflags"], 0)
        self.assertIsNone(call.kwargs["stdin"])
        self.assertIsNone(call.kwargs["stdout"])
        self.assertIsNone(call.kwargs["stderr"])


if __name__ == "__main__":
    unittest.main()
