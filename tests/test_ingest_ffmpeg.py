"""FFmpeg command regressions: no frame-by-frame audio diagnostic spam."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from auto_translate.ingest import Ingest


class AudioCommandTests(unittest.TestCase):
    def setUp(self):
        self.ingest = object.__new__(Ingest)
        self.ingest.source = SimpleNamespace(
            start_s=0.0, live=False, paced=False, location=str(Path(__file__).resolve())
        )
        self.ingest.pitch_semitones = 0.0
        self.ingest.separation_enabled = False
        self.ingest.deepfilter_enabled = False

    def test_direct_audio_command_disables_per_frame_diagnostics(self):
        cmd = self.ingest._audio_commands()[0]
        self.assertNotIn("ashowinfo", " ".join(cmd))
        self.assertEqual(cmd[cmd.index("-loglevel") + 1], "warning")

    def test_fmp4_fallback_audio_disables_per_frame_diagnostics(self):
        root = Path(__file__).resolve().parent / "fake-fmp4"
        self.ingest._cache_root = root
        self.ingest._http_server = SimpleNamespace(server_port=8971)
        self.ingest.video_out = str(root / "session-1" / "playlist.m3u8")
        self.ingest._cache_source_start_s = 0.0
        cmd = self.ingest._video_audio_command()
        self.assertNotIn("ashowinfo", " ".join(cmd))
        self.assertEqual(cmd[cmd.index("-loglevel") + 1], "warning")
        self.assertIn("http://127.0.0.1:8971/session-1/playlist.m3u8", cmd)


if __name__ == "__main__":
    unittest.main()
