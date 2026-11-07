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

    def test_live_hls_audio_starts_near_live_edge(self):
        self.ingest.source.live = True
        self.ingest.source.location = "https://www.youtube.com/watch?v=live"
        self.ingest._vod_audio_url = None
        self.ingest._resolve_audio_url = lambda: "https://example.com/hls_playlist/index.m3u8"
        cmd = self.ingest._audio_commands()[0]
        self.assertEqual(cmd[cmd.index("-live_start_index") + 1], "-3")

    def test_non_hls_live_audio_does_not_receive_hls_flags(self):
        self.ingest.source.live = True
        self.ingest.source.location = "https://www.youtube.com/watch?v=live"
        self.ingest._vod_audio_url = None
        self.ingest._resolve_audio_url = lambda: "https://example.com/videoplayback?id=audio"
        self.assertNotIn("-live_start_index", self.ingest._audio_commands()[0])

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
