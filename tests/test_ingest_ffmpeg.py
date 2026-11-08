"""FFmpeg command regressions: no frame-by-frame audio diagnostic spam."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from auto_translate.ingest import Ingest
from auto_translate.config import IngestConfig, load_config


class AudioCommandTests(unittest.TestCase):
    def setUp(self):
        self.ingest = object.__new__(Ingest)
        self.ingest.source = SimpleNamespace(
            start_s=0.0, live=False, paced=False, location=str(Path(__file__).resolve())
        )
        self.ingest.pitch_semitones = 0.0
        self.ingest.separation_enabled = False
        self.ingest.deepfilter_enabled = False

    def test_pitch_shift_is_disabled_by_default(self):
        self.assertEqual(IngestConfig().pitch_shift_semitones, 0.0)
        self.assertEqual(self.ingest._pitch_filter(16000), [])
        cmd = self.ingest._audio_commands()[0]
        self.assertNotIn("asetrate=", " ".join(cmd))
        self.assertNotIn("atempo=", " ".join(cmd))

    def test_two_semitones_down_uses_duration_corrected_asr_audio_only(self):
        self.ingest.pitch_semitones = -2.0
        cmd = self.ingest._audio_commands()[0]
        graph = cmd[cmd.index("-af") + 1]
        self.assertIn("asetrate=", graph)
        self.assertIn("aresample=16000", graph)
        self.assertIn("atempo=", graph)
        self.assertIn("aformat=sample_fmts=s16:channel_layouts=mono", graph)
        ratio = 2 ** (-2 / 12)
        self.assertIn(f"asetrate={16000 * ratio:.8f}", graph)
        self.assertIn(f"atempo={1 / ratio:.8f}", graph)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")

    def test_explicit_opt_in_and_legacy_pitch_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.toml"
            file.write_text("[ingest]\npitch_semitones = -3.0\n", encoding="utf-8")
            self.assertEqual(load_config(file).ingest.pitch_shift_semitones, 0.0)
            file.write_text("[ingest]\npitch_shift_semitones = -2.0\n", encoding="utf-8")
            self.assertEqual(load_config(file).ingest.pitch_shift_semitones, -2.0)
            file.write_text("[ingest]\npitch_shift_semitones = -6.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pitch_shift_semitones"):
                load_config(file)

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
