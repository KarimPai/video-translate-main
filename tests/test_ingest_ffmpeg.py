"""FFmpeg command regressions: no frame-by-frame audio diagnostic spam."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from auto_translate.ingest import Ingest
from auto_translate.config import IngestConfig, load_config


class AudioCommandTests(unittest.TestCase):
    def setUp(self):
        self.ingest = object.__new__(Ingest)
        self.ingest.source = SimpleNamespace(
            start_s=0.0, live=False, paced=False, location=str(Path(__file__).resolve())
        )
        self.ingest.separation_enabled = False

    def test_direct_audio_uses_original_pitch_only(self):
        cmd = self.ingest._audio_commands()[0]
        graph = cmd[cmd.index("-af") + 1]
        self.assertNotIn("asetrate=", graph)
        self.assertNotIn("atempo=", graph)
        self.assertIn("aresample=16000", graph)
        self.assertIn("aformat=sample_fmts=s16:channel_layouts=mono", graph)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertFalse(hasattr(IngestConfig(), "pitch_shift_semitones"))
        self.assertFalse(hasattr(self.ingest, "_pitch_filter"))

    def test_both_retired_pitch_config_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.toml"
            file.write_text(
                "[ingest]\npitch_semitones = -3.0\npitch_shift_semitones = -2.0\n",
                encoding="utf-8",
            )
            cfg = load_config(file)
            self.assertFalse(hasattr(cfg.ingest, "pitch_semitones"))
            self.assertFalse(hasattr(cfg.ingest, "pitch_shift_semitones"))

    def test_direct_audio_command_disables_per_frame_diagnostics(self):
        cmd = self.ingest._audio_commands()[0]
        self.assertNotIn("ashowinfo", " ".join(cmd))
        self.assertEqual(cmd[cmd.index("-loglevel") + 1], "warning")

    def test_completed_vod_prefers_https_audio_with_fallback(self):
        self.ingest.cfg = IngestConfig()
        self.ingest._audio_format_override = None
        self.ingest.source.location = "https://www.youtube.com/watch?v=some-vod"
        with patch("auto_translate.ingest.subprocess.run", return_value=SimpleNamespace(
            stdout="https://example.com/direct-audio\n",
        )) as run:
            self.assertEqual(
                self.ingest._resolve_audio_url(), "https://example.com/direct-audio",
            )
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-f") + 1], self.ingest.cfg.vod_ytdlp_format)
        self.assertIn("bestaudio[protocol=https]", cmd[cmd.index("-f") + 1])
        self.assertIn("/bestaudio", cmd[cmd.index("-f") + 1])

    def test_live_audio_selector_and_manual_recovery_override_unchanged(self):
        self.ingest.cfg = IngestConfig()
        self.ingest.source.live = True
        self.ingest.source.location = "https://www.youtube.com/live/some-stream"
        self.ingest._audio_format_override = None
        with patch("auto_translate.ingest.subprocess.run", return_value=SimpleNamespace(
            stdout="https://example.com/live-audio\n",
        )) as run:
            self.ingest._resolve_audio_url()
        self.assertEqual(
            run.call_args.args[0][run.call_args.args[0].index("-f") + 1],
            self.ingest.cfg.ytdlp_format,
        )
        self.ingest.source.live = False
        self.ingest._audio_format_override = "bestaudio[ext=m4a]/bestaudio"
        with patch("auto_translate.ingest.subprocess.run", return_value=SimpleNamespace(
            stdout="https://example.com/fallback-audio\n",
        )) as run:
            self.ingest._resolve_audio_url()
        self.assertEqual(
            run.call_args.args[0][run.call_args.args[0].index("-f") + 1],
            self.ingest._audio_format_override,
        )

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
        self.assertNotIn("asetrate=", " ".join(cmd))
        self.assertNotIn("atempo=", " ".join(cmd))


if __name__ == "__main__":
    unittest.main()
