"""Offline regressions for Windows shared HLS playlist stalls and recovery."""
import functools
import http.server
import io
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.ingest import Ingest, Source, _QuietHLSHandler
from auto_translate.output.mpv import MpvOutput


class SharedHlsRecoveryTests(unittest.TestCase):
    def test_playlist_is_served_from_memory_without_open_file_stream(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "index.m3u8"
            payload = b"#EXTM3U\n#EXT-X-VERSION:7\n#EXTINF:2.0,\nsegment_000001.m4s\n"
            manifest.write_bytes(payload)
            handler = functools.partial(_QuietHLSHandler, directory=td)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/index.m3u8"
                with urllib.request.urlopen(url, timeout=2) as response:
                    self.assertEqual(response.read(), payload)
                    self.assertIn("no-store", response.headers["Cache-Control"])
                    self.assertIn("application/vnd.apple.mpegurl",
                                  response.headers["Content-type"])
                # The handler must not leave an index.m3u8 file handle attached
                # to its network socket when the request has completed.
                replacement = Path(td) / "index.m3u8.tmp"
                replacement.write_bytes(payload + b"# updated\n")
                replacement.replace(manifest)
                self.assertTrue(manifest.read_bytes().endswith(b"# updated\n"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_hls_stall_detection_uses_both_playlist_and_media(self):
        self.assertIsNone(Ingest._video_stall_reason(5., 5.))
        self.assertEqual(Ingest._video_stall_reason(19., 1.),
                         "stalled_media_output")
        self.assertEqual(Ingest._video_stall_reason(2., 19.),
                         "stale_hls_playlist")

    def test_nonadvancing_ffmpeg_heartbeat_does_not_reset_watchdog(self):
        ing = object.__new__(Ingest)
        ing._video_progress_lock = threading.Lock()
        ing._video_progress_at = 100.
        ing._video_output_time_us = 2000000
        ing.video_out = "temp/index.m3u8"
        output = io.BytesIO(
            b"out_time_us=2000000\nprogress=continue\n"
            b"out_time_us=2000000\nprogress=continue\n"
        )
        with patch("auto_translate.ingest.time.monotonic", return_value=250.):
            ing._read_stderr(output, 0, capture_pts=False)
        self.assertEqual(ing._video_progress_at, 100.)
        with patch("auto_translate.ingest.time.monotonic", return_value=251.):
            ing._read_stderr(io.BytesIO(b"out_time_us=2000001\n"), 0,
                             capture_pts=False)
        self.assertEqual(ing._video_progress_at, 251.)

    def test_remote_live_hls_retries_network_errors_before_restart(self):
        ing = object.__new__(Ingest)
        ing.source = Source("https://youtube.example/live", live=True)
        ing._inputs = ["https://example.org/video.m3u8",
                       "https://example.org/audio.m3u8"]
        ing.video_out = "temp/index.m3u8"
        ing.burst_s = 0.
        command = ing._video_command()
        self.assertIn("-reconnect_on_network_error", command)
        self.assertEqual(
            command[command.index("-reconnect_on_network_error") + 1], "1"
        )
        self.assertIn("429,5xx", command)

    def test_three_recovery_limit_is_time_bounded(self):
        output = object.__new__(MpvOutput)
        output._video_retries = 3
        output._last_video_recovery_at = 100.
        output.stop = threading.Event()
        output._now = 0.
        output._shown = ""
        output._lock = threading.Lock()
        output._lines = []
        output.ingest = SimpleNamespace(
            source=SimpleNamespace(start_s=0., live=True),
            video_out="temp/index.m3u8",
            request_restart=Mock(),
            go=Mock(),
        )
        output.ipc = Mock()
        output.reset_buffer = Mock()
        with patch("auto_translate.output.mpv.time.monotonic", return_value=281.):
            self.assertTrue(output._recover_video_stream())
        self.assertEqual(output._video_retries, 1)
        output.ingest.request_restart.assert_called_once()
        self.assertFalse(output.stop.is_set())

    def test_repeated_restart_loop_is_bounded(self):
        output = object.__new__(MpvOutput)
        output._video_retries = 3
        output._last_video_recovery_at = 270.
        output.stop = threading.Event()
        with patch("auto_translate.output.mpv.time.monotonic", return_value=280.):
            self.assertFalse(output._recover_video_stream())
        self.assertTrue(output.stop.is_set())


if __name__ == "__main__":
    unittest.main()
