"""Guard against shared live HLS video-only playback and MPV audio underruns."""
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.ingest import Ingest, Source
from auto_translate.output.mpv import MpvOutput


class LiveAudioUnderrunTests(unittest.TestCase):
    @staticmethod
    def player():
        player = object.__new__(MpvOutput)
        player.ingest = SimpleNamespace(
            source=SimpleNamespace(live=True),
            video_out="temp/fmp4/live/index.m3u8",
            video_failed=threading.Event(),
        )
        player.ipc = Mock()
        player.ipc.get.side_effect = lambda key: {
            "pause": False, "paused-for-cache": True
        }[key]
        player._live_cache_pause_since = None
        player._next_live_cache_check_at = 0.0
        return player

    def test_sustained_mpv_cache_pause_requests_shared_recovery(self):
        player = self.player()
        with patch("auto_translate.output.mpv.time.monotonic", return_value=100.):
            self.assertFalse(player._check_live_cache_stall())
        with patch("auto_translate.output.mpv.time.monotonic", return_value=105.):
            self.assertFalse(player._check_live_cache_stall())
        with patch("auto_translate.output.mpv.time.monotonic", return_value=110.):
            self.assertTrue(player._check_live_cache_stall())
        self.assertTrue(player.ingest.video_failed.is_set())

    def test_live_cache_pause_clears_when_playback_resumes(self):
        player = self.player()
        with patch("auto_translate.output.mpv.time.monotonic", return_value=100.):
            self.assertFalse(player._check_live_cache_stall())
        player.ipc.get.side_effect = lambda key: {
            "pause": False, "paused-for-cache": False
        }[key]
        with patch("auto_translate.output.mpv.time.monotonic", return_value=101.):
            self.assertFalse(player._check_live_cache_stall())
        self.assertIsNone(player._live_cache_pause_since)

    def test_manual_pause_is_not_a_stall(self):
        player = self.player()
        player.ipc.get.side_effect = lambda key: {
            "pause": True, "paused-for-cache": True
        }[key]
        with patch("auto_translate.output.mpv.time.monotonic", return_value=100.):
            self.assertFalse(player._check_live_cache_stall())
        with patch("auto_translate.output.mpv.time.monotonic", return_value=150.):
            self.assertFalse(player._check_live_cache_stall())
        self.assertFalse(player.ingest.video_failed.is_set())

    @staticmethod
    def video_writer():
        ingest = object.__new__(Ingest)
        ingest.source = Source("https://youtube.invalid/live", live=True)
        ingest.video_out = "temp/fmp4/live/index.m3u8"
        ingest._proc_lock = threading.Lock()
        ingest._video_proc = Mock()
        ingest._stopped = False
        ingest._video_failed = threading.Event()
        ingest._video_progress_lock = threading.Lock()
        ingest._video_output_time_us = None
        ingest._video_progress_at = 0.0
        return ingest

    def test_single_missing_segment_does_not_interrupt_live_playback(self):
        ingest = self.video_writer()
        error = io.BytesIO(
            b"[in#1/hls] Failed to open segment 3423 of playlist 0\n"
        )
        ingest._read_stderr(
            error, 0, capture_pts=False, video_process=ingest._video_proc
        )
        self.assertFalse(ingest.video_failed.is_set())

    def test_unrecoverable_playlist_write_failure_still_restarts(self):
        ingest = self.video_writer()
        ingest._read_stderr(
            io.BytesIO(
                b"Failed to rename index.m3u8.tmp to index.m3u8: "
                b"Operation not permitted\n"
            ),
            0, capture_pts=False, video_process=ingest._video_proc,
        )
        self.assertTrue(ingest.video_failed.is_set())

    def test_stale_worker_cannot_recover_new_session(self):
        ingest = self.video_writer()
        old_worker = Mock()
        ingest._read_stderr(
            io.BytesIO(b"[in#1/hls] Failed to open segment 3423\n"),
            0, capture_pts=False, video_process=old_worker,
        )
        self.assertFalse(ingest.video_failed.is_set())

    def test_live_cache_rebuffer_is_two_seconds_only(self):
        # Live option is restricted to the active player, not global mpv.conf.
        import inspect
        source = inspect.getsource(MpvOutput.__init__)
        self.assertIn('["--cache-pause-wait=2"] if ingest.source.live', source)


if __name__ == "__main__":
    unittest.main()
