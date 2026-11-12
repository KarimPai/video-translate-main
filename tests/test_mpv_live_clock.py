"""Exercise the scheduler's real observation path without a display or GPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate.live_clock import Discontinuity, LiveMediaClock
from auto_translate.output.mpv import MpvOutput, _live_buffer_seconds


class LiveMpvClockTests(unittest.TestCase):
    def setUp(self):
        self.player = MpvOutput.__new__(MpvOutput)
        self.player._live_clock = LiveMediaClock(3)
        self.player._live_diagnostic_at = float("inf")
        self.player.ingest = SimpleNamespace(live_audio_span=(15260, 15360))
        self.player.ipc = Mock()

    def sample(self, pos, at, audio=None):
        self.player.ipc.get.return_value = pos if audio is None else audio
        return self.player._sample_live_clock(pos, at)

    def test_different_reader_opens_and_placeholder(self):
        self.assertIsNone(self.sample(0, 1))
        self.assertEqual(self.sample(15267, 3), 15267)
        self.assertEqual(self.sample(15268, 4), 15268)

    def test_zero_origin_and_unreliable_observations(self):
        self.player.ingest.live_audio_span = (0, 3)
        self.assertIsNone(self.sample(0, 1, 15260))
        self.assertIsNone(self.sample(float("nan"), 2))
        self.assertEqual(self.sample(0, 3), 0)

    def test_startup_reserve_uses_available_media(self):
        from auto_translate.config import MpvConfig
        target = MpvConfig().delay_s
        self.assertEqual(target, 20)
        for pos in (0, None, 808):
            self.assertLess(_live_buffer_seconds((808, 812.5), pos), target)
            self.assertEqual(_live_buffer_seconds((808, 828), pos), target)
        self.assertEqual(_live_buffer_seconds((808, 830), 825), 5)
        self.assertEqual(_live_buffer_seconds(None, 808), 0)

    def test_pause_and_long_poll_gap(self):
        self.sample(15261, 1)
        self.assertEqual(self.sample(15261, 400), 15261)
        self.assertEqual(self.sample(15321, 460), 15321)

    def test_reset_or_jump_requires_a_new_epoch(self):
        self.sample(15261, 1)
        for target in (0, 15280):
            with self.assertRaises(Discontinuity):
                self.sample(target, 2)

    def test_six_hour_coordinate(self):
        self.player.ingest.live_audio_span = (15260, 15260 + 21601)
        for delta in range(0, 21601, 60):
            self.assertEqual(self.sample(15260 + delta, delta), 15260 + delta)

    def test_shared_recovery_clears_and_reloads_and_stops_after_three(self):
        import threading
        player = self.player
        player.stop = threading.Event()
        player._lock = threading.Lock()
        player._lines = [object()]
        player._shown = "stale"
        player._video_retries = 0
        player._now = 15260
        player.ingest = Mock()
        player.ingest.source.start_s = 0
        player.ingest.source.live = True
        player.reset_buffer = Mock()
        for _ in range(3):
            player._direct_video_loaded = True
            self.assertTrue(player._recover_video_stream())
            self.assertFalse(player._direct_video_loaded)
        self.assertFalse(player._recover_video_stream())
        self.assertTrue(player.stop.is_set())
        self.assertEqual(player.ingest.request_restart.call_count, 3)
        player.ingest.request_restart.assert_called_with(0)
        self.assertEqual(player._lines, [])
        self.assertEqual(player._shown, "")
