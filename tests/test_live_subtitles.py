"""CPU-only regression tests for English Haas/PLaMo live subtitle playback.

MPV process is not launched. Validates HLS vs PCM clock mapping and whether
translated cues survive ordinary live inference delay.
"""
import threading
import unittest
from types import SimpleNamespace

from auto_translate.output.mpv import MpvOutput
from auto_translate.types import Subtitle


class EnglishLiveSubtitleTests(unittest.TestCase):
    @staticmethod
    def player(*, live=True, now=0.0):
        player = object.__new__(MpvOutput)
        player.ingest = SimpleNamespace(generation=0, source=SimpleNamespace(live=live))
        player._session_start = 0.0
        player._clock_source_anchor = 0.0
        player._mpv_anchor = 560000.0
        player._uses_original_timeline = True
        player._now = now
        player._lines = []
        player._lock = threading.Lock()
        player._subtitle_ready = threading.Event()
        return player

    def test_live_hls_uses_shared_media_pts(self):
        p = self.player()
        self.assertEqual(p._source_time_from_mpv(560000.0), 560000.0)
        self.assertEqual(p._source_time_from_mpv(560006.5), 560006.5)

    def test_live_clock_mapping_independent_of_first_audio_arrival(self):
        p = self.player()
        p._clock_source_anchor = 12.0
        p._mpv_anchor = None
        self.assertEqual(p._source_time_from_mpv(999999.0), 999999.0)

    def test_vod_direct_playback_still_uses_absolute_timeline(self):
        p = self.player(live=False)
        self.assertEqual(p._source_time_from_mpv(560000.0), 560000.0)

    def test_live_caption_retains_original_timing_when_on_time(self):
        p = self.player(now=3.0)
        p.show(Subtitle(2.0, 4.0, "待って", "Wait!"))
        self.assertEqual(len(p._lines), 1)
        self.assertEqual(p._lines[0].show_from, 2.0)
        self.assertEqual(p._lines[0].show_until, 4.0)

    def test_late_live_caption_remains_visible(self):
        p = self.player(now=9.0)
        p.show(Subtitle(2.0, 7.0, "待って", "Wait!"))
        self.assertEqual(len(p._lines), 1)
        self.assertEqual(p._lines[0].show_from, 9.0)
        self.assertEqual(p._lines[0].show_until, 11.0)
        self.assertTrue(p._subtitle_ready.is_set())

    def test_stale_epoch_and_late_burst(self):
        player = self.player(now=9)
        player.ingest.generation = 1
        player.show(Subtitle(8, 10, "待って", "Stale!", generation=0))
        self.assertFalse(player._lines)
        for _ in range(3):
            player.show(Subtitle(5, 8, "待って", "Wait!", generation=1))
        self.assertEqual(len(player._lines), 1)
        self.assertEqual(player._late_dropped, 2)

    def test_manual_pause_bounds_future_cues(self):
        player = self.player(now=1)
        for i in range(1000):
            player.show(Subtitle(i + 2, i + 3, "待って", "Wait!"))
        self.assertLessEqual(len(player._lines), 256)

    def test_stale_live_caption_dropped(self):
        p = self.player(now=100.0)
        p.show(Subtitle(2.0, 7.0, "待って", "Wait!"))
        self.assertEqual(len(p._lines), 0)

    def test_vod_late_caption_still_dropped(self):
        p = self.player(live=False, now=9.0)
        p.show(Subtitle(2.0, 7.0, "待って", "Wait!"))
        self.assertEqual(len(p._lines), 0)


if __name__ == "__main__":
    unittest.main()
