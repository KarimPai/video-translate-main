"""CPU-only regression tests for English Voxtral/PLaMo live subtitle playback.

MPV process is not launched. Validates HLS vs PCM clock mapping and whether
translated cues survive ordinary live inference delay.
"""
import threading
import unittest
from types import SimpleNamespace

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvOutput
from auto_translate.types import Subtitle


class EnglishLiveSubtitleTests(unittest.TestCase):
    @staticmethod
    def player(*, live=True, now=0.0):
        player = object.__new__(MpvOutput)
        player.cfg = MpvConfig()
        player.ingest = SimpleNamespace(source=SimpleNamespace(live=live))
        player._session_start = 0.0
        player._clock_source_anchor = 0.0
        player._mpv_anchor = 560000.0
        player._uses_original_timeline = True
        player._now = now
        player._lines = []
        player._lock = threading.Lock()
        player._subtitle_ready = threading.Event()
        return player

    def test_live_hls_timestamp_uses_relative_pcm_clock(self):
        p = self.player()
        self.assertEqual(p._source_time_from_mpv(560000.0), 0.0)
        self.assertEqual(p._source_time_from_mpv(560006.5), 6.5)

    def test_live_clock_without_measured_anchor_is_unavailable(self):
        p = self.player()
        p._clock_source_anchor = 12.0
        p._mpv_anchor = None
        self.assertIsNone(p._source_time_from_mpv(999999.0))

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


class MinimumDurationTests(unittest.TestCase):
    def test_presentation_extends_short_cues_without_changing_cached_end(self):
        from auto_translate.session_cache import SessionSubtitleCache
        player = EnglishLiveSubtitleTests.player()
        sub = Subtitle(1.0, 1.1, "え", "Huh?")
        player.show(sub)
        self.assertEqual(player._lines[0].show_until, 2.5)  # 1.5 s minimum
        cache = SessionSubtitleCache()
        cache.add(sub)
        self.assertEqual(cache.active_at(2.1, 1.2), [sub])
        self.assertEqual(cache.active_at(2.2, 1.2), [])
        self.assertEqual(sub.end_s, 1.1)
