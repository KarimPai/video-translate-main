"""CPU-only regression tests for live Malay MPV subtitle timing.

No MPV or network connection is needed. The playback object is constructed
without launching a subprocess so the tests exercise the real mapping/display.
"""
import threading
import unittest
from types import SimpleNamespace

from auto_translate.output.mpv import MpvOutput
from auto_translate.types import Subtitle


class MalayLiveSubtitleTests(unittest.TestCase):
    def player(self, *, live=True, now=0.0):
        mpv = object.__new__(MpvOutput)
        mpv.ingest = SimpleNamespace(source=SimpleNamespace(live=live))
        mpv._session_start = 0.0
        mpv._clock_source_anchor = 0.0
        mpv._mpv_anchor = 560000.0
        mpv._uses_original_timeline = True
        mpv._now = now
        mpv._lines = []
        mpv._lock = threading.Lock()
        mpv._subtitle_ready = threading.Event()
        return mpv

    def test_live_hls_timestamp_uses_relative_pcm_clock(self):
        mpv = self.player(live=True)
        self.assertEqual(mpv._source_time_from_mpv(560000.0), 0.0)
        self.assertEqual(mpv._source_time_from_mpv(560006.5), 6.5)

    def test_live_clock_without_anchor_starts_at_audio_origin(self):
        mpv = self.player(live=True)
        mpv._clock_source_anchor = 12.0
        mpv._mpv_anchor = None
        self.assertEqual(mpv._source_time_from_mpv(999999.0), 12.0)

    def test_vod_direct_playback_still_uses_absolute_timeline(self):
        mpv = self.player(live=False)
        self.assertEqual(mpv._source_time_from_mpv(560000.0), 560000.0)

    def test_live_caption_is_queued_at_original_time_if_on_time(self):
        mpv = self.player(live=True, now=3.0)
        sub = Subtitle(2.0, 4.0, "待って", "Tunggu!")
        mpv.show(sub)
        self.assertEqual(len(mpv._lines), 1)
        self.assertEqual(mpv._lines[0].show_from, 2.0)
        self.assertEqual(mpv._lines[0].show_until, 4.0)

    def test_slightly_late_live_caption_is_shown_instead_of_dropped(self):
        mpv = self.player(live=True, now=9.0)
        sub = Subtitle(2.0, 7.0, "待って", "Tunggu!")
        mpv.show(sub)
        self.assertEqual(len(mpv._lines), 1)
        self.assertEqual(mpv._lines[0].show_from, 9.0)
        self.assertEqual(mpv._lines[0].show_until, 11.0)
        self.assertTrue(mpv._subtitle_ready.is_set())

    def test_excessively_stale_live_caption_is_discarded(self):
        mpv = self.player(live=True, now=100.0)
        mpv.show(Subtitle(2.0, 7.0, "待って", "Tunggu!"))
        self.assertEqual(len(mpv._lines), 0)

    def test_vod_still_discards_late_subtitles(self):
        mpv = self.player(live=False, now=9.0)
        mpv.show(Subtitle(2.0, 7.0, "待って", "Tunggu!"))
        self.assertEqual(len(mpv._lines), 0)


if __name__ == "__main__":
    unittest.main()
