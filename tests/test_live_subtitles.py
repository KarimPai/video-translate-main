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


class LiveSubtitleSyncTests(unittest.TestCase):
    """Regression tests for live subtitle sync: display time vs speech time."""

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

    def test_on_time_subtitle_shows_at_speech_start_not_padded_start(self):
        # speech_start_s excludes VAD pre-roll; subtitle must wait for it.
        p = self.player(now=2.0)
        p.show(Subtitle(0.0, 4.0, "待って", "Wait!", speech_start_s=2.0))
        self.assertEqual(p._lines[0].show_from, 2.0)
        self.assertEqual(p._lines[0].show_until, 4.0)

    def test_subtitle_without_speech_start_uses_pcm_start(self):
        # No VAD speech data: falls back to padded PCM start.
        p = self.player(now=2.0)
        p.show(Subtitle(0.0, 4.0, "待って", "Wait!"))
        self.assertEqual(p._lines[0].show_from, 0.0)

    def test_live_late_subtitle_records_display_lag(self):
        import io
        from auto_translate.runtime_diagnostics import record_event
        captured = []
        orig = record_event

        def capture(event, **data):
            captured.append((event, data))
            orig(event, **data)

        import auto_translate.output.mpv as mpv_mod
        mpv_mod.record_event = capture
        try:
            p = self.player(now=9.0)
            p.show(Subtitle(2.0, 7.0, "待って", "Wait!", speech_start_s=2.0))
            lag_events = [d for e, d in captured if e == "subtitle_display_lag"]
            self.assertTrue(lag_events, "expected a subtitle_display_lag event")
            lag = lag_events[-1]
            self.assertAlmostEqual(lag["lag_s"], 9.0 - 2.0, places=2)
            self.assertTrue(lag["live"])
        finally:
            mpv_mod.record_event = orig

    def test_live_subtitle_does_not_show_before_speech_start(self):
        # Subtitle is enqueued before playback reaches speech_start;
        # it should be held until now >= show_from.
        p = self.player(now=1.0)
        p.show(Subtitle(2.0, 4.0, "待って", "Wait!", speech_start_s=2.0))
        self.assertEqual(len(p._lines), 1)
        # show_from should equal speech_start, not PCM start.
        self.assertEqual(p._lines[0].show_from, 2.0)
        # Not active yet because now (1.0) < show_from (2.0).
        active = [ln for ln in p._lines if ln.show_from <= p._now < ln.show_until]
        self.assertEqual(len(active), 0)


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
