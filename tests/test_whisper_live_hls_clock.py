"""Whisper live HLS timestamps must not replace elapsed subtitle time."""

import unittest
from types import SimpleNamespace

from auto_translate.output.mpv import MpvOutput


class WhisperLiveHlsClockTests(unittest.TestCase):
    def make_output(self, *, live, now, original_timeline=False):
        output = MpvOutput.__new__(MpvOutput)
        output.ingest = SimpleNamespace(source=SimpleNamespace(live=live))
        output._now = now
        output._clock_source_anchor = 0.0
        output._mpv_anchor = 15260.0
        output._uses_original_timeline = original_timeline
        return output

    def test_live_multi_hour_hls_timestamp_uses_elapsed_audio_clock(self):
        output = self.make_output(live=True, now=3.25)
        self.assertEqual(output._source_time_from_mpv(15263.25), 3.25)
        output._now = 4.25
        self.assertEqual(output._source_time_from_mpv(15264.25), 4.25)

    def test_live_discontinuous_hls_clock_does_not_jump(self):
        output = self.make_output(live=True, now=8.0)
        self.assertEqual(output._source_time_from_mpv(0.0), 8.0)
        self.assertEqual(output._source_time_from_mpv(17890.0), 8.0)

    def test_live_before_first_playback_tick_uses_session_start(self):
        output = self.make_output(live=True, now=None)
        self.assertEqual(output._source_time_from_mpv(15260.0), 0.0)

    def test_vod_original_timeline_is_unchanged(self):
        output = self.make_output(live=False, now=4.0, original_timeline=True)
        self.assertEqual(output._source_time_from_mpv(15263.25), 15263.25)

    def test_vod_fallback_timeline_is_unchanged(self):
        output = self.make_output(live=False, now=4.0)
        self.assertEqual(output._source_time_from_mpv(15263.25), 3.25)


if __name__ == "__main__":
    unittest.main()
