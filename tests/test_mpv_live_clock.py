"""Regression tests for live HLS playlist timestamp jumps."""

import unittest
from types import SimpleNamespace

from auto_translate.output.mpv import MpvOutput


class LiveMpvClockTests(unittest.TestCase):
    def make_output(self, *, live=True):
        # Avoid launching MPV; exercise the same helpers used by its scheduler.
        output = MpvOutput.__new__(MpvOutput)
        output.ingest = SimpleNamespace(
            source=SimpleNamespace(live=live),
        )
        output._clock_source_anchor = 0.0
        output._mpv_anchor = None
        output._now = None
        output._last_live_mpv_pos = None
        output._last_live_sample_at = None
        output._uses_original_timeline = False
        return output

    def sample(self, output, mpv_pos, sampled_at):
        output._sample_live_clock(mpv_pos, sampled_at)
        # Mirrors the scheduler's first-sample anchor assignment.
        if output._mpv_anchor is None:
            output._mpv_anchor = float(mpv_pos)
        output._now = output._source_time_from_mpv(float(mpv_pos))
        return output._now

    def test_live_starts_at_large_hls_media_timestamp(self):
        output = self.make_output()
        self.assertAlmostEqual(self.sample(output, 15260.0, 100.0), 0.0)
        self.assertAlmostEqual(self.sample(output, 15261.0, 101.0), 1.0)

    def test_live_startup_zero_to_multi_hour_timestamp_reanchors(self):
        output = self.make_output()
        self.assertAlmostEqual(self.sample(output, 0.0, 100.0), 0.0)
        self.assertAlmostEqual(self.sample(output, 15260.0, 100.1), 0.0)
        self.assertAlmostEqual(self.sample(output, 15261.0, 101.1), 1.0)

    def test_normal_progress_after_long_scheduler_gap_does_not_reanchor(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        self.assertAlmostEqual(self.sample(output, 15290.0, 130.0), 30.0)
        self.assertAlmostEqual(output._mpv_anchor, 15260.0)

    def test_live_hls_timestamp_reset_preserves_subtitle_clock(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        self.sample(output, 15269.0, 109.0)
        self.assertAlmostEqual(self.sample(output, 0.0, 109.1), 9.0)
        self.assertAlmostEqual(self.sample(output, 1.0, 110.1), 10.0)

    def test_original_vod_timeline_is_not_rebased(self):
        output = self.make_output(live=False)
        output._uses_original_timeline = True
        output._mpv_anchor = 15260.0
        self.assertAlmostEqual(output._source_time_from_mpv(15261.0), 15261.0)


if __name__ == "__main__":
    unittest.main()
