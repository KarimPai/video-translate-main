"""Offline tests for opt-in MPV RTX VSR; NVIDIA/Windows requires live testing."""
import unittest
from auto_translate.output.rtx_vsr import (
    VSR_FILTER, VsrPerformanceGuard, filter_present, launch_options,
    source_eligible,
)


class RtxVsrTests(unittest.TestCase):
    def test_windows_safe_hwdec_opt_in(self):
        self.assertEqual(launch_options(False, "auto-safe", "win32"), [])
        self.assertEqual(launch_options(True, "no", "win32"), [])
        self.assertEqual(launch_options(True, "auto-safe", "linux"), [])
        self.assertEqual(launch_options(True, "auto-safe", "win32"), [
            "--vo=gpu-next", "--gpu-api=d3d11", "--gpu-context=d3d11",
        ])

    def test_source_guard_and_labeled_filter(self):
        self.assertTrue(source_eligible({"h": 480}))
        self.assertTrue(source_eligible({"h": 360}))
        self.assertFalse(source_eligible({"h": 720}))
        self.assertFalse(source_eligible({"h": "broken"}))
        self.assertFalse(source_eligible(None))
        self.assertIn("scale=3:scaling-mode=nvidia", VSR_FILTER)
        self.assertTrue(filter_present([{"label": "haas-rtx-vsr"}]))
        self.assertFalse(filter_present([{"label": "unrelated"}]))

    def test_dropped_frames_trigger_then_pauses_reset(self):
        guard = VsrPerformanceGuard()
        values = dict(hwdec="d3d11va", paused=False, caching=False)
        self.assertIsNone(guard.sample(now=0, frame_drops=0, delayed_frames=0, **values))
        self.assertIsNone(guard.sample(now=30, frame_drops=2, delayed_frames=3, **values))
        self.assertIn("15 dropped", guard.sample(
            now=60, frame_drops=17, delayed_frames=4, **values))
        self.assertIsNone(guard.sample(now=90, frame_drops=25, delayed_frames=35,
                                       hwdec="d3d11va", paused=True, caching=False))
        self.assertIsNone(guard.sample(now=120, frame_drops=25, delayed_frames=35, **values))

    def test_delayed_and_sw_decode_fallback(self):
        guard = VsrPerformanceGuard()
        values = dict(hwdec="d3d11va", paused=False, caching=False)
        guard.sample(now=0, frame_drops=0, delayed_frames=0, **values)
        self.assertIn("30 delayed", guard.sample(
            now=30, frame_drops=2, delayed_frames=30, **values))
        self.assertIn("software", guard.sample(
            now=60, frame_drops=0, delayed_frames=0,
            hwdec="no", paused=False, caching=False))

    def test_reset_and_missing_counters_are_safe(self):
        guard = VsrPerformanceGuard()
        values = dict(hwdec=None, paused=False, caching=False)
        self.assertIsNone(guard.sample(now=0, frame_drops=None, delayed_frames=0, **values))
        self.assertIsNone(guard.sample(now=30, frame_drops=100, delayed_frames=60, **values))
        self.assertIsNone(guard.sample(now=60, frame_drops=0, delayed_frames=0, **values))


if __name__ == "__main__":
    unittest.main()
