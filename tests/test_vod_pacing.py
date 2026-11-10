"""Offline regression tests for MPV VOD audio read-ahead pacing.

No GPU, actual model, FFmpeg, mpv process, or network access is required.
"""
from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from auto_translate.ingest import Ingest
from auto_translate.pipeline import _expected_vod_idle
from auto_translate.output.mpv import BOUNDED_DEMUX_CACHE_OPTIONS, MpvOutput
from auto_translate.types import SAMPLE_RATE


class VodPacingTests(unittest.TestCase):
    @staticmethod
    def ingest(*, live=False, video_out=None, start_s=0.0):
        source = SimpleNamespace(live=live, start_s=start_s)
        ingest = Ingest.__new__(Ingest)
        ingest.source = source
        ingest.video_out = video_out
        ingest._proc_lock = threading.Lock()
        ingest._vod_max_audio_lead_s = 0.0
        ingest._playback_position_s = start_s
        ingest._vod_pacing_waiting_generation = None
        ingest._generation = 0
        ingest._stopped = False
        return ingest

    def test_mp4_vod_gets_bounded_75_second_lead(self):
        ingest = self.ingest()
        ingest.enable_vod_playback_pacing(75)
        self.assertEqual(ingest._vod_max_audio_lead_s, 75.0)
        self.assertTrue(ingest._wait_for_vod_lead(75 * SAMPLE_RATE, 0))
        ingest.set_playback_position(15.0)
        self.assertTrue(ingest._wait_for_vod_lead(90 * SAMPLE_RATE, 0))

    def test_paused_video_halts_background_pcm_and_resumes_with_playback(self):
        ingest = self.ingest()
        ingest.enable_vod_playback_pacing(75.0)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                ingest._wait_for_vod_lead(80 * SAMPLE_RATE, 0)),
            daemon=True,
        )
        worker.start()
        time.sleep(.25)
        self.assertTrue(worker.is_alive())
        ingest.set_playback_position(5.5)
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [True])

    def test_inflight_seek_cancels_blocked_audio_reader(self):
        ingest = self.ingest()
        ingest.enable_vod_playback_pacing(75.0)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                ingest._wait_for_vod_lead(100 * SAMPLE_RATE, 0)),
            daemon=True,
        )
        worker.start()
        time.sleep(.25)
        self.assertTrue(worker.is_alive())
        with ingest._proc_lock:
            ingest._generation = 1
            ingest._playback_position_s = 5 * 3600.0
        worker.join(timeout=1.0)
        self.assertEqual(result, [False])
        self.assertTrue(ingest._wait_for_vod_lead(
            (5 * 3600 + 10) * SAMPLE_RATE, 1,
        ))

    def test_stop_cancels_blocked_audio_reader(self):
        ingest = self.ingest()
        ingest.enable_vod_playback_pacing(75)
        with ingest._proc_lock:
            ingest._stopped = True
        self.assertFalse(ingest._wait_for_vod_lead(
            5 * 3600 * SAMPLE_RATE, 0,
        ))

    def test_live_and_fmp4_are_never_paced(self):
        for live, video_out in ((True, None), (False, "index.m3u8")):
            ingest = self.ingest(live=live, video_out=video_out)
            ingest.enable_vod_playback_pacing(75)
            self.assertEqual(ingest._vod_max_audio_lead_s, 0.0)
            self.assertTrue(ingest._wait_for_vod_lead(
                7 * 3600 * SAMPLE_RATE, 0,
            ))

    def test_disable_lead_limit_preserves_full_fast_ingest(self):
        ingest = self.ingest()
        ingest.enable_vod_playback_pacing(0)
        self.assertTrue(ingest._wait_for_vod_lead(
            7 * 3600 * SAMPLE_RATE, 0,
        ))

    def test_manual_pause_is_not_a_pipeline_stall_even_with_queued_audio(self):
        ingest = self.ingest()
        # The user's 7-hour VOD failure involved one queued speaker item
        # after a prolonged manual pause; no new ASR output is expected.
        mpv = MpvOutput.__new__(MpvOutput)
        mpv._manual_pause = True
        self.assertTrue(_expected_vod_idle(ingest, [mpv]))
        mpv._manual_pause = False
        self.assertFalse(_expected_vod_idle(ingest, [mpv]))

    def test_audio_lead_pacing_suspends_watchdog_then_resumes_normally(self):
        ingest = self.ingest(start_s=20886.6)
        ingest.enable_vod_playback_pacing(75)
        target = round(20962.0 * SAMPLE_RATE)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                ingest._wait_for_vod_lead(target, 0)),
            daemon=True,
        )
        worker.start()
        time.sleep(.25)
        self.assertTrue(worker.is_alive())
        self.assertTrue(ingest.is_vod_audio_lead_limited())
        self.assertTrue(_expected_vod_idle(ingest, []))
        # The user unpauses and MPV reaches the next audio chunk.
        ingest.set_playback_position(20888.0)
        worker.join(timeout=1)
        self.assertEqual(result, [True])
        self.assertFalse(ingest.is_vod_audio_lead_limited())
        self.assertFalse(_expected_vod_idle(ingest, []))

    def test_seek_clears_parked_audio_generation_and_watchdog_exemption(self):
        ingest = self.ingest(start_s=20886.6)
        ingest.enable_vod_playback_pacing(75)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(ingest._wait_for_vod_lead(
                20965 * SAMPLE_RATE, 0,
            )),
            daemon=True,
        )
        worker.start()
        time.sleep(.25)
        self.assertTrue(ingest.is_vod_audio_lead_limited())
        with ingest._proc_lock:
            ingest._generation = 1
            ingest._vod_pacing_waiting_generation = None
            ingest._playback_position_s = 5 * 3600
        worker.join(timeout=1)
        self.assertEqual(result, [False])
        self.assertFalse(ingest.is_vod_audio_lead_limited())

    def test_watchdog_exemption_never_applies_to_live_streams(self):
        ingest = self.ingest(live=True)
        ingest._vod_max_audio_lead_s = 75.0
        ingest._vod_pacing_waiting_generation = 0
        mpv = MpvOutput.__new__(MpvOutput)
        mpv._manual_pause = True
        self.assertFalse(_expected_vod_idle(ingest, [mpv]))

    def test_unexpected_work_queue_still_uses_normal_watchdog(self):
        ingest = self.ingest()
        mpv = MpvOutput.__new__(MpvOutput)
        self.assertFalse(_expected_vod_idle(ingest, [mpv]))
        self.assertFalse(mpv.is_manually_paused())

    def test_mpv_network_cache_has_actual_seconds_limit(self):
        self.assertIn("--cache=yes", BOUNDED_DEMUX_CACHE_OPTIONS)
        self.assertIn("--cache-secs=30", BOUNDED_DEMUX_CACHE_OPTIONS)
        self.assertIn("--demuxer-hysteresis-secs=10", BOUNDED_DEMUX_CACHE_OPTIONS)


if __name__ == "__main__":
    unittest.main()
