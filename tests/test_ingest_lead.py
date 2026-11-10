"""Offline regression for MPV VOD audio lead pacing and pause-safe health."""

import threading
import time
import unittest
from types import SimpleNamespace

from auto_translate.ingest import Ingest
from auto_translate.output.mpv import MpvOutput
from auto_translate.types import SAMPLE_RATE


def dummy_ingest(*, live=False):
    ing = object.__new__(Ingest)
    ing.source = SimpleNamespace(live=live, start_s=0.0)
    ing.video_out = None
    ing._proc_lock = threading.Lock()
    ing._stopped = False
    ing._generation = 0
    ing._vod_max_audio_lead_s = 0.0
    ing._playback_position_s = 0.0
    ing._vod_pacing_waiting_generation = None
    return ing


class PlaybackPacingTests(unittest.TestCase):
    def test_audio_lead_waits_then_resumes_as_video_advances(self):
        ing = dummy_ingest()
        ing.enable_vod_playback_pacing(75.0)
        self.assertTrue(ing._wait_for_vod_lead(70 * SAMPLE_RATE, 0))
        result = []
        t = threading.Thread(
            target=lambda: result.append(ing._wait_for_vod_lead(80 * SAMPLE_RATE, 0)),
            daemon=True,
        )
        t.start()
        time.sleep(0.20)
        self.assertTrue(ing.is_vod_audio_lead_limited())
        ing.set_playback_position(20.0)
        t.join(timeout=2.0)
        if t.is_alive():
            ing._stopped = True
            t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [True])
        self.assertFalse(ing.is_vod_audio_lead_limited())

    def test_seek_generation_unblocks_old_audio_reader(self):
        ing = dummy_ingest()
        ing.enable_vod_playback_pacing(75.0)
        result = []
        t = threading.Thread(
            target=lambda: result.append(ing._wait_for_vod_lead(90 * SAMPLE_RATE, 0)),
            daemon=True,
        )
        t.start()
        time.sleep(0.20)
        with ing._proc_lock:
            ing._generation = 1
            ing._playback_position_s = 90.0
        t.join(timeout=2.0)
        if t.is_alive():
            ing._stopped = True
            t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [False])

    def test_live_stream_does_not_enable_playback_pacing(self):
        ing = dummy_ingest(live=True)
        ing.enable_vod_playback_pacing(75.0)
        self.assertEqual(ing._vod_max_audio_lead_s, 0.0)

    def test_manual_pause_status_is_read_without_ipc_roundtrip(self):
        output = object.__new__(MpvOutput)
        output._manually_paused = True
        self.assertTrue(output.is_manually_paused())
        output._manually_paused = False
        self.assertFalse(output.is_manually_paused())


if __name__ == "__main__":
    unittest.main()
