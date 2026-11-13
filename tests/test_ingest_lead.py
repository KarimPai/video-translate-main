"""Offline regression for MPV VOD audio lead pacing and pause-safe health."""

import threading
import queue
import time
import unittest
from unittest.mock import Mock, patch
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


class YoutubeAudioRecoveryTests(unittest.TestCase):
    @staticmethod
    def failed_vod_ingest():
        ing = dummy_ingest()
        ing.source.location = "https://www.youtube.com/watch?v=example"
        ing._vod_audio_url = "https://googlevideo.invalid/signed"
        ing._audio_client_fallback = False
        ing._restart_at = None
        return ing

    def test_initial_vod_audio_failure_refreshes_even_without_prior_cache(self):
        ing = self.failed_vod_ingest()
        ing._vod_audio_url = None  # freshly resolved inside first _read_session
        observed = []

        def read(_q):
            observed.append((ing._vod_audio_url, ing._audio_client_fallback))
            return len(observed) < 3

        outputs = queue.Queue()
        with patch.object(ing, "_read_session", side_effect=read), patch(
                "auto_translate.ingest.time.sleep", return_value=None):
            ing.run(outputs)

        self.assertEqual(len(observed), 3)
        self.assertEqual([state[1] for state in observed], [False, False, True])
        self.assertIsNone(ing._vod_audio_url)
        self.assertIsNone(outputs.get_nowait())

    def test_persistent_vod_audio_error_fails_visibly_after_bounded_retries(self):
        ing = self.failed_vod_ingest()
        outputs = queue.Queue()
        with patch.object(ing, "_read_session", return_value=True) as reader, patch(
                "auto_translate.ingest.time.sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "audio inaccessible"):
                ing.run(outputs)

        self.assertEqual(reader.call_count, 3)
        self.assertIsNone(outputs.get_nowait())

    def test_mpv_drain_handles_closed_ipc_pipe(self):
        output = object.__new__(MpvOutput)
        output.stop = threading.Event()
        output._proc = Mock()
        output._proc.poll.return_value = None
        output.ipc = Mock()
        output.ipc.get.side_effect = BrokenPipeError(232, "The pipe is being closed")
        output.drain()
        output.ipc.get.assert_called_once_with("eof-reached")


if __name__ == "__main__":
    unittest.main()
