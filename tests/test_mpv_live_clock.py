"""Measured-media clock mapping, never first-observation/arrival anchoring."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvOutput
from auto_translate.types import Subtitle


class LiveMpvClockTests(unittest.TestCase):
    def make_output(self, *, live=True, media_origin=15260.0, source_origin=0.0):
        output = MpvOutput.__new__(MpvOutput)
        output.ingest = SimpleNamespace(
            source=SimpleNamespace(live=live), generation=0,
            live_clock_anchor=lambda: (0, source_origin, media_origin),
        )
        output._clock_source_anchor = source_origin
        output._clock_generation = None
        output._live_clock_discontinuity = False
        output._mpv_anchor = None
        output._now = None
        output._last_live_mpv_pos = None
        output._last_live_sample_at = None
        output._uses_original_timeline = False
        output._session_start = source_origin
        output.cfg = MpvConfig()
        output._lines = []
        output._shown = ""
        output._lock = threading.Lock()
        output._subtitle_ready = threading.Event()
        output._player_command = Mock(return_value=True)
        return output

    def sample(self, output, mpv_pos, sampled_at):
        if output._sample_live_clock(mpv_pos, sampled_at):
            output._now = output._source_time_from_mpv(mpv_pos)
        return output._now

    def test_first_observation_is_not_pcm_origin(self):
        output = self.make_output()
        self.assertAlmostEqual(self.sample(output, 15267.0, 100.0), 7.0)
        self.assertAlmostEqual(output._mpv_anchor, 15260.0)

    def test_live_startup_zero_waits_for_measured_media_timeline(self):
        output = self.make_output()
        self.assertIsNone(self.sample(output, 0.0, 100.0))
        self.assertIsNone(output._mpv_anchor)
        self.assertAlmostEqual(self.sample(output, 15260.0, 100.1), 0.0)
        self.assertAlmostEqual(self.sample(output, 15261.0, 101.1), 1.0)
        self.assertFalse(output._live_clock_discontinuity)

    def test_actual_zero_media_origin_is_valid(self):
        output = self.make_output(media_origin=0.0)
        self.assertEqual(self.sample(output, 0.0, 100.0), 0.0)
        self.assertEqual(self.sample(output, 1.0, 101.0), 1.0)

    def test_unavailable_or_nonfinite_timestamp_cannot_anchor(self):
        output = self.make_output()
        output.ingest.live_clock_anchor = lambda: None
        self.assertIsNone(self.sample(output, 15260, 1))
        output.ingest.live_clock_anchor = lambda: (0, 0, 15260)
        self.assertIsNone(self.sample(output, float("nan"), 2))
        self.assertIsNone(self.sample(output, float("inf"), 3))

    def test_normal_progress_after_long_scheduler_gap_does_not_reanchor(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        self.assertAlmostEqual(self.sample(output, 15290.0, 130.0), 30.0)
        self.assertAlmostEqual(output._mpv_anchor, 15260.0)

    def test_pts_reset_invalidates_alignment_instead_of_preserving_old_cues(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        self.sample(output, 15269.0, 109.0)
        self.assertIsNone(self.sample(output, 0.0, 109.1))
        self.assertTrue(output._live_clock_discontinuity)
        self.assertIsNone(output._source_time_from_mpv(1.0))

    def test_forward_jump_and_changed_generation_invalidate(self):
        for new_anchor, new_position in [((0, 0.0, 15260.0), 15263.0),
                                         ((1, 20.0, 800.0), 15260.1)]:
            output = self.make_output()
            self.sample(output, 15260.0, 100.0)
            output.ingest.live_clock_anchor = lambda: new_anchor
            self.assertIsNone(self.sample(output, new_position, 100.1))
            self.assertTrue(output._live_clock_discontinuity)

    def test_stall_and_pause_do_not_integrate_wall_time_into_media(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        self.assertEqual(self.sample(output, 15260.0, 150.0), 0.0)
        self.assertEqual(self.sample(output, 15261.0, 151.0), 1.0)

    def test_lost_audio_timestamp_evidence_invalidates_existing_alignment(self):
        output = self.make_output()
        self.sample(output, 15260.0, 100.0)
        output.ingest.live_clock_anchor = lambda: None
        self.assertIsNone(self.sample(output, 15260.1, 100.1))
        self.assertTrue(output._live_clock_discontinuity)
        self.assertIsNone(output._mpv_anchor)

    def test_live_delay_and_variable_buffering_do_not_change_shared_offset(self):
        for delay in (0, 5, 20):
            output = self.make_output()
            self.assertEqual(self.sample(output, 15260, 100 + delay), 0)
            self.assertEqual(self.sample(output, 15260, 110 + delay), 0)
            self.assertEqual(self.sample(output, 15261, 111 + delay), 1)
            self.assertEqual(output._mpv_anchor, 15260)

    def test_long_running_clock_has_no_integrated_drift(self):
        output = self.make_output(media_origin=560000.0)
        self.sample(output, 560000.0, 100.0)
        for seconds in range(1, 10 * 3600, 137):
            # Arrival scheduling drifts, the shared media clock does not.
            self.assertAlmostEqual(self.sample(output, 560000.0 + seconds,
                                              100.0 + seconds * 1.00001), seconds)
        self.assertEqual(output._mpv_anchor, 560000.0)

    def test_fast_translation_waits_for_speech_not_padded_start(self):
        output = self.make_output(media_origin=5.0)
        cue = Subtitle(0, 4, "こんにちは", "Hello.", speech_start_s=3.0)
        output.show(cue)  # Finishes before MPV has reached PCM sample zero.
        shown = []
        for tick in range(51):
            media = 2.0 + tick / 10  # MPV starts 3s before PCM's first PTS.
            now = self.sample(output, media, 100 + tick / 10)
            if now is not None:
                output._render(now)
                if output._shown:
                    shown.append(media)
        # Need the interval containing true onset at PTS 8.
        for tick in range(51, 71):
            media = 2.0 + tick / 10
            now = self.sample(output, media, 100 + tick / 10)
            output._render(now)
            if output._shown:
                shown.append(media)
        self.assertTrue(shown)
        self.assertAlmostEqual(min(shown) - 8.0, 0.0, places=6)
        # Former first-observation anchor would show it at PTS 2 + 3 = 5:
        # synthetic before/after error is -3.0s / 0.0s at this poll phase.

    def test_old_generation_is_not_rendered_and_live_cache_is_ignored(self):
        output = self.make_output()
        output.show(Subtitle(0, 4, "前", "Old.", generation=0))
        output.ingest.generation = 1
        output._subtitle_cache = Mock()
        output._render(2)
        self.assertEqual(output._shown, "")
        output._subtitle_cache.active_at.assert_not_called()

    def test_final_renderer_rejects_japanese_and_cleans_dangling_quotes(self):
        from auto_translate.output.mpv import _ass_many
        output = self.make_output()
        output.show(Subtitle(0, 4, "こんにちは", "Hello. こんにちは"))
        self.assertEqual(output._lines, [])
        # The final renderer defends direct/cache callers as well.
        self.assertEqual(_ass_many([Subtitle(0, 4, "はい", "はい")], output.cfg), "")
        rendered = _ass_many([Subtitle(0, 4, "はい", '"Don\'t worry!')], output.cfg)
        self.assertIn("Don't worry!", rendered)
        self.assertNotIn('"', rendered)

    def test_recovery_maps_new_media_origin_to_new_source_position(self):
        output = self.make_output(media_origin=45.0, source_origin=120.0)
        output.ingest.generation = 4
        output.ingest.live_clock_anchor = lambda: (4, 120.0, 45.0)
        self.assertEqual(self.sample(output, 46.25, 1000), 121.25)

    def test_original_vod_timeline_is_not_rebased(self):
        output = self.make_output(live=False)
        output._uses_original_timeline = True
        output._mpv_anchor = 15260.0
        self.assertAlmostEqual(output._source_time_from_mpv(15261.0), 15261.0)

    def test_new_file_invalidates_previous_observed_playback_position(self):
        import io
        import queue
        from auto_translate.output.mpv import MpvIpc
        ipc = object.__new__(MpvIpc)
        ipc._properties = {"time-pos": 42.0, "eof-reached": True}
        ipc._f = io.BytesIO(b'{"event":"start-file"}\n')
        ipc.events = queue.Queue()
        ipc._responses = queue.Queue()
        ipc._read_messages()
        self.assertIsNone(ipc._properties["time-pos"])
        self.assertIsNone(ipc._properties["eof-reached"])
        self.assertEqual(ipc.events.get_nowait()["event"], "start-file")


if __name__ == "__main__":
    unittest.main()
