"""Observed MPV telemetry and owned-process diagnostics without a media player."""
from collections import deque
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvOutput
from auto_translate.runtime_diagnostics import RuntimeDiagnostics
from auto_translate.types import Subtitle
from auto_translate.windows_subprocess import owned_process_status


def player(*, live=False, now=10.0, generation=2):
    output = object.__new__(MpvOutput)
    output.cfg = MpvConfig()
    output.ingest = SimpleNamespace(generation=generation,
                                   source=SimpleNamespace(live=live, duration=None if live else 600.0))
    output.ipc = SimpleNamespace(_properties={}, get=Mock(side_effect=AssertionError("unexpected IPC read")))
    output._last_gui_status_at = output._last_diagnostics_at = 0.0
    output._now = now
    output._session_start = 0.0
    output._lines = []
    output._shown = ""
    output._lock = threading.Lock()
    output._subtitle_ready = threading.Event()
    output._player_command = Mock(return_value=True)
    return output


class PlaybackObservationTests(unittest.TestCase):
    def observe(self, output, tick=10):
        with patch("auto_translate.output.mpv.time.monotonic", return_value=tick), patch(
                "auto_translate.output.mpv.update_metrics") as update, patch(
                "auto_translate.output.mpv.emit_event") as emit:
            output._emit_gui_status(False)
        return update, emit

    def test_observed_resolution_decoder_and_hwdec_use_only_cached_values(self):
        output = player()
        output.cfg.hwdec = "auto-safe"
        output.ipc._properties = {"video-params": {"w": 1280, "h": 720},
                                  "video-codec": "h264", "hwdec-current": "d3d11va"}
        update, emit = self.observe(output)
        values = update.call_args.kwargs["playback"]
        self.assertEqual(values["resolution"], "1280x720")
        self.assertEqual(values["decoder"], "h264")
        self.assertEqual(values["hwdec"], "d3d11va")
        self.assertTrue(values["hardware_acceleration"])
        output.ipc.get.assert_not_called()
        emit.assert_called_once_with("status", live=False, position=10.0, duration=600.0, paused=False)

    def test_requested_hardware_decoding_is_not_a_confirmed_observation(self):
        output = player(live=True)
        output.cfg.hwdec = "auto-safe"
        update, _ = self.observe(output)
        values = update.call_args.kwargs["playback"]
        for name in ("resolution", "decoder", "hwdec", "hardware_acceleration"):
            self.assertIsNone(values[name])
        output.ipc.get.assert_not_called()

    def test_unknown_vod_duration_keeps_only_the_existing_status_duration_read(self):
        output = player()
        output.ingest.source.duration = None
        output.ipc.get = Mock(return_value=620.)
        _, emit = self.observe(output)
        output.ipc.get.assert_called_once_with("duration")
        self.assertEqual(emit.call_args.kwargs["duration"], 620.)

    def test_observed_software_decoding_is_reported_without_changing_policy(self):
        output = player()
        output.ipc._properties = {"hwdec-current": "no"}
        update, _ = self.observe(output)
        self.assertFalse(update.call_args.kwargs["playback"]["hardware_acceleration"])
        self.assertEqual(output.cfg.hwdec, "auto-safe")
        output.ipc.get.assert_not_called()

    def test_monitoring_is_throttled_without_additional_player_requests(self):
        output = player()
        with patch("auto_translate.output.mpv.time.monotonic", side_effect=[10., 10.1, 10.6, 12.]), patch(
                "auto_translate.output.mpv.update_metrics") as update, patch(
                "auto_translate.output.mpv.emit_event") as emit:
            for _ in range(4):
                output._emit_gui_status(False)
        self.assertEqual(update.call_count, 2)
        self.assertEqual(emit.call_count, 3)
        output.ipc.get.assert_not_called()


class SubtitleDiagnosticCountersTests(unittest.TestCase):
    def test_pending_expired_and_stale_generation_counts_match_display_queue(self):
        output = player()
        diagnostics = RuntimeDiagnostics(SimpleNamespace(enabled=False))
        with patch("auto_translate.runtime_diagnostics._ACTIVE", diagnostics):
            output.show(Subtitle(9, 12, "今", "Now.", generation=2))
            output.show(Subtitle(15, 17, "次", "Next.", generation=2))
            output.show(Subtitle(1, 2, "古い", "Expired.", generation=2))
            output.show(Subtitle(16, 18, "以前", "Old session.", generation=1))
            counts = diagnostics.snapshot()["counters"]
            self.assertEqual(counts["pending_subtitles"], 1)
            self.assertEqual(counts["expired"], 1)
            self.assertEqual(counts["dropped"], 1)
            self.assertEqual(len(output._lines), 2)
            self.assertTrue(all(line.sub.generation == 2 for line in output._lines))
            output._render(16)
            counts = diagnostics.snapshot()["counters"]
            self.assertEqual(counts["pending_subtitles"], 0)
            self.assertEqual(counts["expired"], 1)
            self.assertEqual(counts["dropped"], 1)

    def test_live_expiry_budget_drops_very_late_cues_and_counts_only_actual_drops(self):
        output = player(live=True, now=12)
        diagnostics = RuntimeDiagnostics(SimpleNamespace(enabled=False))
        with patch("auto_translate.runtime_diagnostics._ACTIVE", diagnostics):
            output.show(Subtitle(7, 10, "遅れ", "Late but usable.", generation=2))
            output.show(Subtitle(1, 2, "古い", "Too late.", generation=2))
            self.assertEqual(len(output._lines), 1)
            self.assertEqual(output._lines[0].show_from, 12)
            self.assertEqual(diagnostics.snapshot()["counters"]["expired"], 1)
            self.assertEqual(diagnostics.snapshot()["counters"]["pending_subtitles"], 0)


class OwnedProcessObservationTests(unittest.TestCase):
    def test_reports_only_registered_children_without_external_enumeration(self):
        active = Mock(pid=1201, returncode=None)
        active.poll.return_value = None
        ended = Mock(pid=1202, returncode=7)
        ended.poll.return_value = 7
        unrelated = Mock(pid=9901)
        registry = deque([("audiocpp_server.exe", active), ("ffmpeg.exe", ended)], maxlen=64)
        with patch("auto_translate.windows_subprocess._owned", registry), patch(
                "auto_translate.windows_subprocess.subprocess.Popen", side_effect=AssertionError("no child spawn")), patch(
                "auto_translate.windows_subprocess.subprocess.check_output", side_effect=AssertionError("no process enumeration")):
            observed = owned_process_status()
        self.assertEqual(observed, [
            {"name": "audiocpp_server.exe", "pid": 1201, "running": True, "exit_code": None},
            {"name": "ffmpeg.exe", "pid": 1202, "running": False, "exit_code": 7},
        ])
        active.poll.assert_called_once_with()
        ended.poll.assert_called_once_with()
        unrelated.poll.assert_not_called()

    def test_empty_owned_registry_reports_no_workers(self):
        with patch("auto_translate.windows_subprocess._owned", deque(maxlen=64)):
            self.assertEqual(owned_process_status(), [])


if __name__ == "__main__":
    unittest.main()
