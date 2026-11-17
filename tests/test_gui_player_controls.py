"""Qt-independent regression tests for embedded MPV UI protocol and seeks."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.gui_control import (
    EVENT_PREFIX, FULLSCREEN_MESSAGE, EXIT_FULLSCREEN_MESSAGE,
    format_event, parse_command, parse_event,
)
from auto_translate.output.mpv import MpvError, MpvOutput


class FakeIpc:
    def __init__(self):
        self.calls = []
        self.events = queue.Queue()

    def command(self, *args):
        self.calls.append(args)

    def get(self, name):
        if name in ("pause", "seeking"):
            return False
        return None


def mock_player(live=False, duration=600.0):
    player = MpvOutput.__new__(MpvOutput)
    player.cfg = SimpleNamespace(window_id=123)
    player.ingest = SimpleNamespace(source=SimpleNamespace(live=live, duration=duration))
    player.ipc = FakeIpc()
    player._now = 150.0
    player._uses_original_timeline = True
    player._pending_seek = None
    player._native_seek_at = None
    player._restart_audio_at = Mock()
    player._gui_commands = queue.Queue(maxsize=16)
    player._subtitle_ready = threading.Event()
    player._last_gui_status_at = 0.0
    return player


class GuiPlayerProtocolTests(unittest.TestCase):
    def test_valid_commands(self):
        self.assertEqual(parse_command('{"action":"quit"}'), ("quit", None))
        self.assertEqual(parse_command('{"action":"seek","value":3650.5}'), ("seek", 3650.5))
        self.assertEqual(parse_command('{"action":"seek_relative","value":-10}'), ("seek_relative", -10))
        self.assertEqual(parse_command('{"action":"pause"}'), ("pause", None))

    def test_reject_unsafe_commands(self):
        for data in (
            "", "not json", "{}", "[]",
            '{"action":"seek","value":"100"}',
            '{"action":"seek","value":-3}',
            '{"action":"seek","value":true}',
            '{"action":"seek","value":1e300}',
            '{"action":"seek_relative","value":100000}',
            '{"action":"set_property","value":"fullscreen"}',
        ):
            with self.subTest(data=data):
                self.assertIsNone(parse_command(data))

    def test_event_format_and_parse(self):
        data = parse_event(format_event("status", live=False, position=12.2, duration=210))
        self.assertEqual(data, {"event": "status", "live": False, "position": 12.2, "duration": 210})
        self.assertIsNone(parse_event("JP: こんにちは"))
        self.assertIsNone(parse_event(EVENT_PREFIX + "not-json"))

    def test_seek_waits_for_clock_and_confirms_before_audio_restart(self):
        player = mock_player()
        player._now = None
        player.queue_gui_command("seek", 340.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            player._handle_gui_commands()
            self.assertEqual(player.ipc.calls, [])
            player._now = 150.0
            player._handle_gui_commands()
            self.assertEqual(player.ipc.calls, [("seek", 340.0, "absolute+exact")])
            self.assertFalse(player._settle_seek(150.0))
            emit.assert_not_called()
            player.ipc.events.put({"event": "seek"})
            player._read_player_events()
            self.assertTrue(player._settle_seek(340.1))
            emit.assert_called_once_with("seek_ack", accepted=True, target=340.0)
            self.assertFalse(player._settle_seek(340.2))
            player._restart_audio_at.assert_called_once_with(340.1)

    def test_relative_and_clamped_cached_seek(self):
        for action, value, target in [("seek_relative", -10, 140), ("seek", 599, 595)]:
            with self.subTest(action=action):
                player = mock_player()
                player._cached_seek_ready = lambda pos: pos == target
                player.queue_gui_command(action, value)
                with patch("auto_translate.output.mpv.emit_event") as emit:
                    player._handle_gui_commands()
                    player.ipc.events.put({"event": "seek"})
                    player._read_player_events()
                    self.assertFalse(player._settle_seek(target))
                    emit.assert_called_once_with("seek_ack", accepted=True, target=target)
                player._restart_audio_at.assert_not_called()

    def test_native_seek_burst_and_short_jump(self):
        player = mock_player()
        with patch("auto_translate.output.mpv.time.monotonic") as clock:
            for tick in [1.0, 1.2]:
                clock.return_value = tick
                player.ipc.events.put({"event": "seek"})
                player._read_player_events()
                self.assertFalse(player._settle_seek(151.0))
            clock.return_value = 1.6
            self.assertTrue(player._settle_seek(151.0))
        player._restart_audio_at.assert_called_once_with(151.0)

    def test_rejected_seek_allows_next_command_but_disconnect_propagates(self):
        player = mock_player()
        player.ipc.command = Mock(side_effect=[MpvError("rejected"), None, ConnectionError()])
        player.queue_gui_command("seek", 200)
        player.queue_gui_command("pause", None)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            player._handle_gui_commands()
            emit.assert_called_once_with("seek_ack", accepted=False, target=200.0)
        self.assertEqual(player.ipc.command.call_args.args, ("set_property", "pause", True))
        with self.assertRaises(ConnectionError):
            player._jump(20)

    def test_late_native_landing_after_timeout(self):
        player = mock_player()
        with patch("auto_translate.output.mpv.time.monotonic") as clock, patch(
            "auto_translate.output.mpv.emit_event"
        ) as emit:
            clock.return_value = 1.0
            player._jump(30, gui=True)
            clock.return_value = 7.0
            self.assertFalse(player._settle_seek(150))
            emit.assert_called_once_with("seek_ack", accepted=False, target=180.0)
            player.ipc.events.put({"event": "seek"})
            player._read_player_events()
            clock.return_value = 7.5
            self.assertTrue(player._settle_seek(180))
        player._restart_audio_at.assert_called_once_with(180)

    def test_live_seeking_rejected_and_pause_toggles(self):
        player = mock_player(live=True)
        player._jump = lambda delta: self.fail("Live must never seek")
        player.queue_gui_command("seek", 300.0)
        player.queue_gui_command("pause", None)
        self.assertFalse(player._handle_gui_commands())
        self.assertEqual(player.ipc.calls, [("set_property", "pause", True)])

    def test_fullscreen_mpvoverlay_key_forwards_to_gui(self):
        player = mock_player()
        player.ipc.events.put({"event":"client-message","args":[FULLSCREEN_MESSAGE]})
        player.ipc.events.put({"event":"client-message","args":[EXIT_FULLSCREEN_MESSAGE]})
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertEqual(player._read_player_events(), (False, 0.0))
        self.assertEqual(emit.call_args_list[0].args, ("toggle_fullscreen",))
        self.assertEqual(emit.call_args_list[1].args, ("exit_fullscreen",))

    def test_esc_is_not_lost_when_seek_event_was_already_queued(self):
        player = mock_player()
        player.ipc.events.put({"event": "client-message", "args": ["auto-translate-jump", "10"]})
        player.ipc.events.put({"event": "client-message", "args": [EXIT_FULLSCREEN_MESSAGE]})
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertEqual(player._read_player_events(), (False, 10.0))
        emit.assert_called_once_with("exit_fullscreen")

    def test_time_status_uses_source_timeline(self):
        player = mock_player(live=False, duration=7200.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            player._emit_gui_status(False)
        event_name = emit.call_args.args[0]
        payload = emit.call_args.kwargs
        self.assertEqual(event_name, "status")
        self.assertEqual(payload["position"], 150.0)
        self.assertEqual(payload["duration"], 7200.0)
        self.assertFalse(payload["live"])


if __name__ == "__main__":
    unittest.main()


class ShutdownTests(unittest.TestCase):
    def test_quit_is_immediate_even_when_scheduler_is_busy(self):
        player = mock_player()
        player.stop = threading.Event()
        for _ in range(2):
            player.queue_gui_command("quit", None)
        self.assertTrue(player.stop.is_set())
        self.assertTrue(player._gui_commands.empty())


class PauseIntentTests(unittest.TestCase):
    def test_automatic_resume_respects_intent_on_each_new_session(self):
        for manual in (True, False):
            player = mock_player()
            player._user_paused = manual
            player._lock = threading.Lock()
            player._lines = []
            player._direct_video_loaded = True
            player._session_start = 10.
            player.ingest.generation = 1
            player.ingest.source.timeline = None
            player.ingest.video_failed = threading.Event()
            player.ingest.set_playback_position = Mock()
            player.stop = threading.Event()
            player._translation_ready = lambda: True
            player.ipc.get = lambda name: True if name == "pause" else None
            player._settle_seek = Mock(return_value=False)
            player._emit_gui_status = Mock()
            player._subtitle_ready.wait = lambda timeout, current=player: current.stop.set()
            player._play_session(False)
            self.assertEqual(("set_property", "pause", False) in player.ipc.calls, not manual)
