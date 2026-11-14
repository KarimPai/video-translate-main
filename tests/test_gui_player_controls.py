"""Qt-independent regression tests for embedded MPV UI protocol and seeks."""
import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.gui_control import (
    EVENT_PREFIX, FULLSCREEN_MESSAGE, EXIT_FULLSCREEN_MESSAGE,
    format_event, parse_command, parse_event,
)
from auto_translate.output.mpv import MpvOutput


class FakeIpc:
    def __init__(self):
        self.calls = []
        self.events = queue.Queue()

    def command(self, *args):
        self.calls.append(args)

    def get(self, name):
        if name == "pause":
            return False
        return None


def mock_player(live=False, duration=600.0):
    player = MpvOutput.__new__(MpvOutput)
    player.cfg = SimpleNamespace(window_id=123)
    player.ingest = SimpleNamespace(source=SimpleNamespace(live=live, duration=duration))
    player.ipc = FakeIpc()
    player._now = 150.0
    player._gui_commands = queue.Queue(maxsize=16)
    player._subtitle_ready = threading.Event()
    player._last_gui_status_at = 0.0
    return player


class GuiPlayerProtocolTests(unittest.TestCase):
    def test_valid_commands(self):
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
            '{"action":"quit"}',
            '{"action":"set_property","value":"fullscreen"}',
        ):
            with self.subTest(data=data):
                self.assertIsNone(parse_command(data))

    def test_event_format_and_parse(self):
        data = parse_event(format_event("status", live=False, position=12.2, duration=210))
        self.assertEqual(data, {"event": "status", "live": False, "position": 12.2, "duration": 210})
        self.assertIsNone(parse_event("JP: こんにちは"))
        self.assertIsNone(parse_event(EVENT_PREFIX + "not-json"))

    def test_absolute_vod_seek_reuses_scheduler(self):
        player = mock_player()
        called = []
        player._jump = lambda delta: called.append(delta) or True
        player.queue_gui_command("seek", 340.0)
        self.assertTrue(player._handle_gui_commands())
        self.assertEqual(called, [190.0])

    def test_seek_while_rebuffering_is_retried_when_mpv_clock_returns(self):
        player = mock_player()
        player._now = None
        called = []
        player._jump = lambda delta: called.append(delta) or True
        player.queue_gui_command("seek", 340.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertFalse(player._handle_gui_commands())
            emit.assert_not_called()
            self.assertEqual(player._gui_commands.qsize(), 1)
            player._now = 150.0
            self.assertTrue(player._handle_gui_commands())
            emit.assert_called_once_with("seek_ack", accepted=True, target=340.0)
        self.assertEqual(called, [190.0])

    def test_cached_seek_reports_success_without_audio_restart(self):
        player = mock_player()
        player._jump = lambda delta: False  # cached seek path
        player.queue_gui_command("seek", 260.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertFalse(player._handle_gui_commands())
        emit.assert_called_once_with("seek_ack", accepted=True, target=260.0)

    def test_seek_is_clamped_to_end_of_vod_for_ack(self):
        player = mock_player(duration=600.0)
        player._jump = lambda delta: True
        player.queue_gui_command("seek", 599.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertTrue(player._handle_gui_commands())
        emit.assert_called_once_with("seek_ack", accepted=True, target=595.0)

    def test_relative_seek_cache_path_continues_scheduler(self):
        player = mock_player()
        called = []
        player._jump = lambda delta: called.append(delta) or False
        player.queue_gui_command("seek_relative", -10.0)
        self.assertFalse(player._handle_gui_commands())
        self.assertEqual(called, [-10.0])

    def test_live_seeking_rejected_and_pause_toggles(self):
        player = mock_player(live=True)
        player._jump = lambda delta: self.fail("Live must never seek")
        player.queue_gui_command("seek", 300.0)
        player.queue_gui_command("pause", None)
        self.assertFalse(player._handle_gui_commands())
        self.assertEqual(player.ipc.calls, [("cycle", "pause")])

    def test_fullscreen_mpvoverlay_key_forwards_to_gui(self):
        player = mock_player()
        player.ipc.events.put({"event":"client-message","args":[FULLSCREEN_MESSAGE]})
        player.ipc.events.put({"event":"client-message","args":[EXIT_FULLSCREEN_MESSAGE]})
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertFalse(player._replay_requested())
        self.assertEqual(emit.call_args_list[0].args, ("toggle_fullscreen",))
        self.assertEqual(emit.call_args_list[1].args, ("exit_fullscreen",))

    def test_esc_is_not_lost_when_seek_event_was_already_queued(self):
        player = mock_player()
        player.ipc.events.put({"event": "client-message", "args": ["auto-translate-jump", "10"]})
        player.ipc.events.put({"event": "client-message", "args": [EXIT_FULLSCREEN_MESSAGE]})
        with patch("auto_translate.output.mpv.emit_event") as emit:
            self.assertFalse(player._replay_requested())
        emit.assert_called_once_with("exit_fullscreen")
        self.assertEqual(player._jump_request(), 10.0)

    def test_time_status_uses_source_timeline(self):
        player = mock_player(live=False, duration=7200.0)
        with patch("auto_translate.output.mpv.emit_event") as emit:
            player._emit_gui_status()
        event_name = emit.call_args.args[0]
        payload = emit.call_args.kwargs
        self.assertEqual(event_name, "status")
        self.assertEqual(payload["position"], 150.0)
        self.assertEqual(payload["duration"], 7200.0)
        self.assertFalse(payload["live"])


if __name__ == "__main__":
    unittest.main()
