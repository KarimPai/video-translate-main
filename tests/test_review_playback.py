"""Playback command concurrency and cancellable ingest/MPV waits."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate.ingest import Ingest
from auto_translate.output.mpv import MpvOutput


def player():
    result = object.__new__(MpvOutput)
    result.cfg = SimpleNamespace(window_id=123)
    result.stop = threading.Event()
    result._subtitle_ready = threading.Event()
    result._gui_commands = queue.Queue(maxsize=16)
    result._now = None
    result._pending_seek = None
    result.ingest = SimpleNamespace(source=SimpleNamespace(live=False, duration=600))
    result._jump = Mock()
    result._toggle_user_pause = Mock()
    return result


class RefilledQueue(queue.Queue):
    """Interleave one producer write after the scheduler removes a command."""
    refill = True

    def get_nowait(self):
        item = super().get_nowait()
        if self.refill:
            self.refill = False
            self.put_nowait(("seek", 400.0))
        return item


class GuiSeekConcurrencyTests(unittest.TestCase):
    def test_buffering_seek_survives_concurrent_refill_without_requeue(self):
        p = player()
        p._gui_commands = RefilledQueue(maxsize=16)
        for value in range(16):
            p.queue_gui_command("seek", float(value))
        p._handle_gui_commands()
        p._handle_gui_commands()
        self.assertTrue(p._gui_commands.empty())
        p._now = 10.0
        p._handle_gui_commands()
        p._jump.assert_called_once_with(390.0, gui=True)

    def test_latest_absolute_and_relative_seeks_coalesce_while_buffering(self):
        p = player()
        p.queue_gui_command("seek_relative", 10)
        p._handle_gui_commands()
        p.queue_gui_command("seek", 200)
        p.queue_gui_command("seek_relative", 15)
        p.queue_gui_command("pause", None)
        p._handle_gui_commands()
        p._toggle_user_pause.assert_called_once_with()
        p._jump.assert_not_called()
        p._now = 100.0
        p._handle_gui_commands()
        p._jump.assert_called_once_with(115.0, gui=True)

    def test_full_queue_retry_tolerates_concurrent_producer(self):
        p = player()
        p._gui_commands = RefilledQueue(maxsize=16)
        for value in range(16):
            p.queue_gui_command("seek", float(value))
        p.queue_gui_command("seek", 500.0)
        p._handle_gui_commands()
        p._now = 100.0
        p._handle_gui_commands()
        p._jump.assert_called_once_with(400.0, gui=True)


class CancellablePlayerWaitTests(unittest.TestCase):
    def test_event_wait_stops_and_preserves_unrelated_events(self):
        p = player()

        class CancelOnRead(queue.Queue):
            def get(self, *args, **kwargs):
                event = super().get(*args, **kwargs)
                p.stop.set()
                return event

        events = CancelOnRead()
        events.put({"event": "client-message", "args": ["fullscreen"]})
        p.ipc = SimpleNamespace(events=events)
        self.assertFalse(p._wait_for_event("file-loaded", timeout_s=30))
        self.assertEqual(events.get_nowait()["event"], "client-message")

    def test_initial_seek_confirmation_observes_stop(self):
        p = player()
        p.cfg.window_id = None
        p._direct_video_loaded = False
        p._vsr_requested = False
        p.ingest.video_out = None
        p.ingest.source.start_s = 100.0
        p.ingest.source.location = "https://example.test/video"
        p._wait_for_event = Mock(return_value=True)
        p._restart_audio_at = Mock()
        p.ipc = Mock()
        p.ipc.get.return_value = [{"type": "video"}]
        p.ipc.command.side_effect = lambda name, *args: p.stop.set() if name == "seek" else None
        p._play_session(False)
        p._restart_audio_at.assert_not_called()
        p.ipc.get.assert_called_once_with("track-list")


class IngestQueueCancellationTests(unittest.TestCase):
    @staticmethod
    def ingest():
        ingest = object.__new__(Ingest)
        ingest._proc_lock = threading.Condition()
        ingest._stopped = False
        ingest._generation = 1
        return ingest

    def exercise_cancelled_write(self, *, seek=False):
        ingest = self.ingest()
        attempted = threading.Event()

        class ObservedQueue(queue.Queue):
            def put(self, *args, **kwargs):
                attempted.set()
                return super().put(*args, **kwargs)

        output = ObservedQueue(maxsize=1)
        output.put("existing")
        attempted.clear()
        result = []
        thread = threading.Thread(target=lambda: result.append(ingest._put_audio(output, "stale", 1)))
        thread.start()
        self.assertTrue(attempted.wait(1))
        with ingest._proc_lock:
            if seek:
                ingest._generation += 1
            else:
                ingest._stopped = True
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        self.assertEqual(output.get_nowait(), "existing")
        self.assertTrue(output.empty())

    def test_stop_cancels_full_audio_queue(self):
        self.exercise_cancelled_write()

    def test_seek_cancels_old_generation_waiting_on_full_queue(self):
        self.exercise_cancelled_write(seek=True)

    def test_stopped_ingest_does_not_block_sending_eof(self):
        ingest = self.ingest()
        ingest._stopped = True
        output = queue.Queue(maxsize=1)
        output.put("existing")
        self.assertFalse(ingest._put_audio(output, None))

    def test_normal_eof_follows_audio_in_queue(self):
        ingest = self.ingest()
        output = queue.Queue(maxsize=1)
        output.put("existing")
        result = []
        thread = threading.Thread(target=lambda: result.append(ingest._put_audio(output, None)))
        thread.start()
        self.assertEqual(output.get(timeout=1), "existing")
        thread.join(1)
        self.assertEqual(result, [True])
        self.assertIsNone(output.get_nowait())
