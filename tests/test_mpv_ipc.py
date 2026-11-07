import json
import queue
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvError, MpvIpc, MpvOutput, _WindowsPipe, _Line
from auto_translate.types import Subtitle


class WindowsPipeTests(unittest.TestCase):
    def test_overlapped_transport_preserves_fragmented_json_lines(self):
        chunks = iter([b'{"event":', b'"start-file"}\n{"error":"success"}\n'])
        writes = []

        class Operation:
            event = 1

            def __init__(self, data):
                self.data = data

            def GetOverlappedResult(self, wait):
                return len(self.data), 0

            def getbuffer(self):
                return self.data

        api = SimpleNamespace(
            GENERIC_READ=1, GENERIC_WRITE=2, OPEN_EXISTING=3,
            FILE_FLAG_OVERLAPPED=4,
            CreateFile=lambda *args: args,
            ReadFile=lambda handle, size, *, overlapped: (Operation(next(chunks)), 0),
            WriteFile=lambda handle, data, *, overlapped: (writes.append(data) or Operation(data), 0),
            WaitForSingleObject=lambda event, timeout: 0,
        )
        with patch.dict(sys.modules, {'_winapi': api}):
            pipe = _WindowsPipe('test-pipe')
        self.assertEqual(pipe._handle[3], 0)  # NULL security attributes must be an integer.
        self.assertEqual(pipe._handle[-2], api.FILE_FLAG_OVERLAPPED)
        pipe.write(b'command\n')
        self.assertEqual(json.loads(pipe.readline()), {'event': 'start-file'})
        self.assertEqual(json.loads(pipe.readline()), {'error': 'success'})
        self.assertEqual(writes, [b'command\n'])


class IpcTests(unittest.TestCase):
    def test_reader_can_wait_while_command_is_sent(self):
        replies = queue.Queue()
        seen = []

        class Transport:
            def readline(self):
                return replies.get(timeout=2)

            def write(self, data):
                self.assert_newline(data)
                command = json.loads(data)
                seen.append(command)
                replies.put(b'{"event":"start-file"}\n')
                replies.put(b'{"event":"file-loaded"}\n')
                replies.put(json.dumps({'request_id': command['request_id'], 'error': 'success', 'data': 2.5}).encode() + b'\n')

            def assert_newline(self, data):
                if not data.endswith(b'\n'):
                    raise AssertionError('IPC command lacks newline')

            def close(self):
                replies.put(b'')

        with patch('auto_translate.output.mpv.sys.platform', 'win32'), patch(
            'auto_translate.output.mpv._WindowsPipe', return_value=Transport()
        ):
            ipc = MpvIpc('test-pipe')
        try:
            self.assertEqual(ipc.get('time-pos'), 2.5)
            out = object.__new__(MpvOutput)
            out.ipc = ipc
            self.assertTrue(out._wait_for_event('start-file', 1))
            self.assertTrue(out._wait_for_event('file-loaded', 1))
            self.assertEqual(seen[0]['command'], ['get_property', 'time-pos'])
        finally:
            ipc.close()
            ipc._reader.join(2)


class OverlayTests(unittest.TestCase):
    def test_overlay_is_committed_only_after_acknowledgement(self):
        out = object.__new__(MpvOutput)
        out.cfg = MpvConfig()
        out._lock = threading.Lock()
        out._lines = [_Line(Subtitle(1, 3, '日本語', 'Visible translation'), 1, 3)]
        out._shown = ''
        out._lags = []
        out._logged_displayed_subtitles = set()
        calls = []

        def command(*args, **named):
            calls.append((args, named))
            if len(calls) == 1:
                raise MpvError('invalid parameter')

        out.ipc = SimpleNamespace(command=command)
        with self.assertRaises(MpvError):
            out._render(2)
        self.assertEqual(out._shown, '')
        out._render(2)
        self.assertIn('Visible translation', out._shown)
        self.assertEqual(calls[-1][1]['format'], 'ass-events')
        out._render(4)
        self.assertEqual(calls[-1][1]['format'], 'none')
        self.assertEqual(out._shown, '')



class MpvShutdownTests(unittest.TestCase):
    def test_normal_mpv_close_does_not_report_ipc_as_playback_failure(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = SimpleNamespace(poll=lambda: 0)

        def closed_player(show_hint):
            raise BrokenPipeError("MPV window closed")

        out._play_session = closed_player
        with patch("auto_translate.output.mpv.log") as log:
            out._run()
            self.assertTrue(out.stop.is_set())
            log.info.assert_called_once()
            log.exception.assert_not_called()
            log.error.assert_not_called()

    def test_unexpected_mpv_exit_exposes_exit_code(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = SimpleNamespace(poll=lambda: 2)

        def crashed_player(show_hint):
            raise BrokenPipeError("MPV IPC closed unexpectedly")

        out._play_session = crashed_player
        with patch("auto_translate.output.mpv.log") as log:
            out._run()
            self.assertTrue(out.stop.is_set())
            log.error.assert_called_once()
            self.assertIn("code %s", log.error.call_args.args[0])
            self.assertEqual(log.error.call_args.args[1], 2)


class TranslationBufferTests(unittest.TestCase):
    def test_manual_play_bypasses_prebuffer_without_forced_pause(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out.cfg = MpvConfig()
        out._direct_video_loaded = True
        out._subtitle_ready = threading.Event()
        out._lock = threading.Lock()
        out._lines = []
        out._buffer_condition = threading.Condition()
        out._source_finished = False
        out._translated_until = 0.0
        out._now = None
        out._clock_last_sample_at = 0.0
        out._clock_last_sample_pos = None
        out._logged_displayed_subtitles = set()
        out._lags = []
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(
                start_s=0.0, live=False, duration=200.0, timeline=None
            ),
            generation=0,
            first_pts=0.0,
            first_pts_ready=SimpleNamespace(wait=lambda timeout: True),
            video_failed=threading.Event(),
        )

        commands = []
        out.ipc = SimpleNamespace(
            command=lambda *args, **kwargs: commands.append(args),
            get=lambda key: False if key == 'pause' else (
                0.0 if key == 'time-pos' else None
            ),
        )
        out._translation_ready = lambda: False
        out._render = lambda now: None
        out._replay_requested = lambda: False
        out._jump_request = lambda: 0

        def exit_after_tick(timeout):
            out.stop.set()

        out._subtitle_ready.wait = exit_after_tick
        out._play_session(show_hint=False)
        self.assertFalse(
            any(args[:3] == ('set_property', 'pause', True) for args in commands),
            'Manual Play must not be immediately reversed by MPV controller',
        )

    def test_vod_never_has_a_continuous_playback_low_watermark(self):
        # Fast ingest and the initial/seek buffer should not force recurring
        # stop/start playback when translation runs slower than real-time.
        self.assertFalse(hasattr(MpvOutput, 'PAUSE_LOW_WATERMARK_SECONDS'))

    def test_five_seconds_of_translated_coverage_required(self):
        out = object.__new__(MpvOutput)
        out._buffer_condition = threading.Condition()
        out._now = 120.0
        out._translated_until = 124.9
        out._source_finished = False

        self.assertFalse(out._translation_ready())
        out.advance(125.0)
        self.assertTrue(out._translation_ready())
        out._now = 126.0
        self.assertFalse(out._translation_ready())
        out.finish()
        self.assertTrue(out._translation_ready())

    def test_seek_resets_old_translation_coverage_immediately(self):
        out = object.__new__(MpvOutput)
        out._buffer_condition = threading.Condition()
        out._lock = threading.Lock()
        out._now = 120.0
        out._translated_until = 900.0
        out._source_finished = True
        out._lines = [_Line(Subtitle(119, 121, '日本語', 'Old'), 119, 121)]
        out._shown = 'Old'
        out._mpv_anchor = 120.0
        out._clock_source_anchor = 120.0
        out._pending_seek_offset = None
        out._clock_sample_logged = True
        out._last_direct_pos = 120.0
        out._last_direct_sample_at = 0.0
        calls = []
        out.ipc = SimpleNamespace(
            get=lambda prop: 120.0 if prop == 'time-pos' else None,
            command=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        restarts = []
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(duration=3600.0),
            request_restart=lambda pos: restarts.append(pos),
            go=lambda: restarts.append('go'),
        )

        self.assertTrue(out._restart_audio_at(500.0, seek_video=False))
        self.assertEqual(restarts, [500.0, 'go'])
        self.assertEqual(out._translated_until, 500.0)
        self.assertFalse(out._source_finished)
        self.assertIsNone(out._now)
        self.assertFalse(out._lines)
        self.assertFalse(out._translation_ready())
        self.assertTrue(any(args[:1] == ('osd-overlay',) for args, _ in calls))

    def test_session_anchor_uses_new_ingest_start_after_seek(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out.cfg = MpvConfig()
        out._direct_video_loaded = True
        out._now = None
        out._buffer_condition = threading.Condition()
        out._translated_until = 0.0
        out._source_finished = False
        source = SimpleNamespace(start_s=10.0, live=False, duration=3600.0)

        def first_audio_ready(timeout):
            source.start_s = 600.0
            out.ingest.first_pts = 600.0
            return True

        out.ingest = SimpleNamespace(
            source=source,
            generation=1,
            first_pts=None,
            first_pts_ready=SimpleNamespace(wait=first_audio_ready),
        )

        def get(prop):
            if prop == 'time-pos':
                out.stop.set()
                return 600.0
            return None

        out.ipc = SimpleNamespace(get=get)
        out._play_session(show_hint=False)
        self.assertEqual(out._session_start, 600.0)
        self.assertEqual(out._clock_source_anchor, 600.0)
        self.assertEqual(out._mpv_anchor, 600.0)

if __name__ == '__main__':
    unittest.main()
