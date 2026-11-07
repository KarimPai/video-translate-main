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


if __name__ == '__main__':
    unittest.main()
