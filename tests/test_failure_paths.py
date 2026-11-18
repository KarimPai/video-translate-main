import http.client
import io
import json
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


from auto_translate.config import load_config
from auto_translate.ingest import parse_time, start_from_url
from auto_translate.output.mpv import MpvIpc, MpvOutput, _WindowsPipe, _create_kill_job
from auto_translate.output.web import BACKLOG, WebOutput
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.translate.plamo import clean_output as _clean
from auto_translate.types import Subtitle
from auto_translate.ytdlp import cache_key


class FailurePathTests(unittest.TestCase):

    def test_scheduler_failure_stops_backend(self):
        player = object.__new__(MpvOutput)
        player.stop = threading.Event()
        player._play_session = Mock(side_effect=KeyError("bad state"))
        with self.assertLogs("auto_translate.output.mpv", level="ERROR"):
            player._run()
        self.assertTrue(player.stop.is_set())
        self.assertIsInstance(player.scheduler_failure, KeyError)

    def test_config_unknown_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text("[vad]\nfuture_option=true\nthreshold=0.5\n")
            with self.assertLogs("auto_translate.config", level="WARNING"):
                self.assertEqual(load_config(config).vad.threshold, .5)
            for contents in ('[asr]\nblocklist=[""]', '[vad]\nneg_threshold=0.9',
                             '[vad]\nmin_silence_ms=-1'):
                config.write_text(contents)
                with self.assertRaises(ValueError):
                    load_config(config)

    def test_viewer_is_loopback_unless_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[output]\n")
            self.assertEqual(load_config(path).output.web_host, "127.0.0.1")
            path.write_text('[output]\nweb_host = "0.0.0.0"\n')
            self.assertEqual(load_config(path).output.web_host, "0.0.0.0")

    def test_deepfilter_close_enqueues_one_stop_and_retains_error(self):
        try:
            from auto_translate.deepfilter import DeepFilterWorker
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                self.skipTest("optional DeepFilter scipy dependency unavailable")
            raise
        with patch.object(DeepFilterWorker, "_run"):
            worker = DeepFilterWorker(queue.Queue(), 0, 0)
        worker.error = ValueError("enhancer failed")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "enhancer failed"):
                worker.close()
        self.assertIsNone(worker.input_q.get_nowait())
        self.assertTrue(worker.input_q.empty())

    def test_mpv_job_skips_mock_before_native_calls(self):
        with patch("auto_translate.output.mpv.sys.platform", "win32"):
            self.assertIsNone(_create_kill_job(Mock()))

    def test_mpv_write_preserves_timeout_during_cleanup(self):
        pipe = object.__new__(_WindowsPipe)
        pipe._api = Mock()
        pipe._handle = 42
        ov = Mock()
        pipe._api.WriteFile.return_value = ov, 0
        pipe._api.WaitForSingleObject.return_value = 258
        ov.cancel.side_effect = OSError("pipe closed")
        with self.assertRaisesRegex(TimeoutError, "pipe write timed out"):
            pipe.write(b"quit")

    def test_mpv_reader_keeps_valid_reply_after_malformed_event(self):
        ipc = object.__new__(MpvIpc)
        ipc._f = io.BytesIO(b'[]\n{"event":"property-change"}\n'
                            b'{"event":"property-change","name":"pause","data":true}\n'
                            b'{"request_id":1}\n')
        ipc._properties = {}
        ipc._responses = queue.Queue()
        ipc._read_messages()
        self.assertEqual(ipc._properties, {"pause": True})
        self.assertEqual(ipc._responses.get_nowait(), {"request_id": 1})
        self.assertIsInstance(ipc._reader_failure, ConnectionError)

    def test_mpv_context_closes_on_exception(self):
        player = object.__new__(MpvOutput)
        player.close = Mock()
        with self.assertRaises(ValueError), player as active:
            self.assertIs(active, player)
            raise ValueError()
        player.close.assert_called_once()

    def test_long_cue_multiline_translation_and_session_cache(self):
        text = _clean('EN: “The first customer arrived.\n\nThe next customer is still outside.”')
        self.assertEqual(text, "“The first customer arrived. The next customer is still outside.”")
        cue = Subtitle(60, 76, "長い発話", text)
        cache = SessionSubtitleCache()
        self.assertTrue(cache.add(cue))
        self.assertFalse(cache.add(cue))
        self.assertEqual(cache.active_at(75.9), [cue])
        self.assertEqual(cache.match(60.05, 76).en, text)
        self.assertFalse(cache.active_at(76))

    def test_web_query_invalid_event_id_and_backpressure(self):
        hub = WebOutput("127.0.0.1", 0)
        self.addCleanup(hub.close)
        stalled = queue.Queue(maxsize=BACKLOG)
        hub._clients.add(stalled)
        for i in range(BACKLOG + 10):
            hub.show(Subtitle(i, i+1, "日本語", str(i)))
        self.assertEqual(stalled.qsize(), BACKLOG)
        self.assertEqual(stalled.get_nowait().id, 11)
        conn = http.client.HTTPConnection(*hub._server.server_address, timeout=3)
        self.addCleanup(conn.close)
        conn.request("GET", "/events?viewer=phone", headers={"Last-Event-ID": "bad"})
        with conn.getresponse() as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.readline().strip(), b"id: 58")
            self.assertEqual(json.loads(response.readline()[6:])["en"], "57")
        with patch("auto_translate.output.web.MAX_CLIENTS", 1):
            second = http.client.HTTPConnection(*hub._server.server_address, timeout=3)
            self.addCleanup(second.close)
            second.request("GET", "/events")
            self.assertEqual(second.getresponse().status, 503)

    def test_video_id_and_time_formats(self):
        for video_id in ("../unsafe", "a/b", "CON", "a:b", "%(ext)s"):
            self.assertRegex(cache_key(video_id), r"^id\.[0-9a-f]{64}$")
        self.assertEqual(cache_key("normal_ID-1"), "normal_ID-1")
        self.assertEqual(parse_time("1:02:03"), 3723)
        self.assertEqual(start_from_url("https://example.org/watch#t=16m2s"), 962)


@unittest.skipUnless(sys.platform == "win32", "Windows Job Object integration")
class WindowsJobTests(unittest.TestCase):
    def test_exited_session_owner_with_retained_handle(self):
        from auto_translate.ingest import _owner_gone
        with subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                              stdin=subprocess.PIPE) as owner:
            self.assertFalse(_owner_gone(owner.pid))
            owner.communicate(timeout=5)
            self.assertTrue(_owner_gone(owner.pid))

    def test_child_terminates_when_backend_is_killed(self):
        import ctypes
        from ctypes import wintypes

        script = """
from auto_translate.windows_job import bind_process_tree
import subprocess, sys, time
bind_process_tree()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print(child.pid, flush=True)
time.sleep(60)
"""
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        with subprocess.Popen([sys.executable, "-u", "-c", script], stdout=subprocess.PIPE) as parent:
            try:
                handle = kernel.OpenProcess(0x100000, False, int(parent.stdout.readline()))
                self.assertTrue(handle)
                try:
                    parent.kill()
                    parent.wait(timeout=5)
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                finally:
                    kernel.CloseHandle(handle)
            finally:
                parent.kill()
