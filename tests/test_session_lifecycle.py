import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from auto_translate import gui_control, gui_session, windows_job
from auto_translate.windows_subprocess import stopping


class LifecycleTests(unittest.TestCase):
    def test_authenticated_eof_cancels_initialization(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            stop = threading.Event()
            self.addCleanup(stopping.clear)
            control = gui_control.BackendControl(server.getsockname()[1], "test", stop)
            self.addCleanup(control.close)
            with server.accept()[0] as peer, peer.makefile("rb") as reader:
                self.assertEqual(reader.readline(), b"test\n")
                peer.sendall(b"OK\n")
            self.assertTrue(stop.wait(3))
            self.assertTrue(stopping.is_set())

    def test_assignment_failure_prevents_backend_creation(self):
        with patch.dict(os.environ, VIDEO_TRANSLATE_SESSION="test"), \
             patch.object(gui_session, "join_session", side_effect=OSError), \
             patch.object(subprocess, "Popen") as spawn:
            with self.assertRaises(OSError):
                gui_session.main()
            spawn.assert_not_called()

    def test_job_creation_failure_prevents_assignment(self):
        with patch.object(windows_job, "_job", None), \
             patch.object(windows_job.os, "name", "nt"), \
             patch.object(windows_job, "WindowsJob", side_effect=OSError), \
             patch.object(windows_job, "_assign_self") as assign:
            with self.assertRaises(OSError):
                windows_job.bind_process_tree()
            assign.assert_not_called()


@unittest.skipUnless(os.name == "nt" and importlib.util.find_spec("PySide6"),
                     "requires Windows and Qt offscreen")
class WindowsSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import venv
        cls.venv_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.venv_dir.cleanup)
        venv.EnvBuilder(with_pip=False).create(cls.venv_dir.name)

    def exercise(self, scenario):
        import _winapi
        fixture = Path(__file__).parent.resolve() / "fixtures" / "lifecycle_dummy.py"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                   DUMMY_PYTHONW=str(Path(self.venv_dir.name) / "Scripts" / "pythonw.exe"),
                   PYTHONPATH=str(fixture.parents[2] / "src"))
        # Same interpreter and native image as the owned dummy servers.
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(150)"])
        self.addCleanup(external.wait, 5)
        self.addCleanup(external.kill)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frontend = subprocess.Popen([sys.executable, str(fixture), "frontend", directory, scenario], env=env)
            try:
                for cycle in range(2 if scenario == "stop" else 1):
                    deadline = time.monotonic() + 25
                    ready = folder / "ready"
                    session_file = folder / f"session{cycle}"
                    while not (ready.exists() and session_file.exists() and session_file.stat().st_size):
                        self.assertIsNone(frontend.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(.02)
                    session = session_file.read_text()
                    members = windows_job.session_members(session)
                    expected = json.loads(ready.read_text())
                    self.assertTrue(set(expected).issubset(members), (expected, members))
                    self.assertNotIn(external.pid, members)
                    # A kernel handle pins the creation identity through PID reuse.
                    handles = []
                    try:
                        for pid in members:
                            handles.append(_winapi.OpenProcess(0x100000, False, pid))
                        if scenario == "gui-crash":
                            frontend.kill()
                        else:
                            (folder / "act").touch()
                        for handle in handles:
                            self.assertEqual(_winapi.WaitForSingleObject(handle, 15000), 0)
                    finally:
                        for handle in handles:
                            _winapi.CloseHandle(handle)
                code = frontend.wait(10)
                if scenario != "gui-crash":
                    self.assertEqual(code, 0)
                self.assertEqual(windows_job.session_members(session), [])
                self.assertIsNone(external.poll())
            finally:
                frontend.kill()
                frontend.wait(5)

    def test_window_close(self):
        self.exercise("close")

    def test_repeated_stop_and_restart(self):
        self.exercise("stop")

    def test_close_during_initialization(self):
        self.exercise("initializing")

    def test_close_during_seek(self):
        self.exercise("seek")

    def test_hung_workers_force_escalation(self):
        self.exercise("hang")

    def test_terminate_gui(self):
        self.exercise("gui-crash")

    def test_backend_crash(self):
        self.exercise("backend-crash")

    def test_socket_disconnect_during_translation(self):
        self.exercise("disconnect")
