"""Mocked lifecycle regressions for SpeechCppServer (no real process, no GPU).

Mirrors tests/test_gpu_lifecycle.py: the speech.cpp child must never survive
a cancelled/failed startup, failed termination must retain ownership, and a
health timeout must leave the server restartable for HaasASR._restart().
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_translate import speech_server as ss


class FakeProc:
    _next_pid = 2000

    def __init__(self) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.killed = True
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def make_server(tmp_path, monkeypatch):
    spawned: list[FakeProc] = []

    def fake_popen(*_args, **_kwargs):
        proc = FakeProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(ss, "owned_popen", fake_popen)
    monkeypatch.setattr(ss, "_free_port", lambda: 54321)
    servers: list[ss.SpeechCppServer] = []

    def factory(*, timeout=0.4, get=None):
        exe = tmp_path / "speech.exe"
        server = ss.SpeechCppServer("haas-ja-v1.0", exe, startup_timeout_s=timeout)
        server.log_path = tmp_path / "haas-server.log"
        server.client = SimpleNamespace(
            get=get or (lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down"))),
            close=lambda: None,
        )
        servers.append(server)
        return server

    factory.spawned = spawned
    yield factory
    for server in servers:
        server.close()


def test_cancellation_during_startup_reaps_child_and_stays_restartable(make_server, monkeypatch):
    server = make_server(timeout=5.0)
    calls = {"n": 0}

    def cancel_on_third(*_a):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise CancelledError("stopping")

    monkeypatch.setattr(ss, "check_running", cancel_on_third)
    with pytest.raises(CancelledError):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated, "loading child must not survive a cancelled startup"
    assert server.process is None
    assert not server._closed, "cancellation of one start must not poison the server"


def test_health_timeout_kills_child_and_remains_restartable(make_server):
    server = make_server(timeout=0.3)
    with pytest.raises(ss.SpeechServerError, match="did not start"):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated
    assert server.process is None
    assert not server._closed, "health timeout must leave the server restartable"
    server.client.get = lambda *_a, **_k: SimpleNamespace(status_code=200)
    server.start()
    assert len(make_server.spawned) == 2
    assert make_server.spawned[-1].returncode is None


def test_start_lock_wait_obeys_operation_deadline(make_server):
    server = make_server(timeout=5.0)
    server._lock.acquire()
    deadline = time.monotonic() + 0.03
    started = time.monotonic()
    try:
        with pytest.raises(ss.SpeechServerError, match="deadline"):
            server.start(deadline=deadline)
        assert time.monotonic() - started < 0.2
        assert make_server.spawned == []
    finally:
        server._lock.release()


def test_unexpected_error_while_waiting_kills_child(make_server):
    def boom(*_a, **_k):
        raise RuntimeError("client closed")

    server = make_server(timeout=5.0, get=boom)
    with pytest.raises(RuntimeError):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated
    assert server.process is None


def test_child_exit_during_startup_raises_and_clears_process(make_server, monkeypatch):
    server = make_server(timeout=5.0)

    def die_immediately(*_a, **_k):
        proc = FakeProc()
        proc.returncode = 3
        make_server.spawned.append(proc)
        return proc

    monkeypatch.setattr(ss, "owned_popen", die_immediately)
    with pytest.raises(ss.SpeechServerError, match="exited during startup"):
        server.start()
    assert server.process is None
    assert server._log_stream is None
    assert not server.alive()


def test_restart_never_runs_two_children(make_server):
    ok = SimpleNamespace(status_code=200)
    server = make_server(timeout=2.0, get=lambda *a, **k: ok)
    server.start()
    server.start()
    first, second = make_server.spawned
    assert first.terminated, "old server must be stopped before the replacement is launched"
    assert second.returncode is None
    assert server.process is second
    assert sum(p.returncode is None for p in make_server.spawned) == 1


def test_spawn_failure_closes_log_stream(make_server, monkeypatch):
    server = make_server()

    def fail_spawn(*_args, **_kwargs):
        raise OSError("simulated Popen failure")

    monkeypatch.setattr(ss, "owned_popen", fail_spawn)
    with pytest.raises(OSError, match="simulated Popen failure"):
        server.start()
    assert server.process is None
    assert server._log_stream is None
    assert not server._closed


def test_watcher_start_failure_reaps_spawned_child(make_server, monkeypatch):
    server = make_server()

    class BrokenThread:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("watcher failed")

    monkeypatch.setattr(ss, "threading", SimpleNamespace(Thread=BrokenThread))
    with pytest.raises(RuntimeError, match="watcher failed"):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated
    assert server.process is None
    assert server._log_stream is None


def test_failed_termination_keeps_process_owned_for_retry(make_server, monkeypatch):
    class StubbornProc(FakeProc):
        failed_kill = False

        def terminate(self):
            self.terminated = True  # Simulate a terminate request being ignored.

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("speech", timeout)
            return self.returncode

        def kill(self):
            self.killed = True
            if not self.failed_kill:
                self.failed_kill = True
                raise PermissionError("simulated kill failure")
            self.returncode = 1

    proc = StubbornProc()
    server = make_server(get=lambda *_a, **_k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(ss, "owned_popen", lambda *_a, **_k: proc)
    server.start()

    with pytest.raises(PermissionError, match="simulated kill failure"):
        server.close()
    assert server.process is proc, "failed termination must retain the child handle"
    assert server._log_stream is None, "the log must close even when kill fails"
    server.close()  # Repeated close must retry the remaining child.
    assert server.process is None
    assert proc.killed and proc.returncode is not None


def test_haas_honors_speech_server_path_override(monkeypatch):
    """--speech-server writes asr.speech_server_path; Haas must actually use it."""
    from auto_translate.asr import haas
    from auto_translate.config import AsrConfig

    seen = {}

    def fake_resolve(path=None):
        seen["path"] = path
        raise ss.SpeechServerError("stop before spawn")

    monkeypatch.setattr(haas, "resolve_speech_executable", fake_resolve)
    cfg = AsrConfig(model_path="haas-ja-v1.0", server_path="cfg-speech.exe",
                    speech_server_path="cli-speech.exe")
    with pytest.raises(ss.SpeechServerError):
        haas.HaasASR(cfg)
    assert seen["path"] == "cli-speech.exe"


def test_haas_falls_back_to_server_path(monkeypatch):
    from auto_translate.asr import haas
    from auto_translate.config import AsrConfig

    seen = {}

    def fake_resolve(path=None):
        seen["path"] = path
        raise ss.SpeechServerError("stop before spawn")

    monkeypatch.setattr(haas, "resolve_speech_executable", fake_resolve)
    cfg = AsrConfig(model_path="haas-ja-v1.0", server_path="cfg-speech.exe")
    with pytest.raises(ss.SpeechServerError):
        haas.HaasASR(cfg)
    assert seen["path"] == "cfg-speech.exe"