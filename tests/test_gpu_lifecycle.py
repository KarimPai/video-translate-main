"""Mocked lifecycle regressions for LlamaServer (no real process, no GPU)."""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_translate import llama_server as ls


class FakeProc:
    _next_pid = 1000

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

    monkeypatch.setattr(ls, "owned_popen", fake_popen)
    monkeypatch.setattr(ls, "_free_port", lambda: 54321)
    servers: list[ls.LlamaServer] = []

    def factory(*, timeout=0.4, get=None):
        exe = tmp_path / "llama-server.exe"
        server = ls.LlamaServer("t", exe, ["--model", "x.gguf"], startup_timeout_s=timeout)
        server.log_path = tmp_path / "t-server.log"
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

    monkeypatch.setattr(ls, "check_running", cancel_on_third)
    with pytest.raises(CancelledError):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated, "loading child must not survive a cancelled startup"
    assert server.process is None
    assert not server._closed, "cancellation of one start must not poison the server"


def test_health_timeout_kills_child(make_server):
    server = make_server(timeout=0.3)
    with pytest.raises(ls.LlamaServerError, match="did not load"):
        server.start()
    (proc,) = make_server.spawned
    assert proc.terminated
    assert server.process is None


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

    monkeypatch.setattr(ls, "owned_popen", die_immediately)
    with pytest.raises(ls.LlamaServerError, match="exited during startup"):
        server.start()
    assert server.process is None
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


def test_real_llama_server_spawn_is_blocked_by_default(monkeypatch, tmp_path):
    """The conftest audit hook refuses llama-server before any process exists."""
    monkeypatch.delenv("VIDEO_TRANSLATE_RUN_REAL_MODELS", raising=False)
    with pytest.raises(RuntimeError, match="Blocked"):
        subprocess.Popen([str(tmp_path / "llama-server.exe"), "--version"])
    with pytest.raises(RuntimeError, match="Blocked"):
        subprocess.Popen([str(tmp_path / "LLAMA-SERVER.EXE"), "--list-devices"])


def test_guard_allows_opt_in_and_other_programs(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_TRANSLATE_RUN_REAL_MODELS", "1")
    missing = tmp_path / "llama-server.exe"  # does not exist: spawn fails with OSError, not the guard
    with pytest.raises(OSError):
        subprocess.Popen([str(missing)])
    monkeypatch.delenv("VIDEO_TRANSLATE_RUN_REAL_MODELS")
    proc = subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert proc.returncode == 0
