"""Mocked Win32 Job Object ownership tests; never spawn child processes."""
from __future__ import annotations

import ctypes

import pytest

from auto_translate import windows_job


def test_job_object_configures_kill_on_close_and_releases_handle(monkeypatch):
    closed = []
    flags = []

    def fake_api(name, *_signature):
        if name == "CreateJobObjectW":
            return lambda *_args: 123
        if name == "SetInformationJobObject":
            def configure(_handle, _class, limits_ptr, _size):
                limits = ctypes.cast(
                    limits_ptr, ctypes.POINTER(windows_job.ExtendedLimits),
                ).contents
                flags.append(limits.basic.flags)
                return 1

            return configure
        if name == "CloseHandle":
            return lambda handle: closed.append(handle) or 1
        raise AssertionError(f"unexpected Win32 API: {name}")

    monkeypatch.setattr(windows_job, "_api", fake_api)
    monkeypatch.setattr(windows_job.c, "set_last_error", lambda _value: None, raising=False)
    monkeypatch.setattr(windows_job.c, "get_last_error", lambda: 0, raising=False)

    job = windows_job.WindowsJob()

    assert flags == [0x2000]
    job.close()
    assert closed == [123]
    assert job.handle is None


def test_bind_process_tree_closes_job_if_process_assignment_fails(monkeypatch):
    class FakeJob:
        closed = False
        handle = 123

        def __init__(self):
            pass

        def close(self):
            self.closed = True

    job = FakeJob()
    monkeypatch.setattr(windows_job.os, "name", "nt")
    monkeypatch.setattr(windows_job, "_job", None)
    monkeypatch.setattr(windows_job, "WindowsJob", lambda: job)
    monkeypatch.setattr(
        windows_job, "_assign_self",
        lambda _handle: (_ for _ in ()).throw(OSError("assignment failed")),
    )

    with pytest.raises(OSError, match="assignment failed"):
        windows_job.bind_process_tree()

    assert job.closed
    assert windows_job._job is None