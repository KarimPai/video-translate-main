"""Win32 job ownership. All handles are non-inheritable by default."""
import ctypes as c
from ctypes import wintypes as w
import os
import uuid

_job = None
_console_handler = None


class BasicLimits(c.Structure):
    _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                ("active", w.DWORD), ("affinity", c.c_size_t), ("priority", w.DWORD),
                ("scheduling", w.DWORD)]


class ExtendedLimits(c.Structure):
    _fields_ = [("basic", BasicLimits), ("io", c.c_uint64 * 6),
                ("memory", c.c_size_t * 4)]


def _api(name, result, *args):
    fn = getattr(c.WinDLL("kernel32", use_last_error=True), name)
    fn.restype = result
    fn.argtypes = args
    return fn


def _check(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


def job_name(session):
    return "Local\\VideoTranslate-" + uuid.UUID(session).hex


class WindowsJob:
    def __init__(self, session=None):
        create = _api("CreateJobObjectW", w.HANDLE, c.c_void_p, w.LPCWSTR)
        c.set_last_error(0)
        self.handle = _check(create(None, job_name(session) if session else None))
        try:
            if c.get_last_error() == 183:
                raise FileExistsError("Session job already exists")
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway
            configure = _api("SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int,
                             c.c_void_p, w.DWORD)
            _check(configure(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def terminate(self):
        _check(_api("TerminateJobObject", w.BOOL, w.HANDLE, w.UINT)(self.handle, 1))

    def pids(self):
        return _members(self.handle)

    def close(self):
        if self.handle:
            _check(_api("CloseHandle", w.BOOL, w.HANDLE)(self.handle))
            self.handle = None


def _members(handle):
    query = _api("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int,
                 c.c_void_p, w.DWORD, c.POINTER(w.DWORD))
    size = 512
    while True:
        buf = c.create_string_buffer(size)
        if query(handle, 3, buf, size, None):
            count = w.DWORD.from_buffer(buf, 4).value
            return list((c.c_size_t * count).from_buffer(buf, 8))
        if c.get_last_error() != 234:  # ERROR_MORE_DATA
            raise c.WinError(c.get_last_error())
        size *= 2


def _open_session(session, access):
    return _check(_api("OpenJobObjectW", w.HANDLE, w.DWORD, w.BOOL, w.LPCWSTR)(
        access, False, job_name(session)))


def session_members(session):
    try:
        handle = _open_session(session, 4)  # QUERY only
    except FileNotFoundError:
        return []
    try:
        return _members(handle)
    finally:
        _api("CloseHandle", w.BOOL, w.HANDLE)(handle)


def _assign_self(handle):
    _check(_api("AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)(handle, -1))


def join_session(session):
    # Join before loading any library that may spawn a helper. Close the
    # temporary assign-only handle immediately; the GUI owns the last handle.
    handle = _open_session(session, 1)
    try:
        _assign_self(handle)
    finally:
        _api("CloseHandle", w.BOOL, w.HANDLE)(handle)


def bind_process_tree():
    global _job, _console_handler
    if os.name != "nt" or _job is not None:
        return
    job = WindowsJob()
    try:
        _assign_self(job.handle)
    except BaseException:
        job.close()
        raise
    _job = job  # Kernel releases the handle even on TerminateProcess.
    @c.WINFUNCTYPE(w.BOOL, w.DWORD)
    def console_handler(event):
        if event == 2:
            os._exit(1)
        return False
    _console_handler = console_handler
    _api("SetConsoleCtrlHandler", w.BOOL, type(console_handler), w.BOOL)(console_handler, True)
