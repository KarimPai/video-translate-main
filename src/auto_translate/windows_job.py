"""Release owned descendants when a Windows console is closed abruptly."""
import logging
import os

_job = None


def bind_process_tree():
    global _job
    if os.name != "nt" or _job is not None:
        return
    import ctypes as c
    from ctypes import wintypes as w

    class BasicLimits(c.Structure):
        _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                    ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                    ("active", w.DWORD), ("affinity", c.c_size_t), ("priority", w.DWORD),
                    ("scheduling", w.DWORD)]

    class ExtendedLimits(c.Structure):
        _fields_ = [("basic", BasicLimits), ("io", c.c_uint64 * 6),
                    ("memory", c.c_size_t * 4)]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateJobObjectW
    create.argtypes = [c.c_void_p, w.LPCWSTR]
    create.restype = w.HANDLE
    configure = kernel.SetInformationJobObject
    configure.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    assign = kernel.AssignProcessToJobObject
    assign.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    handle = create(None, None)
    if handle and configure(handle, 9, c.byref(limits), c.sizeof(limits)) and assign(handle, -1):
        _job = handle  # non-inheritable handle, released by the OS on parent exit
    else:
        logging.getLogger(__name__).warning("Windows process job unavailable: %s", c.get_last_error())
        if handle:
            kernel.CloseHandle(handle)
