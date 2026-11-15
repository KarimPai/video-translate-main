"""Prevent console flashes from backend helpers launched by the Qt GUI.

The classic Start translate.bat retains its existing console behavior. Only
GUI-launched sessions request CREATE_NO_WINDOW for console-mode children.
"""
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import CancelledError


def hidden_console_flags() -> int:
    if os.name == "nt" and os.environ.get("VIDEO_TRANSLATE_GUI") == "1":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


# Each backend interpreter serves one translation session.
stopping = threading.Event()


def check_running():
    if stopping.is_set():
        raise CancelledError("translation session is stopping")


def guard_gui_spawns():
    # Also covers multiprocessing's direct _winapi.CreateProcess call and
    # subprocess.run/check_output inside third-party libraries.
    def check_spawn(event, args):
        if event in ("subprocess.Popen", "_winapi.CreateProcess"):
            check_running()
    sys.addaudithook(check_spawn)


def owned_popen(*args, **kwargs):
    check_running()
    proc = subprocess.Popen(*args, **kwargs)
    logging.getLogger(__name__).info("session=%s name=%s pid=%s parent=%s owned=true",
        os.getenv("VIDEO_TRANSLATE_SESSION", "cli"), os.path.basename(str(args[0][0])),
        proc.pid, os.getpid())
    return proc
