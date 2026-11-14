"""Crash-visible, console-free entry point for the Windows GUI launcher.

A Pythonw launch has no attached console; write diagnostic logs *before*
importing Qt, and report startup exceptions using a native Windows dialog.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


def startup_log_path() -> Path:
    # WScript sets the working directory to the repository root.
    root = Path(os.environ.get("VIDEO_TRANSLATE_PROJECT_DIR") or os.getcwd())
    return root / "logs" / "gui-startup.log"


def record(message: str) -> None:
    try:
        log = startup_log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
    except OSError:
        pass


def show_error(message: str) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, message, "Video Translate - Startup Error", 0x10 | 0x00040000
        )
    except Exception:
        pass


def main() -> int:
    record(f"Launcher started: {sys.executable}")
    try:
        # Import only here so missing Qt / broken DLLs can be diagnosed.
        from auto_translate.gui import main as start_gui
        record("PySide6 imported; entering Qt application")
        if "--smoke-test" in sys.argv:
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication
            # Use the regular startup, with a tiny scheduled exit. The Qt
            # event loop is still exercised by the real GUI entry point.
            original_exec = QApplication.exec

            def short_exec(self):
                QTimer.singleShot(150, self.quit)
                return original_exec(self)

            QApplication.exec = short_exec
        result = start_gui()
        record(f"Qt application exited: {result}")
        return int(result)
    except BaseException:
        tb = traceback.format_exc()
        record("STARTUP FAILURE\n" + tb)
        show_error(
            "Video Translate could not open.\n\n"
            + tb[-1400:]
            + "\n\nFull details: " + str(startup_log_path())
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
