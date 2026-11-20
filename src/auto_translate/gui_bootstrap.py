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
    except Exception:  # noqa: S110, BLE001
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

            from auto_translate.gui import VideoTranslateWindow
            from auto_translate.gui_runner import discover_project_root
            app = QApplication([])
            window = VideoTranslateWindow(discover_project_root())
            window.show()
            QTimer.singleShot(250, app.quit)
            result = app.exec()
            window.close()
            record("GUI offscreen smoke test completed")
        else:
            result = start_gui()
        record(f"Qt application exited: {result}")
        return int(result)
    except BaseException:  # noqa: BLE001
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
