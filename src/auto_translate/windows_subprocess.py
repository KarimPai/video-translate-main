"""Prevent console flashes from backend helpers launched by the Qt GUI.

The classic Start translate.bat retains its existing console behavior. Only
GUI-launched sessions request CREATE_NO_WINDOW for console-mode children.
"""
import os
import subprocess


def hidden_console_flags() -> int:
    if os.name == "nt" and os.environ.get("VIDEO_TRANSLATE_GUI") == "1":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0
