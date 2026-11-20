"""Output package — playback and display backends."""

from .base import Output
from .mpv import MpvOutput
from .terminal import TerminalOutput
from .web import WebOutput

__all__ = ["MpvOutput", "Output", "TerminalOutput", "WebOutput"]
