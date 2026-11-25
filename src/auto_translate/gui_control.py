"""Loopback player commands and line-oriented backend status events."""
from __future__ import annotations

import json
import logging
import math
import socket
import sys
import threading

GUI_SHUTDOWN_MS = 20000
WORKER_DRAIN_S = 3.0


EVENT_PREFIX = "VIDEO_TRANSLATE_GUI_EVENT:"
FULLSCREEN_MESSAGE = "auto-translate-gui-fullscreen"
EXIT_FULLSCREEN_MESSAGE = "auto-translate-gui-exit-fullscreen"
_STDOUT_LOCK = threading.Lock()


def parse_command(line: str) -> tuple[str, float | None] | None:
    """Whitelist actions so GUI input cannot turn into arbitrary MPV commands."""
    if not isinstance(line, str) or len(line) > 4096:
        return None
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    if action in ("pause", "quit"):
        return (action, None)
    if action not in ("seek", "seek_relative", "volume_relative"):
        return None
    raw = obj.get("value")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except OverflowError:
        return None
    if not math.isfinite(value):
        return None
    if action == "seek" and 0 <= value <= 1e8:
        return (action, value)
    if action == "seek_relative" and abs(value) <= 3600:
        return (action, value)
    if action == "volume_relative" and abs(value) <= 100:
        return (action, value)
    return None


def format_event(event: str, **data: object) -> str:
    return EVENT_PREFIX + json.dumps({"event": event, **data}, ensure_ascii=False, separators=(",", ":"))


def parse_event(line: str) -> dict | None:
    if not isinstance(line, str) or not line.startswith(EVENT_PREFIX):
        return None
    try:
        obj = json.loads(line[len(EVENT_PREFIX):])
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("event"), str):
        return None
    return obj


def write_stdout(text: str) -> None:
    """Write a complete newline-terminated block without interleaving threads."""
    with _STDOUT_LOCK:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def emit_event(event: str, **data: object) -> None:
    write_stdout(format_event(event, **data))


def read_control_socket(port: int, token: str, queue_command, *, connected=None, disconnected=None) -> None:
    """Read only validated commands from the authenticated GUI connection."""
    log = logging.getLogger(__name__)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock, \
             sock.makefile("rb") as incoming:
                if connected:
                    connected(sock)
                sock.sendall(token.encode("ascii") + b"\n")
                if incoming.readline(16) != b"OK\n":
                    raise ConnectionError("GUI handshake failed")
                sock.settimeout(None)
                log.info("GUI player control_ready (loopback)")
                emit_event("control_ready")
                while line := incoming.readline(4097):
                    if len(line) > 4096 or not line.endswith(b"\n"):
                        raise ConnectionError("invalid GUI command frame")
                    command = parse_command(line.decode("utf-8", errors="replace"))
                    if command is None:
                        log.warning("GUI player rejected malformed command")
                    else:
                        queue_command(*command)
    except OSError as exc:
        log.warning("GUI player control connection failed: %s", exc)
    finally:
        if disconnected:
            disconnected()
        log.info("GUI player control_closed")
        emit_event("control_closed")

class BackendControl:
    """Own the socket and reader from early startup through backend unwind."""
    def __init__(self, port, token, stop):
        self.stop = stop
        self.player = None
        self.socket = None
        self.closed = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=read_control_socket,
            args=(port, token, self.command),
            kwargs={"connected": self._connected, "disconnected": self.request_stop},
            name="gui-command-reader", daemon=True)
        self.thread.start()

    def _connected(self, sock):
        with self.lock:
            if self.closed:
                raise ConnectionAbortedError("backend is closing")
            self.socket = sock

    def command(self, action, value):
        if action == "quit":
            self.request_stop()
        elif self.player is not None and not self.stop.is_set():
            self.player.queue_gui_command(action, value)

    def request_stop(self):
        from .windows_subprocess import stopping
        stopping.set()
        self.stop.set()

    def close(self):
        # Notify the GUI before intentionally shutting down the control socket.
        # It will use the final process exit code for the user-facing reason.
        emit_event("session_ending")
        with self.lock:
            self.closed = True
            if self.socket is not None:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.socket = None
        self.thread.join(timeout=1)
