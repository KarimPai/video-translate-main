"""Loopback player commands and line-oriented backend status events."""
from __future__ import annotations

import json
import math
import logging
import socket


EVENT_PREFIX = "VIDEO_TRANSLATE_GUI_EVENT:"
FULLSCREEN_MESSAGE = "auto-translate-gui-fullscreen"
EXIT_FULLSCREEN_MESSAGE = "auto-translate-gui-exit-fullscreen"


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
    if action == "pause":
        return ("pause", None)
    if action not in ("seek", "seek_relative"):
        return None
    raw = obj.get("value")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None
    value = float(raw)
    if not math.isfinite(value):
        return None
    if action == "seek" and 0 <= value <= 1e8:
        return (action, value)
    if action == "seek_relative" and abs(value) <= 3600:
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


def emit_event(event: str, **data: object) -> None:
    print(format_event(event, **data), flush=True)


def read_control_socket(port: int, token: str, queue_command) -> None:
    """Read only validated commands from the authenticated GUI connection."""
    log = logging.getLogger(__name__)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            with sock.makefile("rb") as incoming:
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
        log.info("GUI player control_closed")
        emit_event("control_closed")
