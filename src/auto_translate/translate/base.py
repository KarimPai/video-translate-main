"""Translation backend interface."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Protocol


class TranslatorUnavailable(RuntimeError):
    """Translation cannot continue for this session."""


_TRANSLATION_DEADLINE: ContextVar[float | None] = ContextVar(
    "translation_deadline", default=None,
)


@contextmanager
def absolute_translation_deadline(timeout_s: float):
    with translation_deadline(time.monotonic() + timeout_s):
        yield


@contextmanager
def translation_deadline(deadline: float | None):
    current = _TRANSLATION_DEADLINE.get()
    effective = min(current, deadline) if current is not None and deadline is not None else (
        current if deadline is None else deadline
    )
    token = _TRANSLATION_DEADLINE.set(effective)
    try:
        yield
    finally:
        _TRANSLATION_DEADLINE.reset(token)


@contextmanager
def acquire_until(lock, deadline: float | None):
    acquired = False
    if deadline is None:
        lock.acquire()
        acquired = True
    else:
        while time.monotonic() < deadline:
            acquired = lock.acquire(timeout=min(deadline - time.monotonic(), 0.05))
            if acquired:
                break
    if not acquired:
        raise TimeoutError("Translation deadline exceeded while acquiring backend recovery lock")
    try:
        yield
    finally:
        lock.release()


def remaining_translation_timeout(timeout_s: float) -> float:
    deadline = _TRANSLATION_DEADLINE.get()
    if deadline is None:
        return timeout_s
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("End-to-end translation deadline exceeded")
    return min(timeout_s, remaining)


def active_translation_deadline() -> float | None:
    return _TRANSLATION_DEADLINE.get()


def start_translation_server(server) -> None:
    deadline = active_translation_deadline()
    if deadline is None:
        server.start()
    else:
        server.start(deadline=deadline)


def with_translation_deadline(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with absolute_translation_deadline(self.cfg.timeout_s):
            result = method(self, *args, **kwargs)
            remaining_translation_timeout(float("inf"))
            return result

    return wrapped


class TranslationBackend(Protocol):
    """Any class with this method counts as a translation backend.

    All translation backends translate Japanese text to English.
    """

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate one line of Japanese to English.

        Args:
            ja: Japanese text to translate.
            history: previous (ja, en) pairs, oldest first (may be ignored).

        Returns:
            English translation.
        """
        ...

    def close(self) -> None:
        """Release resources (e.g. stop the server process)."""
        ...

    def interrupt(self) -> None:
        """Interrupt any in-flight translation request."""
        ...


__all__ = [
    "TranslationBackend",
    "TranslatorUnavailable",
    "absolute_translation_deadline",
    "acquire_until",
    "active_translation_deadline",
    "remaining_translation_timeout",
    "start_translation_server",
    "translation_deadline",
    "with_translation_deadline",
]
