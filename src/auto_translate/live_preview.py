"""Opt-in, best-effort Japanese live previews outside the subtitle pipeline.

The controller uses an existing secondary worker through a nonblocking lease.
Only the newest pending PCM snapshot is retained. Consecutive hypotheses agree
on a prefix before it is marked stable; neither draft nor stable preview text
is a finalized subtitle or enters translation history/cache.
"""
from collections import deque
from dataclasses import dataclass
import logging
import math
import threading
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewSnapshot:
    start_s: float
    end_s: float
    audio: np.ndarray
    generation: int = 0
    final: bool = False
    reset: bool = False


def agreed_prefix(hypotheses: list[str]) -> str:
    """Character agreement also works for Japanese without whitespace words."""
    if not hypotheses:
        return ""
    prefix = hypotheses[0]
    for hypothesis in hypotheses[1:]:
        stop = min(len(prefix), len(hypothesis))
        index = 0
        while index < stop and prefix[index] == hypothesis[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix.rstrip()


class LivePreview:
    """One background preview request and one replaceable pending snapshot.

    ``decode`` follows DualASR.try_preview's signature and returns None when
    final recognition owns the worker. ``emit`` must enqueue an event without
    waiting for UI/network work; it is called under the epoch lock so a clear
    event can never be overtaken by a stale decoder result.
    """

    def __init__(self, decode: Callable, emit: Callable[[dict], None], *,
                 min_agreement: int = 2, timeout_s: float = 0.25):
        if type(min_agreement) is not int or not 1 <= min_agreement <= 5:
            raise ValueError("preview min_agreement must be in 1..5")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 2:
            raise ValueError("preview timeout must be in (0, 2] seconds")
        self.decode = decode
        self.emit = emit
        self.timeout_s = timeout_s
        self.min_agreement = min_agreement
        self._condition = threading.Condition()
        self._closed = False
        self._pending: PreviewSnapshot | None = None
        self._key = None
        self._epoch = 0
        self._cancel: threading.Event | None = None
        self._hypotheses: deque[str] = deque(maxlen=min_agreement)
        self._thread = threading.Thread(target=self._run, name="live-preview", daemon=True)
        self._thread.start()

    @staticmethod
    def _event(snapshot: PreviewSnapshot, *, ja: str = "", stable: str = "", clear=False):
        return {"kind": "preview", "provisional": True,
                "generation": snapshot.generation, "start_s": snapshot.start_s,
                "end_s": snapshot.end_s, "ja": ja, "stable_ja": stable,
                "clear": clear}

    def _emit(self, event: dict) -> None:
        try:
            self.emit(event)
        except Exception:
            # An optional viewer must never stop VAD/final recognition.
            log.exception("Live preview viewer failed")

    def submit(self, snapshot: PreviewSnapshot) -> None:
        """Replace pending work immediately; boundary events cancel active work."""
        if (not math.isfinite(snapshot.start_s) or not math.isfinite(snapshot.end_s)
                or snapshot.end_s < snapshot.start_s):
            return
        with self._condition:
            if self._closed:
                return
            key = (snapshot.generation, snapshot.start_s)
            if snapshot.final or snapshot.reset or key != self._key:
                self._epoch += 1
                if self._cancel is not None:
                    self._cancel.set()
                self._hypotheses.clear()
                self._key = key
                self._pending = None
            if snapshot.final or snapshot.reset:
                self._emit(self._event(snapshot, clear=True))
            elif (snapshot.end_s > snapshot.start_s and snapshot.audio.ndim == 1
                  and len(snapshot.audio) and np.isfinite(snapshot.audio).all()):
                self._pending = snapshot
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                snapshot, self._pending = self._pending, None
                epoch = self._epoch
                cancel = threading.Event()
                self._cancel = cancel
            try:
                text = self.decode(snapshot.audio, snapshot.generation,
                                   timeout=self.timeout_s, cancel_event=cancel)
            except Exception:
                log.debug("Best-effort live preview skipped", exc_info=True)
                text = None
            with self._condition:
                if self._cancel is cancel:
                    self._cancel = None
                if self._closed or cancel.is_set() or epoch != self._epoch:
                    continue
                if not isinstance(text, str) or not text.strip():
                    continue
                text = text.strip()
                self._hypotheses.append(text)
                stable = (agreed_prefix(list(self._hypotheses))
                          if len(self._hypotheses) >= self.min_agreement else "")
                self._emit(self._event(snapshot, ja=text, stable=stable))

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            if self._cancel is not None:
                self._cancel.set()
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
