"""Track the exact source utterances outstanding in a VOD session."""
from dataclasses import dataclass
from threading import RLock

from .session_cache import SessionSubtitleCache
from .types import Subtitle, Transcript


@dataclass
class SourceTranscript(Transcript):
    cache_ids: tuple[int, ...] = ()


@dataclass
class SourceSubtitle(Subtitle):
    cache_ids: tuple[int, ...] = ()


class VodCoverage:
    def __init__(self, cache: SessionSubtitleCache) -> None:
        self.cache = cache
        self._lock = RLock()
        self._generation: int | None = None
        self._next_id = 0
        self._pending: dict[int, tuple[float, float]] = {}
        self._retry: list[tuple[float, float]] = []

    def set_generation(self, generation: int) -> None:
        with self._lock:
            if self._generation != generation:
                self._retry.extend(self._pending.values())
                self._pending.clear()
                self._generation = generation

    def register(self, start: float, end: float, generation: int) -> tuple[int, ...]:
        with self._lock:
            if self._generation not in (None, generation) or end <= start:
                return ()
            self._next_id += 1
            self._pending[self._next_id] = (start, end)
            self.cache.record_unprocessed(start, end)
            return (self._next_id,)

    def needs_retry(self, start: float, end: float) -> bool:
        with self._lock:
            return any(left < end and start < right for left, right in self._retry)

    def resolve(self, ids: tuple[int, ...], generation: int) -> None:
        with self._lock:
            if self._generation not in (None, generation):
                return
            for key in ids:
                interval = self._pending.pop(key, None)
                if interval is not None:
                    self.cache.record_translated(*interval)

    def publish(self, sub: Subtitle, ids: tuple[int, ...]) -> None:
        with self._lock:
            if self._generation not in (None, sub.generation):
                return
            self.cache.add(sub)
            self.resolve(ids, sub.generation)

    def advance(self, start: float, end: float, generation: int) -> None:
        with self._lock:
            if self._generation not in (None, generation) or end <= start:
                return
            # A fresh pass may segment the same audio differently. Retire only
            # the revisited portion of old failures. Current failures have
            # their own registrations, including at identical timestamps.
            remaining = []
            for left, right in self._retry:
                if right <= start or left >= end:
                    remaining.append((left, right))
                    continue
                for a, b in ((left, start), (end, right)):
                    if a < b:
                        self.cache.record_unprocessed(a, b)
                        remaining.append((a, b))
                self.cache.record_translated(left, right)
            self._retry = remaining
            self.cache.record_processed(start, end)
