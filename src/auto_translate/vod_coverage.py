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
        self._pending: dict[int, tuple[float, float, int]] = {}
        self._retry: dict[int, tuple[float, float, int]] = {}

    def set_generation(self, generation: int) -> None:
        with self._lock:
            if self._generation != generation:
                self._retry.update(self._pending)
                self._pending.clear()
                self._generation = generation

    def register(self, start: float, end: float, generation: int) -> tuple[int, ...]:
        with self._lock:
            if self._generation not in (None, generation) or end <= start:
                return ()
            self._next_id += 1
            identity = self._next_id
            replacing = next(
                (key for key, (left, right, _) in self._retry.items()
                 if left == start and right == end),
                None,
            )
            if replacing is not None:
                self._retry.pop(replacing)
            self._pending[identity] = (start, end, generation)
            self.cache.record_unprocessed(start, end, identity, replacing=replacing)
            return (self._next_id,)

    def needs_retry(self, start: float, end: float) -> bool:
        with self._lock:
            return any(left < end and start < right
                       for left, right, _ in self._retry.values())

    def resolve(self, ids: tuple[int, ...], generation: int) -> None:
        with self._lock:
            if self._generation not in (None, generation):
                return
            for key in ids:
                interval = self._pending.get(key)
                if interval is not None and interval[2] == generation:
                    self._pending.pop(key)
                    self.cache.record_translated(key)

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
            # Progress alone cannot resolve a failed utterance. Keep every
            # retry identity outstanding until that utterance is reprocessed
            # or explicitly rejected.
            self.cache.record_processed(start, end)
