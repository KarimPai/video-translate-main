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
        self._exact_replacements: set[int] = set()

    def set_generation(self, generation: int) -> None:
        with self._lock:
            if self._generation != generation:
                self._retry.update(self._pending)
                self._pending.clear()
                self._exact_replacements.clear()
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
                self._exact_replacements.add(identity)
            self._pending[identity] = (start, end, generation)
            self.cache.record_unprocessed(start, end, identity, replacing=replacing)
            return (self._next_id,)

    def needs_retry(self, start: float, end: float) -> bool:
        with self._lock:
            return any(left < end and start < right
                       for left, right, _ in self._retry.values())

    def _resolve_retry_overlap(self, start: float, end: float) -> None:
        """Retire only positively resolved audio; preserve ambiguous identities."""
        overlaps = [
            (key, left, right, gen)
            for key, (left, right, gen) in self._retry.items()
            if left < end and start < right
        ]
        # One new result cannot establish which of two overlapping failed cues
        # it represents. Keep those identities outstanding rather than guessing.
        overlaps.sort(key=lambda item: item[1])
        previous_end = float("-inf")
        for i, (key, left, right, gen) in enumerate(overlaps):
            clipped_start = max(start, left)
            clipped_end = min(end, right)
            ambiguous = (
                previous_end > clipped_start
                or (i + 1 < len(overlaps)
                    and max(start, overlaps[i + 1][1]) < clipped_end)
            )
            previous_end = max(previous_end, clipped_end)
            if ambiguous:
                continue
            # Reserve residuals before retiring the original so concurrent
            # cache readers can never observe a transient coverage gap.
            for a, b in ((left, min(start, right)), (max(end, left), right)):
                if a < b:
                    self._next_id += 1
                    identity = self._next_id
                    self._retry[identity] = (a, b, gen)
                    self.cache.record_unprocessed(a, b, identity)
            self._retry.pop(key)
            self.cache.record_translated(key)

    def resolve(self, ids: tuple[int, ...], generation: int) -> None:
        with self._lock:
            if self._generation not in (None, generation):
                return
            for key in ids:
                interval = self._pending.get(key)
                if interval is not None and interval[2] == generation:
                    self._pending.pop(key)
                    # Exact registration already replaced one old identity.
                    # Do not also resolve a second overlapping failed cue.
                    if key in self._exact_replacements:
                        self._exact_replacements.discard(key)
                    else:
                        self._resolve_retry_overlap(interval[0], interval[1])
                    self.cache.record_translated(key)

    def publish(self, sub: Subtitle, ids: tuple[int, ...]) -> None:
        with self._lock:
            if self._generation not in (None, sub.generation):
                return
            if self.cache.add(sub) or self.cache.has_equivalent(sub):
                self.resolve(ids, sub.generation)

    def resolve_duplicate(self, sub: Subtitle, ids: tuple[int, ...]) -> None:
        """Only release a suppressed duplicate if its source cue is cached."""
        with self._lock:
            if self._generation not in (None, sub.generation):
                return
            if self.cache.has_equivalent(sub):
                self.resolve(ids, sub.generation)

    def advance(self, start: float, end: float, generation: int) -> None:
        with self._lock:
            if self._generation not in (None, generation) or end <= start:
                return
            # Progress alone cannot resolve a failed utterance. Keep every
            # retry identity outstanding until that utterance is reprocessed
            # or explicitly rejected.
            self.cache.record_processed(start, end)
