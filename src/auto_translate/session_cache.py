"""RAM-only translation cache for one loaded VOD.

Keeps prior Japanese/English subtitle cues and completed audio timeline ranges.
Nothing is written to disk. A new video gets a new instance; closing the
translator drops the cache. MPV remains responsible for its separate video cache.
"""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from math import floor
from threading import RLock
from unicodedata import normalize

from .subtitle_clean import sanitize_english_output
from .types import Subtitle


def same_source_cue(first: Subtitle, second: Subtitle, tolerance_s: float = 0.5) -> bool:
    first_ja = "".join(normalize("NFKC", first.ja).split())
    second_ja = "".join(normalize("NFKC", second.ja).split())
    return (
        bool(first_ja)
        and first_ja == second_ja
        and abs(first.start_s - second.start_s) <= tolerance_s
        and abs(first.end_s - second.end_s) <= tolerance_s
    )


class SessionSubtitleCache:
    """Thread-safe, unbounded (text-only) cache for one currently loaded video."""

    MATCH_TOLERANCE_S = 0.15
    REPLACEMENT_TOLERANCE_S = 0.5

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[int, Subtitle] = {}
        self._by_start: dict[int, list[Subtitle]] = defaultdict(list)
        self._by_second: dict[int, list[Subtitle]] = defaultdict(list)
        self._processed_ranges: list[tuple[float, float]] = []
        self._unprocessed_ranges: dict[int, tuple[float, float]] = {}

    def add(self, sub: Subtitle) -> bool:
        """Store one result per source cue, replacing retranslated corrections."""
        cleaned = sanitize_english_output(sub.en)
        if not cleaned:
            return False
        sub = replace(sub, en=cleaned)
        if sub.end_s <= sub.start_s:
            return False
        with self._lock:
            candidates = []
            tolerance = self.REPLACEMENT_TOLERANCE_S
            for second in range(floor(sub.start_s - tolerance), floor(sub.start_s + tolerance) + 1):
                candidates.extend(self._by_start.get(second, ()))
            replacements = [old for old in candidates if same_source_cue(old, sub, tolerance)]
            if any(old.en == sub.en for old in replacements):
                return False
            for old in replacements:
                self._remove(old)
            self._entries[id(sub)] = sub
            self._by_start[floor(sub.start_s)].append(sub)
            onset = sub.speech_start_s if sub.speech_start_s is not None else sub.start_s
            for second in range(floor(onset), floor(sub.end_s) + 1):
                self._by_second[second].append(sub)
            return True

    def has_equivalent(self, sub: Subtitle) -> bool:
        """Check for the same source cue and validated English already cached."""
        cleaned = sanitize_english_output(sub.en)
        if not cleaned or sub.end_s <= sub.start_s:
            return False
        tolerance = self.REPLACEMENT_TOLERANCE_S
        with self._lock:
            return any(
                same_source_cue(old, sub, tolerance) and old.en == cleaned
                for second in range(floor(sub.start_s - tolerance), floor(sub.start_s + tolerance) + 1)
                for old in self._by_start.get(second, ())
            )

    def _remove(self, sub: Subtitle) -> None:
        self._entries.pop(id(sub), None)
        start = floor(sub.start_s)
        self._by_start[start] = [item for item in self._by_start[start] if item is not sub]
        onset = sub.speech_start_s if sub.speech_start_s is not None else sub.start_s
        for second in range(floor(onset), floor(sub.end_s) + 1):
            self._by_second[second] = [item for item in self._by_second[second] if item is not sub]

    def match(self, start_s: float, end_s: float) -> Subtitle | None:
        """Find a previously processed utterance with matching timestamps.

        A seek can move the first VAD boundary, so unmatched utterances still
        go through regular ASR and translation; only close matches are reused.
        """
        tolerance = self.MATCH_TOLERANCE_S
        with self._lock:
            candidates: list[Subtitle] = []
            for second in range(floor(start_s - tolerance), floor(start_s + tolerance) + 1):
                candidates.extend(self._by_start.get(second, ()))
            matching = [
                sub for sub in candidates
                if abs(sub.start_s - start_s) <= tolerance
                and abs(sub.end_s - end_s) <= tolerance
            ]
            if not matching:
                return None
            if len(matching) > 1:
                return None
            matched = min(
                matching,
                key=lambda sub: abs(sub.start_s - start_s) + abs(sub.end_s - end_s),
            )
            cleaned = sanitize_english_output(matched.en)
            return replace(matched, en=cleaned) if cleaned else None

    def active_at(self, now: float, minimum_s: float = 0.0, display_until=None) -> list[Subtitle]:
        """Return already translated cues that should be visible at this time.

        minimum_s is also the lookback; display_until(sub, onset) may extend
        a cue's end up to onset + minimum_s.
        """
        with self._lock:
            minimum_s = min(10.0, max(0.0, minimum_s))
            items = []
            seen = set()
            for second in range(floor(now - minimum_s), floor(now) + 1):
                for sub in self._by_second.get(second, ()):
                    if id(sub) in seen:
                        continue
                    seen.add(id(sub))
                    onset = sub.speech_start_s if sub.speech_start_s is not None else sub.start_s
                    until = (display_until(sub, onset) if display_until is not None
                             else max(sub.end_s, onset + minimum_s))
                    if onset <= now < until:
                        cleaned = sanitize_english_output(sub.en)
                        if cleaned:
                            items.append(replace(sub, en=cleaned))
        return sorted(items, key=lambda sub: sub.start_s)

    def record_processed(self, start_s: float, end_s: float) -> None:
        """Remember timeline audio completely processed through translation.

        Store unioned ranges so seek replays in previously translated spans
        can satisfy the startup translation-lead check without recomputation.
        """
        if end_s <= start_s:
            return
        with self._lock:
            ranges = self._processed_ranges
            pos = bisect_left(ranges, (start_s, float("-inf")))
            if pos and ranges[pos - 1][1] >= start_s - 0.02:
                pos -= 1
            new_start, new_end = start_s, end_s
            end_pos = pos
            while end_pos < len(ranges) and ranges[end_pos][0] <= new_end + 0.02:
                old_start, old_end = ranges[end_pos]
                new_start = min(new_start, old_start)
                new_end = max(new_end, old_end)
                end_pos += 1
            ranges[pos:end_pos] = [(new_start, new_end)]

    def record_unprocessed(
        self,
        start_s: float,
        end_s: float,
        identity: int,
        *,
        replacing: int | None = None,
    ) -> None:
        """Keep failed or rejected speech out of reusable processed coverage."""
        if end_s <= start_s:
            return
        with self._lock:
            if replacing is not None:
                self._unprocessed_ranges.pop(replacing, None)
            self._unprocessed_ranges[identity] = (start_s, end_s)

    def _resolve_unprocessed(self, identity: int) -> None:
        self._unprocessed_ranges.pop(identity, None)

    def record_confirmed_silence(self, identity: int) -> None:
        """Resolve one known non-speech interval; no subtitle is expected."""
        with self._lock:
            self._resolve_unprocessed(identity)

    def record_rejected(self, identity: int) -> None:
        """Resolve speech deliberately rejected by transcript quality guards."""
        with self._lock:
            self._resolve_unprocessed(identity)

    def record_translated(self, identity: int) -> None:
        """Resolve one failed interval after a valid translated cue is stored."""
        with self._lock:
            self._resolve_unprocessed(identity)

    def covered_until(self, position_s: float) -> float:
        """Return the end of cached continuous processed audio at position."""
        with self._lock:
            ranges = self._processed_ranges
            pos = bisect_left(ranges, (position_s, float("inf")))
            for i in (pos - 1, pos):
                if 0 <= i < len(ranges):
                    start_s, end_s = ranges[i]
                    if start_s - 0.02 <= position_s <= end_s + 0.02:
                        covered = max(position_s, end_s)
                        unresolved = [
                            max(position_s, start)
                            for start, end in self._unprocessed_ranges.values()
                            if end > position_s and start <= covered + 0.02
                        ]
                        return min(covered, min(unresolved)) if unresolved else covered
        return position_s

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
