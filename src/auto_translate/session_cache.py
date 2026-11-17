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

from .types import Subtitle
from .subtitle_clean import sanitize_english_output


class SessionSubtitleCache:
    """Thread-safe, unbounded (text-only) cache for one currently loaded video."""

    MATCH_TOLERANCE_S = 0.15

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[tuple[int, int], Subtitle] = {}
        self._by_start: dict[int, list[Subtitle]] = defaultdict(list)
        self._by_second: dict[int, list[Subtitle]] = defaultdict(list)
        self._processed_ranges: list[tuple[float, float]] = []

    def add(self, sub: Subtitle) -> bool:
        """Save one translated cue, keeping the first result for a timestamp."""
        cleaned = sanitize_english_output(sub.en)
        if not cleaned:
            return False
        sub = replace(sub, en=cleaned)
        if sub.end_s <= sub.start_s:
            return False
        key = (round(sub.start_s * 100), round(sub.end_s * 100))
        with self._lock:
            if key in self._entries:
                return False
            self._entries[key] = sub
            self._by_start[floor(sub.start_s)].append(sub)
            onset = sub.speech_start_s if sub.speech_start_s is not None else sub.start_s
            for second in range(floor(onset), floor(sub.end_s) + 1):
                self._by_second[second].append(sub)
            return True

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
            matched = min(
                matching,
                key=lambda sub: abs(sub.start_s - start_s) + abs(sub.end_s - end_s),
            )
            cleaned = sanitize_english_output(matched.en)
            return replace(matched, en=cleaned) if cleaned else None

    def active_at(self, now: float, minimum_s: float = 0.0) -> list[Subtitle]:
        """Return already translated cues that should be visible at this time."""
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
                    if onset <= now < max(sub.end_s, onset + minimum_s):
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

    def covered_until(self, position_s: float) -> float:
        """Return the end of cached continuous processed audio at position."""
        with self._lock:
            ranges = self._processed_ranges
            pos = bisect_left(ranges, (position_s, float("inf")))
            for i in (pos - 1, pos):
                if 0 <= i < len(ranges):
                    start_s, end_s = ranges[i]
                    if start_s - 0.02 <= position_s <= end_s + 0.02:
                        return max(position_s, end_s)
        return position_s

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
