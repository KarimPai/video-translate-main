"""Contiguous translation coverage tracker.

Tracks which speech intervals have been completed (translated or suppressed)
and computes the contiguous translated watermark that never advances past
an unfinished speech interval.

Design:
- Each pending speech interval has a unique ID (auto-incrementing).
- Completed intervals are stored independently from pending ones.
- The contiguous watermark is computed by sorting all intervals (pending +
  completed) and walking from the start position forward.
- The watermark stops at the start of the first pending (uncompleted) interval.
- Silence (VAD progress) can advance the watermark only through confirmed
  silence that is contiguous with the current watermark position.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechInterval:
    """A speech interval with a stable identity."""

    start_s: float
    end_s: float
    id: int  # unique within a generation


class CoverageTracker:
    """Track contiguous translation coverage over speech intervals.

    Thread-safe. All public methods acquire the internal lock.
    """

    def __init__(self, start_s: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._start_s = start_s
        self._vad_progress = start_s
        self._translated_until = start_s
        # Pending speech intervals (not yet completed)
        self._pending: dict[int, SpeechInterval] = {}
        # Completed speech intervals (translated or suppressed)
        self._completed: dict[int, SpeechInterval] = {}
        self._next_id = 1
        self._generation = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def translated_until(self) -> float:
        """Contiguous watermark: the furthest position covered by translation
        or confirmed silence from the start."""
        with self._lock:
            return self._translated_until

    @property
    def vad_progress(self) -> float:
        with self._lock:
            return self._vad_progress

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def advance_vad(self, progress_s: float) -> None:
        """Record VAD progress (silence confirmed up to this point).

        This does NOT advance the translation watermark directly;
        call _advance_through_silence (or the public methods) to do that.
        """
        with self._lock:
            self._vad_progress = max(self._vad_progress, progress_s)
            self._advance_through_silence_locked()

    def record_speech(self, start_s: float, end_s: float) -> int:
        """Record a new pending speech interval. Returns its unique ID.

        The interval blocks the watermark from advancing past `start_s`
        until it is completed.
        """
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            interval = SpeechInterval(start_s=start_s, end_s=end_s, id=sid)
            self._pending[sid] = interval
            # Adding pending speech might block the watermark — recompute
            self._advance_through_silence_locked()
            return sid

    def complete_speech(self, sid: int) -> bool:
        """Mark a pending speech interval as completed (translated or suppressed).

        Returns True if the interval was found and completed.
        Returns False if the interval was not found (already completed or unknown).
        """
        with self._lock:
            interval = self._pending.pop(sid, None)
            if interval is None:
                return False
            self._completed[sid] = interval
            self._advance_through_silence_locked()
            return True

    def complete_by_interval(self, start_s: float, end_s: float, tolerance: float = 0.5) -> bool:
        """Complete a pending speech interval by matching start and end timestamps.

        Used when the caller only has timestamps (not the ID).
        Returns True if a matching interval was found and completed.
        """
        with self._lock:
            for sid, interval in list(self._pending.items()):
                if abs(interval.start_s - start_s) < tolerance and abs(interval.end_s - end_s) < tolerance:
                    del self._pending[sid]
                    self._completed[sid] = interval
                    self._advance_through_silence_locked()
                    return True
            return False

    def reset(self, progress_s: float) -> None:
        """Reset all state for a new seek position.

        Clears pending and completed intervals, resets VAD progress and
        watermark to the new position.
        """
        with self._lock:
            self._vad_progress = progress_s
            self._translated_until = progress_s
            self._pending.clear()
            self._completed.clear()
            self._next_id = 1
            self._generation = 0

    def set_generation(self, generation: int) -> None:
        """Set the authoritative generation. Called during seek/reset."""
        with self._lock:
            self._generation = generation

    def reject_stale(self, callback_generation: int) -> bool:
        """Return True if a callback belongs to a previous generation and should be rejected."""
        with self._lock:
            return callback_generation != self._generation

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_intervals(self) -> list[tuple[float, float]]:
        """Return a list of (start, end) tuples for pending intervals."""
        with self._lock:
            return [(iv.start_s, iv.end_s) for iv in self._pending.values()]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _advance_through_silence_locked(self) -> None:
        """Recompute the contiguous watermark from all known intervals.

        Must be called with self._lock held.

        Algorithm — full recomputation from the start position:
        1. Collect all intervals (pending + completed), sorted by start time.
        2. Walk from the start position forward contiguously:
           - Completed intervals: advance past their end.
           - Pending intervals: stop at their start (can't advance further).
           - Gaps between intervals: advance through VAD-confirmed silence.
        3. The watermark is the furthest contiguous position reached.

        This is a full recomputation, not an incremental advance, so it
        correctly handles late-registered speech that retroactively blocks
        the watermark.
        """
        if not self._pending and not self._completed:
            self._translated_until = max(self._translated_until, self._vad_progress)
            return

        # Collect all intervals sorted by start time
        all_intervals: list[tuple[float, float, bool]] = []
        for interval in self._pending.values():
            all_intervals.append((interval.start_s, interval.end_s, False))
        for interval in self._completed.values():
            all_intervals.append((interval.start_s, interval.end_s, True))
        all_intervals.sort(key=lambda x: (x[0], x[1]))

        # Walk from the start position forward
        watermark = self._start_s
        i = 0
        while i < len(all_intervals):
            start_s, end_s, is_completed = all_intervals[i]

            if start_s > watermark:
                # Gap (silence) between watermark and next interval
                # Advance through VAD-confirmed silence
                silence_end = min(start_s, self._vad_progress)
                if silence_end <= watermark:
                    # VAD hasn't confirmed this silence yet — can't advance
                    break
                watermark = silence_end

            if is_completed:
                # Completed interval — advance past its end, but check if any
                # pending interval starts within this completed interval's range
                if end_s > watermark:
                    # Check for pending intervals that start between watermark and end_s
                    earliest_pending_in_range = float('inf')
                    for p_interval in self._pending.values():
                        if watermark <= p_interval.start_s < end_s:
                            earliest_pending_in_range = min(earliest_pending_in_range, p_interval.start_s)
                    if earliest_pending_in_range < float('inf'):
                        # Pending speech starts within this completed interval
                        # — stop at the pending speech start
                        watermark = earliest_pending_in_range
                        break
                    else:
                        watermark = end_s
            else:
                # Pending interval — blocks advancement at its start
                # Don't advance past it
                break

            i += 1

        # After all intervals, advance through remaining VAD-confirmed silence
        # only if no pending intervals remain ahead
        if not self._pending:
            watermark = max(watermark, self._vad_progress)
        else:
            # Check if all pending intervals are behind the watermark
            all_pending_behind = all(
                iv.end_s <= watermark + 1e-9 for iv in self._pending.values()
            )
            if all_pending_behind:
                watermark = max(watermark, self._vad_progress)

        # Use the recomputed watermark directly — it may be lower than the
        # previous value if pending speech was registered late (retroactive block).
        self._translated_until = watermark

    def mark_source_finished(self) -> None:
        """Mark that the source has finished (EOF).

        Even at EOF, pending speech still blocks the watermark.
        Only advances through VAD-confirmed silence where no pending speech blocks.
        """
        with self._lock:
            # Recompute the watermark respecting pending speech
            self._advance_through_silence_locked()


__all__ = ["CoverageTracker", "SpeechInterval"]
