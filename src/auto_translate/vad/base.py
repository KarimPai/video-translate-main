"""VAD backend interface."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class VADSegment:
    """A speech segment detected by VAD."""
    start_s: float
    end_s: float
    audio: object  # numpy array, float32 mono 16 kHz
    confidence: float | None = None


class VADBackend(Protocol):
    """Any class with these methods counts as a VAD backend."""

    def feed(self, audio) -> list[VADSegment]:
        """Feed audio samples and return completed segments.

        Args:
            audio: float32 mono 16 kHz numpy array.

        Returns:
            List of completed speech segments.
        """
        ...

    def flush(self) -> list[VADSegment]:
        """Flush remaining audio as a segment if any."""
        ...

    def reset(self) -> None:
        """Reset internal state (e.g. after seek)."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


__all__ = ["VADBackend", "VADSegment"]
