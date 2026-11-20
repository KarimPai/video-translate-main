"""ASR backend interface and factory."""

from typing import Protocol


class ASRBackend(Protocol):
    """Any class with this method counts as an ASR backend.

    All ASR backends transcribe 16 kHz mono float32 audio to Japanese text.
    """

    def transcribe(self, audio, probe: bool = False) -> str:
        """Transcribe audio to Japanese text.

        Args:
            audio: float32 mono 16 kHz numpy array.
            probe: the VAD heard no speech; only keep confident results.

        Returns:
            Japanese text, or "" if nothing usable.
        """
        ...

    def cancel_pending(self) -> None:
        """Cancel any in-flight transcription request."""
        ...

    def close(self) -> None:
        """Release resources (e.g. stop the server process)."""
        ...

    @property
    def last_rejection(self) -> str | None:
        """Reason the last utterance was rejected, or None if accepted."""
        ...

    @property
    def last_raw_transcript(self) -> str:
        """Raw ASR output before quality filtering."""
        ...


__all__ = ["ASRBackend"]
