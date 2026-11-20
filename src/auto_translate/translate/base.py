"""Translation backend interface."""

from typing import Protocol


class TranslatorUnavailable(RuntimeError):
    """Translation cannot continue for this session."""


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


__all__ = ["TranslationBackend", "TranslatorUnavailable"]
