from typing import Protocol


class TranslatorUnavailable(RuntimeError):
    """Translation cannot continue for this session."""


class Translator(Protocol):
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate one line. `history` = previous (ja, en) pairs, oldest first (may be ignored)."""
        ...
