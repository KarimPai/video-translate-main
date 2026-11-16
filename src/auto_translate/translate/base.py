from typing import Protocol


class TranslatorUnavailable(RuntimeError):
    """Translation cannot continue for this session."""


class TranslationCancelled(RuntimeError):
    """The player stopped or sought away from an in-flight cue."""


class TranslationDeadlineExceeded(TimeoutError):
    """A complete cue exhausted its shared translation time budget."""


class TruncatedTranslation(ValueError):
    """A bounded token-limit retry still did not finish the translation."""


class Translator(Protocol):
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate one line. `history` = previous (ja, en) pairs, oldest first (may be ignored)."""
        ...
