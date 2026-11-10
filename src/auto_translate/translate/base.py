from typing import Protocol


class Translator(Protocol):
    def translate(
        self, ja: str, history: list[tuple[str, str]],
        *, fragment_context: tuple[str, ...] = (),
    ) -> str:
        """Translate current line only, with optional prior Japanese context."""
        ...
