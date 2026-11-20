"""Output backend interface."""

from typing import Protocol


class Output(Protocol):
    """Any class with these methods counts as an output backend."""

    def show(self, subtitle: object) -> None:
        """Display a subtitle."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    def drain(self) -> None:
        """Flush any pending output."""
        ...


__all__ = ["Output"]
