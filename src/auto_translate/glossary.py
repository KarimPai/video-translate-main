"""JA → EN fixed terms from glossary.toml."""

import tomllib
from functools import lru_cache
from pathlib import Path


class Glossary:
    def __init__(self, terms: dict[str, str]) -> None:
        # Longest first, so "てぇてぇ" is matched before any shorter term inside it.
        self.terms = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            return cls({})
        with path.open("rb") as f:
            return cls(tomllib.load(f).get("terms", {}))

    def find(self, ja: str) -> list[tuple[str, str]]:
        """Entries that appear in `ja` (non-overlapping, longest match wins)."""
        return list(self._matches(ja))

    @lru_cache(maxsize=128)
    def _matches(self, ja: str) -> tuple[tuple[str, str], ...]:
        hits = []
        for term, en in self.terms:
            if term in ja:
                hits.append((term, en))
                ja = ja.replace(term, "\0")  # mask it so shorter terms can't match inside
        return tuple(hits)
