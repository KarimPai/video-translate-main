"""JA → EN fixed terms from glossary.toml."""

import tomllib
import re
import logging
from functools import lru_cache
from pathlib import Path


class Glossary:
    def __init__(self, terms: dict[str, str], *, scene_pairs=(), name_guards=()) -> None:
        # Longest first, so "てぇてぇ" is matched before any shorter term inside it.
        self.terms = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
        # Optional ASR-only rules: acoustic peer comparison, not inferred words.
        self.scene_pairs = tuple(
            (rule["heard"], rule["correct"], tuple(rule.get("anchors", ())))
            for rule in scene_pairs
        )
        self.name_guards = tuple(
            (re.compile(rule["full"]), re.compile(rule["mention"]))
            for rule in name_guards
        )

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            logging.getLogger(__name__).warning("Glossary missing: %s", path)
            return cls({})
        with path.open("rb") as f:
            data = tomllib.load(f)
            rules = data.get("asr_rules", {})
            return cls(data.get("terms", {}),
                       scene_pairs=rules.get("scene_pairs", ()),
                       name_guards=rules.get("name_guards", ()))

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
