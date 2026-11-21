"""Opt-in, source-backed proper-name hints for Japanese ASR.

The recognizer sees *candidates*, never fabricated transcript text. A name
repeated in accepted audio transcripts may be promoted into short-term memory.
Hints are bounded and disappear after seek/gap resets; metadata-scoped hints
remain for the current video. No external model or network request is used.
"""
from __future__ import annotations

import re
from collections import Counter, deque

from .glossary import Glossary

_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_PLAIN_NAME = re.compile(r"^[A-Z][a-zA-Z0-9-]*(?:[ -][A-Z][a-zA-Z0-9-]*)*$")


class NameFusion:
    """Bounded Japanese name candidates; no transcript or translation rewriting."""

    def __init__(
        self, glossary: Glossary, metadata_text: str, recognition_context: str,
        *, max_hints: int = 4, max_chars: int = 96, confirmations: int = 2,
    ) -> None:
        self.glossary = glossary
        self.max_hints = max_hints
        self.max_chars = max_chars
        self.confirmations = confirmations

        # Only source-backed names with known Japanese spellings and safe
        # English labels can enter memory. A free-form hallucinated name cannot
        # create an arbitrary glossary mapping.
        self._names: dict[str, str] = {
            ja: en for ja, en in glossary.terms
            if len(ja) >= 2 and _JAPANESE.search(ja) and
            _PLAIN_NAME.fullmatch(en) and len(ja) <= 24
        }

        # Title/cast have higher authority than a generic game vocabulary list.
        seeds: list[str] = []
        for ja in glossary.recognition_names(recognition_context):
            if ja in self._names and ja not in seeds:
                seeds.append(ja)
        for ja, en in glossary.find(metadata_text, context=metadata_text):
            if ja in self._names and self._names[ja] == en and ja not in seeds:
                seeds.append(ja)
        # Topic-specific ASR hints must also match named glossary entries.
        # GTA robbery terms, ordinary words and unrelated topic vocabulary
        # never enter this name-prompt channel.
        for ja in glossary.vocabulary:
            if ja in self._names and ja not in seeds:
                seeds.append(ja)

        self._seed = tuple(seeds[:max_hints])
        self._observations: Counter[str] = Counter()
        self._confirmed: deque[str] = deque(maxlen=8)

    def reset(self) -> None:
        """Forget acoustic observations after a seek or long silence gap."""
        self._observations.clear()
        self._confirmed.clear()

    def observe(self, transcript: str) -> bool:
        """Remember a spelling only after repeated accepted ASR utterances.

        The pipeline calls this only after transcript filtering/echo rejection.
        Multiple occurrences within one cue count as one observation.
        Returns whether the prompt has changed.
        """
        if not isinstance(transcript, str) or not transcript.strip():
            return False
        accepted = set()
        for ja, en in self.glossary.find(transcript, context=transcript):
            if self._names.get(ja) == en:
                accepted.add(ja)
        changed = False
        for ja in accepted:
            self._observations[ja] += 1
            if self._observations[ja] >= self.confirmations and ja not in self._confirmed:
                self._confirmed.append(ja)
                changed = True
        return changed

    def hints(self) -> tuple[str, ...]:
        """Prioritize confirmed speech, then scoped title/cast/topic candidates."""
        chosen: list[str] = []
        for name in (*reversed(self._confirmed), *self._seed):
            if name in chosen:
                continue
            candidate = "、".join((*chosen, name))
            if len(candidate) > self.max_chars:
                continue
            chosen.append(name)
            if len(chosen) >= self.max_hints:
                break
        return tuple(chosen)

    def prompt(self) -> str:
        """Short name spellings only. Never include free-form metadata or rules."""
        return "、".join(self.hints())
