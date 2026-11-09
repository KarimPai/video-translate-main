"""Bounded, seek-safe context for direct Whisper speech translation.

Clean accepted English can optionally bias decoding. A tiny rolling audio
tail is used only on uncertain retries; any output from overlapping audio
must pass strict text-overlap removal before it can reach the subtitle track.
"""
from __future__ import annotations

import re

import numpy as np

from .types import SAMPLE_RATE, Utterance


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group().casefold(), match.start(), match.end())
        for match in re.finditer(r"[\w]+(?:'[\w]+)?", text)
    ]


def strip_context_echo(previous: str, candidate: str) -> str | None:
    """Return only new English after exact suffix/prefix word agreement.

    None signals that a contextual decode could not be safely separated.
    A context-only return of the previous sentence is also rejected.
    """
    prev_words = _tokens(previous)
    new_words = _tokens(candidate)
    # Two matching words is the safe minimum; a single 'yes' is ambiguous.
    for size in range(min(24, len(prev_words), len(new_words)), 1, -1):
        if ([word[0] for word in prev_words[-size:]] ==
                [word[0] for word in new_words[:size]]):
            if size == len(new_words):
                return None
            new_start = new_words[size][1]
            remainder = candidate[new_start:].lstrip(" \t\n.,!?;:-").rstrip()
            return remainder or None
    return None


class RecentSpeechContext:
    """Tracks only a few accepted subtitles and the latest utterance audio."""

    def __init__(self, cfg: object | None) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.generation: int | None = None
        self.end_s: float | None = None
        self.last_text = ""
        self.tail = np.empty(0, dtype=np.float32)
        self.lines: list[str] = []

    def _continuation(self, utt: Utterance, gap_limit: float) -> bool:
        if (self.generation != utt.generation or self.end_s is None or
                utt.speaker_handoff):
            return False
        gap = utt.start_s - self.end_s
        return -0.12 <= gap <= gap_limit

    def prepare(self, utt: Utterance) -> None:
        """Reset only for genuinely unrelated speech or a seek generation."""
        max_gap = max(
            float(getattr(self.cfg, "audio_context_max_gap_s", 2.0)),
            float(getattr(self.cfg, "text_context_max_gap_s", 5.0)),
        )
        if self.end_s is not None and not self._continuation(utt, max_gap):
            self.reset()

    def audio_context(self, utt: Utterance) -> np.ndarray | None:
        if not getattr(self.cfg, "audio_context_enabled", False):
            return None
        if not self._continuation(
            utt, float(getattr(self.cfg, "audio_context_max_gap_s", 2.0))
        ):
            return None
        if not self.last_text or self.tail.size == 0:
            return None
        return self.tail

    def prompt(self, utt: Utterance) -> str | None:
        if not getattr(self.cfg, "use_text_context", False):
            return None
        if not self._continuation(
            utt, float(getattr(self.cfg, "text_context_max_gap_s", 5.0))
        ):
            return None
        max_lines = max(0, min(2, int(getattr(self.cfg, "text_context_lines", 2))))
        max_chars = max(0, min(160, int(getattr(self.cfg, "text_context_chars", 120))))
        if not max_lines or not max_chars:
            return None
        return " ".join(self.lines[-max_lines:])[-max_chars:] or None

    def hotwords(self) -> str | None:
        manual = getattr(self.cfg, "hotwords", ()) or ()
        detected = getattr(self.cfg, "hololive_detected_names", ()) or ()
        if not isinstance(manual, (list, tuple)) or not isinstance(detected, (list, tuple)):
            return None
        terms = [*manual, *detected]
        limit = max(0, min(6, int(getattr(self.cfg, "hololive_name_limit", 6))))
        if limit == 0:
            return None
        valid: list[str] = []
        seen: set[str] = set()
        for item in terms:
            if not isinstance(item, str) or "\n" in item or "\r" in item:
                continue
            text = " ".join(item.split())
            # English words/names only; no long instruction-like prompts.
            if (2 <= len(text) <= 36 and len(text.split()) <= 2 and
                    re.fullmatch(r"[A-Za-z][A-Za-z0-9 '\-]*", text)):
                if text.casefold() not in seen:
                    seen.add(text.casefold())
                    valid.append(text)
            if len(valid) == limit:
                break
        return ", ".join(valid) or None

    def accept(self, utt: Utterance, english: str) -> None:
        self.generation = utt.generation
        self.end_s = utt.end_s
        self.last_text = english
        self.lines.append(english[:120])
        self.lines = self.lines[-2:]
        if getattr(self.cfg, "audio_context_enabled", False):
            seconds = min(
                1.5, max(0.0, float(getattr(self.cfg, "audio_context_s", 1.2)))
            )
            count = max(0, round(seconds * SAMPLE_RATE))
            self.tail = np.asarray(utt.audio[-count:], dtype=np.float32).copy() if count else np.empty(0, dtype=np.float32)
        else:
            self.tail = np.empty(0, dtype=np.float32)
