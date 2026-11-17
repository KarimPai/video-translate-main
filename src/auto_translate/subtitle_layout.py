"""Subtitle readability: reading-speed timing, balanced wrapping and cue splitting."""
from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE = re.compile(r"(?<=[,;:—])\s+")
_BREAK_AFTER = (",", ";", ":", "—")
_SENTENCE_BREAK = (".", "?", "!", "…")


def _greedy(words: list[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= max_chars:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def wrap_lines(text: str, max_chars: int) -> list[str]:
    """Split into lines of at most max_chars, preferring two balanced lines."""
    text = " ".join(text.split())
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    words = text.split(" ")
    best = None
    for index in range(1, len(words)):
        first, second = " ".join(words[:index]), " ".join(words[index:])
        if len(first) > max_chars or len(second) > max_chars:
            continue
        # Balanced lines, nudged toward a break after a sentence or clause.
        bonus = 15 if first.endswith(_SENTENCE_BREAK) else 8 if first.endswith(_BREAK_AFTER) else 0
        score = max(len(first), len(second)) - bonus
        if best is None or score < best[0]:
            best = (score, [first, second])
    return best[1] if best else _greedy(words, max_chars)


def split_cue(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Break text into consecutive chunks that each fit max_lines wrapped lines."""
    text = " ".join(text.split())
    if max_chars <= 0 or not text:
        return [text] if text else []

    def fits(value: str) -> bool:
        return len(wrap_lines(value, max_chars)) <= max_lines

    if fits(text):
        return [text]
    atoms: list[str] = []
    for sentence in _SENTENCE.split(text):
        if fits(sentence):
            atoms.append(sentence)
            continue
        for clause in _CLAUSE.split(sentence):
            atoms.extend([clause] if fits(clause) else clause.split(" "))
    chunks: list[str] = []
    for atom in atoms:
        if chunks and fits(chunks[-1] + " " + atom):
            chunks[-1] += " " + atom
        else:
            chunks.append(atom)
    return chunks


def chunk_at(chunks: list[str], start: float, end: float, now: float) -> str:
    """The chunk spoken at now, with time shared in proportion to length."""
    if len(chunks) <= 1 or end <= start:
        return chunks[0] if chunks else ""
    weights = [len(chunk) + 1 for chunk in chunks]
    position = (now - start) / (end - start) * sum(weights)
    for chunk, weight in zip(chunks, weights):
        position -= weight
        if position < 0:
            return chunk
    return chunks[-1]


def display_until(text: str, onset: float, end: float, minimum_s: float,
                  cps: float, maximum_s: float) -> float:
    """Keep a cue up for its speech, the minimum, and its reading time up to a cap."""
    reading = onset + len(text) / cps if cps > 0 else onset
    return max(end, onset + minimum_s, min(reading, onset + max(maximum_s, minimum_s)))
