"""Normalize fully downloaded Japanese YouTube captions for translation.

Prefer JSON3 event timing, but also handle rolling VTT repeats. Only merge
adjacent text with strong evidence of a truncated Japanese grammatical unit,
or the existing conservative sentence-continuation rules. Preserve the
earliest source-media onset when merging; never move captions later to hide
a source timing problem.
"""
from __future__ import annotations

import logging
import re
from bisect import bisect_left
from typing import Sequence

from .captions import CaptionCue, CaptionIndex
from .contextualize import should_join_fragments
from .text_filter import strip_nonverbal_labels
from .types import Transcript

log = logging.getLogger(__name__)

_SPACE = re.compile(r"\s+")
_END_PUNCT = re.compile(r"[。！？!?]$")
# A split verb ending is much stronger joining evidence than simple proximity.
# The following cue may already begin the next sentence, but the current
# phrase should not reach PLaMo without its grammatical ending.
_CLIPPED_ENDINGS = (
    ("わかん", ("ない", "なかった", "なく")),
    ("ござい", ("ます", "ました")),
    ("かっ", ("た", "て", "たり")),
    ("なっ", ("て", "た")),
    ("しまっ", ("た", "て")),
    ("思い", ("ます", "ました")),
    ("まし", ("た", "て")),
    ("だっ", ("た",)),
    ("でし", ("た", "て")),
    ("行っ", ("た", "て")),
    ("おはよう", ("ございます",)),
    ("なりまし", ("た",)),
    ("てい", ("ます", "ました")),
    ("しちゃ", ("った",)),
    ("ごめん", ("なさい",)),
    # Speaker transitions remain uncertain without audio. These are joined
    # only when the next cue begins with an explicit grammatical completion
    # and the existing bounded timing constraints are met.
    ("やめ", ("なさい",)),
    ("貸し", ("て",)),
)


def _clean(text: str) -> str:
    return _SPACE.sub(" ", strip_nonverbal_labels(text)).strip()


def _rolling_suffix(previous: str, current: str) -> int:
    """Shared Japanese suffix/prefix when the source cues overlap in time."""
    if current.startswith(previous) and len(previous) >= 3:
        return len(previous)
    for size in range(min(len(previous), len(current)), 2, -1):
        if previous.endswith(current[:size]):
            return size
    return 0


def _clipped_ending(previous: str, following: str) -> bool:
    left = previous.rstrip(" 、,")
    right = following.lstrip(" 、,")
    return any(
        left.endswith(ending) and any(right.startswith(suffix) for suffix in suffixes)
        for ending, suffixes in _CLIPPED_ENDINGS
    )


def _can_join(first: CaptionCue, following: CaptionCue, gap_s: float) -> bool:
    """Join true clipped grammar across rolling subtitle display intervals.

    For explicit word endings, use consecutive caption onsets, not the
    YouTube subtitle *display duration*: rolling JSON3 cues often overlap
    by several seconds. For ordinary connective words, retain the stricter
    existing end-to-start and lexical checks.
    """
    onset_gap = following.start_s - first.start_s
    display_gap = following.start_s - first.end_s
    if onset_gap < -0.05:
        return False
    if len(first.text) + len(following.text) > 160:
        return False
    if _END_PUNCT.search(first.text):
        return False
    if _clipped_ending(first.text, following.text):
        # A JSON3 event commonly covers several seconds of rolling text.
        # The full-file transcript lets us join a strongly evidenced ending
        # even when adjacent event starts are >4s apart. Never join across
        # a long pause, unrelated cue, or finished sentence.
        return (
            onset_gap <= 5.5
            and display_gap <= 2.5
            and following.end_s - first.start_s <= 17.0
        )
    if following.end_s - first.start_s > 8.0:
        return False
    if not (-0.12 <= display_gap <= max(0.25, gap_s)):
        return False
    return should_join_fragments(
        Transcript(first.start_s, first.end_s, first.text),
        Transcript(following.start_s, following.end_s, following.text),
        max_gap_s=gap_s,
    )


def prepare_caption_cues(index: CaptionIndex, *, join_gap_s: float = 0.25) -> list[CaptionCue]:
    """Reconstruct the entire track once, retaining genuine repeated speech.

    An overlapping expanded rolling caption replaces the shorter earlier cue
    at its ORIGINAL onset; it does not produce a delayed second cue. If
    rolling text is too long to merge, only the genuinely new suffix remains.
    Explicit split grammar endings are joined without inserting spaces.
    """
    output: list[CaptionCue] = []
    previous_raw = ""
    previous_end = -1.0
    rolling_repeats = 0
    fragment_joins = 0
    skipped_clipped = 0

    for cue in index.cues:
        text = _clean(cue.text)
        if not text or cue.end_s <= cue.start_s:
            continue
        start = cue.start_s
        # Equal boundaries can mean two genuinely repeated utterances.
        # Rolling duplication requires actual time overlap, not adjacency.
        overlapping = bool(previous_raw) and start < previous_end - 0.03
        if overlapping:
            if text == previous_raw or previous_raw.endswith(text):
                # Duplicate rolling text, not another spoken occurrence.
                previous_end = max(previous_end, cue.end_s)
                if output and output[-1].text == previous_raw:
                    old = output[-1]
                    output[-1] = CaptionCue(
                        old.start_s, max(old.end_s, cue.end_s), old.text
                    )
                rolling_repeats += 1
                continue
            shared = _rolling_suffix(previous_raw, text)
            if shared:
                prior = output[-1] if output else None
                if (
                    prior is not None
                    and previous_raw == prior.text
                    and text.startswith(previous_raw)
                    and len(text) <= 130
                    and cue.end_s - prior.start_s <= 8.0
                ):
                    output[-1] = CaptionCue(
                        prior.start_s, max(prior.end_s, cue.end_s), text
                    )
                    previous_raw = text
                    previous_end = max(previous_end, cue.end_s)
                    rolling_repeats += 1
                    continue
                text = text[shared:].strip()
                if not text:
                    previous_end = max(previous_end, cue.end_s)
                    rolling_repeats += 1
                    continue
                # Do not shift newly appearing words to the old cue end. If
                # their individual timings are missing, keep source onset.
                rolling_repeats += 1

        # Track the unmodified rolling state, not only the extracted suffix.
        if overlapping:
            previous_end = max(previous_end, cue.end_s)
        else:
            previous_end = cue.end_s
        previous_raw = _clean(cue.text)

        next_cue = CaptionCue(start, cue.end_s, text)
        if output and _can_join(output[-1], next_cue, join_gap_s):
            prior = output[-1]
            # Japanese grammatical fragments concatenate without a space.
            # Do not defer subtitle start to the newly seen ending's time.
            combined = prior.text.rstrip(" 、,") + next_cue.text.lstrip(" 、,")
            output[-1] = CaptionCue(
                prior.start_s, max(prior.end_s, next_cue.end_s), combined
            )
            fragment_joins += 1
        else:
            if (
                output
                and _clipped_ending(output[-1].text, next_cue.text)
                and skipped_clipped < 12
            ):
                skipped_clipped += 1
                prev = output[-1]
                log.info(
                    "Caption split not joined: %.2f-%.2f %r => %.2f-%.2f %r"
                    " (onset_gap=%.2fs display_gap=%.2fs)",
                    prev.start_s, prev.end_s, prev.text[-32:],
                    next_cue.start_s, next_cue.end_s, next_cue.text[:32],
                    next_cue.start_s - prev.start_s,
                    next_cue.start_s - prev.end_s,
                )
            output.append(next_cue)

    log.info(
        "Japanese captions reconstructed: source=%d output=%d "
        "rolling=%d joined=%d unresolved_clipped_logged=%d",
        len(index.cues), len(output), rolling_repeats, fragment_joins,
        skipped_clipped,
    )
    return output


def near_cues(
    cues: list[CaptionCue], position_s: float, horizon_s: float = 30.0,
    *, starts: Sequence[float] | None = None,
) -> range:
    """Indices needed at a source timeline position (including 3s linger)."""
    if starts is None:
        starts = [cue.start_s for cue in cues]
    first = bisect_left(starts, max(0.0, position_s - 3.5))
    # Preserve a cue starting earlier and still overlapping the current time.
    while first and cues[first - 1].end_s >= position_s - 3.5:
        first -= 1
    last = bisect_left(starts, position_s + max(0.0, horizon_s))
    return range(first, last)
