"""Prepare caption-first Japanese cues without audio, VAD, or ASR.

YouTube automatic VTT often contains rolling copies of earlier text. This
module removes only verified overlaps and conservatively joins unfinished
adjacent phrases before sending them to the translation model.
"""
from __future__ import annotations

import re
from bisect import bisect_left

from .captions import CaptionCue, CaptionIndex
from .contextualize import merge_fragments, should_join_fragments
from .text_filter import strip_nonverbal_labels
from .types import Transcript

_SPACE = re.compile(r"\s+")
_END_PUNCT = re.compile(r"[。！？!?]$")


def _clean(text: str) -> str:
    return _SPACE.sub(" ", strip_nonverbal_labels(text)).strip()


def _rolling_suffix(previous: str, current: str) -> int:
    """Number of characters copied from the end of an overlapping cue."""
    if current.startswith(previous) and len(previous) >= 3:
        return len(previous)
    for size in range(min(len(previous), len(current)), 2, -1):
        if previous.endswith(current[:size]):
            return size
    return 0


def prepare_caption_cues(index: CaptionIndex, *, join_gap_s: float = 0.25) -> list[CaptionCue]:
    """Return timestamped speech once, preserving repeated real utterances.

    Deduplication applies only to temporal overlaps; a phrase repeated later
    in the stream is always retained.
    """
    output: list[CaptionCue] = []
    previous_raw = ""
    previous_end = -1.0
    for cue in index.cues:
        text = _clean(cue.text)
        if not text or cue.end_s <= cue.start_s:
            continue
        start = cue.start_s
        if previous_raw and start <= previous_end + 0.05:
            if text == previous_raw or previous_raw.endswith(text):
                # Same rolling caption displayed again, not a new utterance.
                previous_end = max(previous_end, cue.end_s)
                continue
            shared = _rolling_suffix(previous_raw, text)
            if shared:
                text = text[shared:].strip()
                if not text:
                    previous_end = max(previous_end, cue.end_s)
                    continue
                # Newly added text belongs after the preceding rolling cue,
                # rather than being shown at the earlier duplicate timestamp.
                start = max(start, min(previous_end, cue.end_s - 0.05))
        previous_raw = _clean(cue.text)
        previous_end = max(previous_end, cue.end_s) if start <= previous_end + 0.05 else cue.end_s
        if not text:
            continue
        new_cue = CaptionCue(start, cue.end_s, text)
        if output:
            prior = output[-1]
            if (not _END_PUNCT.search(prior.text)
                    and should_join_fragments(
                        Transcript(prior.start_s, prior.end_s, prior.text),
                        Transcript(new_cue.start_s, new_cue.end_s, new_cue.text),
                        max_gap_s=join_gap_s,
                    )):
                merged = merge_fragments(
                    Transcript(prior.start_s, prior.end_s, prior.text),
                    Transcript(new_cue.start_s, new_cue.end_s, new_cue.text),
                )
                output[-1] = CaptionCue(merged.start_s, merged.end_s, merged.ja)
                continue
        output.append(new_cue)
    return output


def near_cues(cues: list[CaptionCue], position_s: float, horizon_s: float = 30.0) -> range:
    """Indices needed at a source timeline position (including 3s linger)."""
    starts = [cue.start_s for cue in cues]
    first = bisect_left(starts, max(0.0, position_s - 3.5))
    # Preserve a cue starting earlier and still overlapping the current time.
    while first and cues[first - 1].end_s >= position_s - 3.5:
        first -= 1
    last = bisect_left(starts, position_s + max(0.0, horizon_s))
    return range(first, last)
