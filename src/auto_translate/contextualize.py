"""Bounded future-context repair between Haas ASR and PLaMo translation.

All timestamps remain on the source-media timeline. The source-only PFML3
prompt is unchanged; only high-confidence Japanese transcript repairs occur.
"""
from __future__ import annotations

import logging
import queue
import re
import time
from collections import deque
from dataclasses import replace
from typing import Callable

from .types import AudioProgress, SAMPLE_RATE, Subtitle, Transcript

log = logging.getLogger(__name__)

# Evidence must be within a local context window, not inferred from a model.
_BURGER_MISHEARD = re.compile(r"防(?:御|号)キング")
_AMMO = re.compile(r"玉(?=を?(?:買|持|補充|切ら|切れ|拾|回収))")
_HIDEOUT = re.compile(r"(?<=時に)アジと(?=集合)")
_INCOMPLETE = re.compile(
    r"(?:で|から|ので|けど|けれど|ながら|とか|ちょっと|ついでに|"
    r"あの|えっと|それで|って|という|に|を|が)[、,\s]*$"
)
_SENTENCE_END = re.compile(r"[。.!?！？]$")
_WEAPON_WORDS = ("アサルト", "銃", "武器", "弾薬", "アーマー", "マガジン", "ライフル")


def repair_nearby_terms(
    japanese: str,
    recent: list[str] | tuple[str, ...] = (),
    future: list[str] | tuple[str, ...] = (),
    title: str = "",
) -> str:
    """Repair narrowly observed homophones only with independent evidence."""
    neighbors = list(recent)[-8:] + list(future)[:4]
    context = " ".join(neighbors)
    if _BURGER_MISHEARD.search(japanese) and (
        "バーガーキング" in context or "ハンバーガー" in context
    ):
        japanese = _BURGER_MISHEARD.sub("バーガーキング", japanese)
    # Do not convert general '玉' to ammunition unless weapons are discussed
    # in nearby *other* speech or explicitly in the current utterance.
    if _AMMO.search(japanese) and any(
        token in context + japanese for token in _WEAPON_WORDS
    ):
        japanese = _AMMO.sub("弾", japanese)
    # 8時にアジと集合 -> 8時にアジト集合 is safe only when the
    # hideout has been mentioned, or the stream title explicitly says GTA.
    if _HIDEOUT.search(japanese) and (
        "アジト" in context or re.search(r"(?i)gta|グラセフ", title)
    ):
        japanese = _HIDEOUT.sub("アジト", japanese)
    # A -ちゃん honorific signals a person; avoid rewriting the generic
    # 'ボタン' referring to a keyboard/controller button.
    if "ボタンちゃん" in japanese:
        japanese = japanese.replace("ボタンちゃん", "ぼたんちゃん")
    return japanese


def should_join_fragments(
    previous: Transcript, following: Transcript, *,
    max_gap_s: float = 0.25,
) -> bool:
    """Join only short explicit continuations without detected handoff.

    A false handoff flag does not prove speaker identity, so the lexical and
    time constraints are intentionally strict.
    """
    if previous.generation != following.generation or following.speaker_handoff:
        return False
    gap = following.start_s - previous.end_s
    if not (-0.12 <= gap <= max_gap_s):
        return False
    if previous.end_s - previous.start_s > 3.0:
        return False
    ja = previous.ja.strip()
    if not ja or len(ja) > 22 or len(ja) + len(following.ja) > 110:
        return False
    if _SENTENCE_END.search(ja):
        return False
    if not _INCOMPLETE.search(ja):
        return False
    if re.match(r"^(?:おはよう|こんにちは|ありがとう|もしもし|うん[。！!]?|はい[。！!]?)", following.ja.strip()):
        return False
    if following.end_s - previous.start_s > 8.0:
        return False
    return True


def merge_fragments(first: Transcript, second: Transcript) -> Transcript:
    # Use the last audio arrival as the joined line's latency origin.
    times = replace(
        first.timings,
        audio_read=max(first.timings.audio_read, second.timings.audio_read),
        asr_end=max(first.timings.asr_end, second.timings.asr_end),
    )
    return replace(
        first,
        end_s=second.end_s,
        ja=first.ja.rstrip(" 、,") + " " + second.ja.lstrip(" 、,"),
        timings=times,
    )


def run_contextualize(
    in_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    title: str = "",
    lookahead_s: float = 2.0,
    join_gap_s: float = 0.25,
    join_enabled: bool = True,
) -> None:
    """Bound context by 2s of media or wall time; never wait indefinitely.

    Progress markers release pending cues after silence. Seek markers clear
    pending work instead of leaking subtitles from the previous generation.
    """
    pending: deque[tuple[Transcript | Subtitle, float]] = deque()
    recent: deque[tuple[float, str]] = deque(maxlen=12)
    generation: int | None = None
    watermark = float("-inf")
    horizon = max(0.0, min(float(lookahead_s), 2.0))
    join_gap = max(0.0, min(float(join_gap_s), 0.35))

    def drain(force: bool = False) -> None:
        now = time.monotonic()
        while pending:
            head, arrived_at = pending[0]
            ready = (
                force or isinstance(head, Subtitle)
                or watermark >= head.end_s + horizon
                or now - arrived_at >= horizon
            )
            if not ready:
                break
            pending.popleft()
            if get_generation is not None and head.generation != get_generation():
                continue
            if isinstance(head, Subtitle):
                out_q.put(head)
                if head.ja:
                    recent.append((head.end_s, head.ja))
                continue
            if join_enabled and pending and isinstance(pending[0][0], Transcript):
                nxt, _ = pending[0]
                if should_join_fragments(head, nxt, max_gap_s=join_gap):
                    pending.popleft()
                    head = merge_fragments(head, nxt)
                    log.debug("Joined adjacent Japanese fragment at %.2fs", head.start_s)
            future = [
                item.ja for item, _ in pending
                if head.end_s <= item.start_s <= head.end_s + horizon
            ]
            nearby = [ja for end_s, ja in recent if 0 <= head.start_s - end_s <= 12.0]
            revised = repair_nearby_terms(head.ja, nearby, future, title=title)
            if revised != head.ja:
                log.debug("Nearby-context correction: %r -> %r", head.ja, revised)
                head = replace(head, ja=revised)
            out_q.put(head)
            recent.append((head.end_s, head.ja))

    while True:
        try:
            item = in_q.get(timeout=0.1)
        except queue.Empty:
            drain()
            continue
        if item is None:
            drain(force=True)
            out_q.put(None)
            return
        if generation is not None and item.generation != generation:
            pending.clear()
            recent.clear()
            watermark = float("-inf")
        generation = item.generation
        if isinstance(item, AudioProgress):
            if item.reset:
                pending.clear()
                recent.clear()
                watermark = float("-inf")
            if get_generation is not None and item.generation != get_generation():
                continue
            watermark = max(watermark, item.sample / SAMPLE_RATE)
            drain()
            # Do not advertise audio past an untranslated cue. MPV uses
            # progress to begin playing after its translation lead, and a
            # forwarded full watermark could start playback before our 2s
            # context wait completes.
            safe_sample = item.sample
            if pending:
                safe_sample = min(
                    safe_sample, max(0, int(pending[0][0].start_s * SAMPLE_RATE))
                )
            out_q.put(replace(item, sample=safe_sample))
            continue
        if get_generation is not None and item.generation != get_generation():
            continue
        watermark = max(watermark, item.end_s)
        pending.append((item, time.monotonic()))
        drain()
