
"""Wire the stages together with threads and queues."""

import logging
import queue
import threading
import time
from difflib import SequenceMatcher
from collections import deque
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Callable

from .asr.base import ASR
from .watame import correct_asr_name, reconcile_caption
from .captions import CaptionIndex
from .text_filter import is_usable_transcript
from .config import Config
from .ingest import Ingest
from .output.base import Output
from .session_cache import SessionSubtitleCache
from .translate.base import Translator
from .types import (
    SAMPLE_RATE,
    AudioChunk,
    AudioProgress,
    Subtitle,
    Transcript,
    Utterance,
)
from .vad import run_segmenter

log = logging.getLogger(__name__)


class FanoutQueue:
    """Send audio to VAD and pyannote, but progress markers only to VAD."""

    def __init__(self, *queues, secondary_active=None):
        self.queues = queues
        self.secondary_active = secondary_active

    def put(self, item):
        if isinstance(item, AudioProgress):
            self.queues[0].put(item)
            return

        for index, q in enumerate(self.queues):
            while True:
                try:
                    q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    # A failed diarizer exits its consumer thread. Do not let
                    # its now-unconsumed bounded queue stall the VAD pipeline.
                    if (
                        index > 0
                        and self.secondary_active is not None
                        and not self.secondary_active()
                    ):
                        break


def run_asr(
    asr: ASR,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    diarizer=None,
    get_generation: Callable[[], int] | None = None,
    captions: CaptionIndex | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Transcribe utterances, optionally using aligned captions as a fallback."""
    while True:
        item = in_q.get()

        if item is None:
            break

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            out_q.put(item)
            continue

        utt = item

        if get_generation is not None and utt.generation != get_generation():
            continue

        # Reuse complete Japanese+English results when a backward seek
        # revisits an utterance with matching VAD boundaries.
        cached = (
            session_cache.match(utt.start_s, utt.end_s)
            if session_cache is not None else None
        )
        if cached is not None:
            if get_generation is not None and utt.generation != get_generation():
                continue
            log.debug(
                "reusing translated VOD cue %.2f-%.2fs after seek",
                cached.start_s, cached.end_s,
            )
            out_q.put(replace(
                cached,
                generation=utt.generation,
                timings=utt.timings,
            ))
            continue

        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        utt.timings.asr_end = time.monotonic()

        caption_match = (
            captions.match(utt.start_s, utt.end_s)
            if captions is not None
            else None
        )
        if text and caption_match is not None:
            if caption_match.reliable:
                text = reconcile_caption(text, caption_match.text)
            haas_compact = "".join(ch for ch in text if ch.isalnum()).casefold()
            caption_compact = "".join(ch for ch in caption_match.text if ch.isalnum()).casefold()
            similarity = SequenceMatcher(
                None, haas_compact, caption_compact
            ).ratio() if haas_compact and caption_compact else 0.0
            if similarity < 0.55:
                log.debug(
                    "Haas/caption disagreement at %.2f-%.2fs "
                    "(similarity=%.2f): Haas=%r captions=%r",
                    utt.start_s, utt.end_s, similarity, text, caption_match.text,
                )

        if (
            not text and caption_match is not None and caption_match.reliable
            and is_usable_transcript(caption_match.text)
        ):
            text = caption_match.text
            log.info(
                "using aligned YouTube caption because Haas returned no text "
                "(%.2f-%.2fs, coverage=%.0f%%): %s",
                utt.start_s, utt.end_s, caption_match.coverage * 100, text,
            )

        if not text:
            continue

        # Proper names must be repaired before translation and glossary lookup.
        # The VOD title is a stronger signal than kanji auto-caption spelling.
        corrected = correct_asr_name(
            text,
            vod_title=vod_title,
            caption=caption_match.text if caption_match else "",
        )
        if corrected != text:
            log.info("Watame ASR name correction: %s -> %s", text, corrected)
            text = corrected

        if get_generation is not None and utt.generation != get_generation():
            continue

        speaker_id = (
            diarizer.identify(
                utt.start_s,
                utt.end_s,
                generation=utt.generation,
            )
            if diarizer is not None
            else ""
        )

        if get_generation is not None and utt.generation != get_generation():
            continue

        out_q.put(
            Transcript(
                utt.start_s,
                utt.end_s,
                text,
                utt.timings,
                speaker_id,
                generation=utt.generation,
                speech_start_s=utt.speech_start_s,
            )
        )

    out_q.put(None)


def run_translate(
    translator: Translator | None,
    context_lines: int,
    in_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
) -> None:
    """Translate transcripts while retaining recent context and forwarding progress."""
    history: deque[tuple[str, str]] = deque(maxlen=context_lines)

    while True:
        item = in_q.get()

        if item is None:
            break

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            if item.reset:
                history.clear()

            out_q.put(item)
            continue

        tr = item

        if get_generation is not None and tr.generation != get_generation():
            continue

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not call PLaMo again for the same VOD cue.
        if isinstance(tr, Subtitle):
            if tr.en and tr.en != "[translation failed]":
                history.append((tr.ja, tr.en))
            tr.timings.mt_start = time.monotonic()
            tr.timings.mt_end = tr.timings.mt_start
            out_q.put(tr)
            continue

        tr.timings.mt_start = time.monotonic()
        en = ""

        if translator is not None:
            try:
                en = translator.translate(tr.ja, list(history))
            except Exception as e:
                log.error("translation failed: %s", e)
                en = "[translation failed]"

            if get_generation is not None and tr.generation != get_generation():
                continue

            if en and en != "[translation failed]":
                history.append((tr.ja, en))

            if not en.strip():
                log.error(
                    "translator returned an empty English result "
                    "(generation=%d, %.2f-%.2fs, Japanese=%r)",
                    tr.generation,
                    tr.start_s,
                    tr.end_s,
                    tr.ja,
                )

        tr.timings.mt_end = time.monotonic()

        out_q.put(
            Subtitle(
                tr.start_s,
                tr.end_s,
                tr.ja,
                en,
                tr.timings,
                tr.speaker_id,
                generation=tr.generation,
                speech_start_s=tr.speech_start_s,
            )
        )

    out_q.put(None)


@dataclass
class Stats:
    """Constant-memory, whole-run timing aggregates."""
    lines: int = 0
    asr_sum_s: float = 0.0
    mt_sum_s: float = 0.0
    total_sum_s: float = 0.0
    asr_max_s: float = 0.0
    mt_max_s: float = 0.0
    total_max_s: float = 0.0
    max_q: dict[str, int] = field(default_factory=dict)

    def add(self, sub: Subtitle) -> None:
        tm = sub.timings
        asr_s = max(0.0, tm.asr_end - tm.asr_start)
        mt_s = max(0.0, tm.mt_end - tm.mt_start)
        total_s = max(0.0, tm.mt_end - tm.audio_read)
        self.lines += 1
        self.asr_sum_s += asr_s
        self.mt_sum_s += mt_s
        self.total_sum_s += total_s
        self.asr_max_s = max(self.asr_max_s, asr_s)
        self.mt_max_s = max(self.mt_max_s, mt_s)
        self.total_max_s = max(self.total_max_s, total_s)

    def summary(self) -> str:
        if not self.lines:
            return "no lines yet"

        def fmt(name: str, total: float, maximum: float) -> str:
            return (
                f"{name} avg {total / self.lines:.2f}s "
                f"max {maximum:.2f}s"
            )

        maxq = " ".join(
            f"{k}={v}" for k, v in self.max_q.items()
        )

        return (
            f"{self.lines} lines ┬╖ "
            f"{fmt('asr', self.asr_sum_s, self.asr_max_s)} ┬╖ "
            f"{fmt('mt', self.mt_sum_s, self.mt_max_s)} ┬╖ "
            f"{fmt('latency', self.total_sum_s, self.total_max_s)} ┬╖ "
            f"max queue {maxq}"
        )


def run(
    ingest: Ingest,
    cfg: Config,
    asr: ASR,
    translator: Translator | None,
    outputs: list[Output],
    stop: threading.Event,
    diarizer=None,
    captions: CaptionIndex | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Run ingestion, VAD, diarization, ASR and translation."""

    # Bounded queues cap queued audio/work during slow ASR, translation, or
    # diarization. Blocking puts apply backpressure upstream instead of
    # allowing memory use and subtitle age to grow without limit.
    # 64 audio chunks ~= 16.4 s at 4096 samples / 16 kHz; work queues hold
    # at most 16 complete utterances/transcripts/subtitles each.
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    text_q: queue.Queue[Transcript | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)

    diarization_q = (
        queue.Queue(maxsize=64) if diarizer is not None else None
    )

    # Ingest audio is copied to VAD and pyannote.
    ingest_output = (
        FanoutQueue(
            audio_q,
            diarization_q,
            secondary_active=lambda: not diarizer.finished and not diarizer.failed,
        )
        if diarizer is not None
        else audio_q
    )

    queues = {
        "audio": audio_q,
        "utt": utt_q,
        "text": text_q,
        "subtitle": sub_q,
    }

    if diarization_q is not None:
        queues["diarization"] = diarization_q

    threads = [
        threading.Thread(
            target=ingest.run,
            args=(ingest_output,),
            name="ingest",
            daemon=True,
        ),
        threading.Thread(
            target=run_segmenter,
            args=(cfg.vad, audio_q, utt_q),
            name="vad",
            daemon=True,
        ),
    ]

    # Pyannote processes continuous audio independently.
    if diarizer is not None:
        threads.append(
            threading.Thread(
                target=diarizer.run,
                args=(diarization_q, lambda: ingest.generation),
                name="diarization",
                daemon=True,
            )
        )

    threads.extend([
        threading.Thread(
            target=run_asr,
            args=(
                asr,
                utt_q,
                text_q,
                diarizer,
                lambda: ingest.generation,
                captions,
                vod_title,
                session_cache,
            ),
            name="asr",
            daemon=True,
        ),
        threading.Thread(
            target=run_translate,
            args=(
                translator,
                cfg.translate.context_lines,
                text_q,
                sub_q,
                lambda: ingest.generation,
            ),
            name="translate",
            daemon=True,
        ),
    ])

    for thread in threads:
        thread.start()

    stats = Stats()
    logged_subtitle_generations: set[int] = set()
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s

    try:
        while not stop.is_set():
            try:
                item = sub_q.get(timeout=0.5)

                if item is None:
                    for out in outputs:
                        finish = getattr(out, "finish", None)
                        if finish is not None:
                            finish()

                    for out in outputs:
                        out.drain()
                    break

                # Discard subtitles and progress from before the latest seek.
                if item.generation != ingest.generation:
                    continue

                if isinstance(item, AudioProgress):
                    progress_s = item.sample / SAMPLE_RATE
                    if item.reset:
                        processed_generation = item.generation
                        processed_from = progress_s
                    elif session_cache is not None:
                        if processed_generation != item.generation:
                            # Reset marker not yet delivered: don't invent a
                            # processed range across a discontinuous seek.
                            processed_generation = item.generation
                            processed_from = progress_s
                        session_cache.record_processed(processed_from, progress_s)
                        processed_from = progress_s
                    for out in outputs:
                        if item.reset:
                            reset = getattr(out, "reset_buffer", None)
                            if reset is not None:
                                reset(progress_s)
                        else:
                            advance = getattr(out, "advance", None)
                            if advance is not None:
                                advance(progress_s)
                    continue

                if item.generation not in logged_subtitle_generations:
                    logged_subtitle_generations.add(item.generation)
                    log.info(
                        "first translated subtitle for generation=%d at %.2fs",
                        item.generation,
                        item.start_s,
                    )

                stats.add(item)

                if session_cache is not None:
                    session_cache.add(item)

                for out in outputs:
                    out.show(item)

            except queue.Empty:
                pass

            for name, q in queues.items():
                stats.max_q[name] = max(
                    stats.max_q.get(name, 0),
                    q.qsize(),
                )

            now = time.monotonic()

            if now >= next_stats:
                next_stats = now + cfg.stats_interval_s

                depths = " ".join(
                    f"{name}={q.qsize()}"
                    for name, q in queues.items()
                )

                log.info(
                    "[%.0fs] queues %s | %s",
                    now - started,
                    depths,
                    stats.summary(),
                )

                # Warn once per stats interval when a bounded queue is
                # nearing capacity. Producers block on full queues, applying
                # backpressure rather than retaining unlimited stale work.
                for name, q in queues.items():
                    if q.maxsize and q.qsize() >= max(1, int(q.maxsize * 0.75)):
                        log.warning(
                            "pipeline backpressure: %s queue = %d/%d",
                            name,
                            q.qsize(),
                            q.maxsize,
                        )


    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")

    finally:
        # Close MPV before deleting its temporary fMP4 cache files. On Windows,
        # MPV may still hold segment files open until its process has exited.
        for out in outputs:
            out.close()

        ingest.stop()
        log.info("done: %s", stats.summary())
