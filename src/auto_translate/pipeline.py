
"""Wire the stages together with threads and queues."""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .asr.base import ASR
from .config import Config
from .ingest import Ingest
from .output.base import Output
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

    def __init__(self, *queues):
        self.queues = queues

    def put(self, item):
        if isinstance(item, AudioProgress):
            self.queues[0].put(item)
            return

        for q in self.queues:
            q.put(item)


def run_asr(
    asr: ASR,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | AudioProgress | None]",
    diarizer=None,
    get_generation: Callable[[], int] | None = None,
) -> None:
    """Transcribe utterances, attach speaker IDs and forward progress."""
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

        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        utt.timings.asr_end = time.monotonic()

        if not text:
            continue

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
            )
        )

    out_q.put(None)


def run_translate(
    translator: Translator | None,
    context_lines: int,
    in_q: "queue.Queue[Transcript | AudioProgress | None]",
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

            if en != "[translation failed]":
                history.append((tr.ja, en))

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
            )
        )

    out_q.put(None)


@dataclass
class Stats:
    lines: int = 0
    asr_s: list[float] = field(default_factory=list)
    mt_s: list[float] = field(default_factory=list)
    total_s: list[float] = field(default_factory=list)
    max_q: dict[str, int] = field(default_factory=dict)

    def add(self, sub: Subtitle) -> None:
        tm = sub.timings
        self.lines += 1
        self.asr_s.append(tm.asr_end - tm.asr_start)
        self.mt_s.append(tm.mt_end - tm.mt_start)
        self.total_s.append(tm.mt_end - tm.audio_read)

    def summary(self) -> str:
        if not self.lines:
            return "no lines yet"

        def fmt(name: str, xs: list[float]) -> str:
            return f"{name} avg {sum(xs) / len(xs):.2f}s max {max(xs):.2f}s"

        maxq = " ".join(
            f"{k}={v}" for k, v in self.max_q.items()
        )

        return (
            f"{self.lines} lines ┬╖ "
            f"{fmt('asr', self.asr_s)} ┬╖ "
            f"{fmt('mt', self.mt_s)} ┬╖ "
            f"{fmt('latency', self.total_s)} ┬╖ "
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
) -> None:
    """Run ingestion, VAD, diarization, ASR and translation."""

    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue()
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue()
    text_q: queue.Queue[Transcript | AudioProgress | None] = queue.Queue()
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue()

    diarization_q = (
        queue.Queue() if diarizer is not None else None
    )

    # Ingest audio is copied to VAD and pyannote.
    ingest_output = (
        FanoutQueue(audio_q, diarization_q)
        if diarizer is not None
        else audio_q
    )

    queues = {
        "audio": audio_q,
        "utt": utt_q,
        "text": text_q,
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
                    for out in outputs:
                        if item.reset:
                            reset = getattr(out, "reset_buffer", None)
                            if reset is not None:
                                reset(item.sample / SAMPLE_RATE)
                        else:
                            advance = getattr(out, "advance", None)
                            if advance is not None:
                                advance(item.sample / SAMPLE_RATE)
                    continue

                if item.generation not in logged_subtitle_generations:
                    logged_subtitle_generations.add(item.generation)
                    log.info(
                        "first translated subtitle for generation=%d at %.2fs",
                        item.generation,
                        item.start_s,
                    )

                stats.add(item)

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

                for name in ("utt", "text"):
                    if queues[name].qsize() > 3:
                        log.warning(
                            "falling behind: %s queue = %d",
                            name,
                            queues[name].qsize(),
                        )

    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")

    finally:
        ingest.stop()

        for out in outputs:
            out.close()

        log.info("done: %s", stats.summary())
