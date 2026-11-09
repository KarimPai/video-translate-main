
"""Wire the stages together with threads and queues."""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Callable

from .asr.base import ASR
from .asr_retry import choose_hypothesis
from .watame import correct_asr_name, reconcile_caption
from .stream_context import (
    repair_peko_nickname, repair_verified_caption, repair_ark_spelling,
    repair_gta_carjacking, repair_horse_racing_spelling,
)
from .captions import CaptionIndex, PendingCaptions, prefer_youtube_japanese
from .context_memory import repair_near_repeated_word
from .text_filter import (
    is_isolated_reaction, is_nonspeech_probe, is_sound_caption_artifact,
    is_usable_transcript, strip_nonverbal_labels, rejection_reason,
    is_short_acknowledgement, is_stationary_mechanical_audio,
    caption_confirms_short_acknowledgement,
)
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
from .speaker import run_speaker
from .contextualize import run_contextualize

log = logging.getLogger(__name__)


def _guard_stage(
    name: str,
    target: Callable,
    failures: "queue.Queue[tuple[str, BaseException]]",
    *args,
) -> None:
    """Report background stage errors to the main thread, never hide them."""
    try:
        target(*args)
    except Exception as exc:
        log.exception("pipeline stage %s crashed", name)
        failures.put((name, exc))


def _raise_stage_failure(failures: "queue.Queue[tuple[str, BaseException]]") -> None:
    try:
        name, exc = failures.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"pipeline stage '{name}' failed: {exc}") from exc


def run_asr(
    asr: ASR,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    captions: CaptionIndex | PendingCaptions | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
    live: bool = False,
) -> None:
    """Decode with Haas; prefer strongly aligned YouTube Japanese captions for VODs."""
    recent_japanese: deque[str] = deque(maxlen=12)
    last_generation: int | None = None
    empty_voiced_streak = 0
    caption_primary_logged = False
    while True:
        item = in_q.get()

        if item is None:
            break

        if isinstance(item, AudioProgress):
            if item.reset or (last_generation is not None and item.generation != last_generation):
                recent_japanese.clear()
            last_generation = item.generation
            if get_generation is not None and item.generation != get_generation():
                continue

            out_q.put(item)
            continue

        utt = item
        if last_generation is not None and utt.generation != last_generation:
            recent_japanese.clear()
        last_generation = utt.generation

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
                speaker_handoff=(utt.speaker_handoff or cached.speaker_handoff),
            ))
            continue

        if utt.probe and is_nonspeech_probe(utt.audio):
            continue

        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        utt.timings.asr_end = time.monotonic()
        # Read captions before retry selection. Only strongly aligned cues
        # are allowed to influence candidate scoring; normal reconciliation
        # and caption fallback still run afterward.
        caption_match = (
            captions.match(utt.start_s, utt.end_s)
            if captions is not None
            else None
        )
        text = choose_hypothesis(
            asr, utt.audio, text,
            start_s=utt.start_s,
            end_s=utt.end_s,
            speech_start_s=utt.speech_start_s,
            probe=utt.probe,
            caption=caption_match,
            live=live,
            cancelled=(
                (lambda: utt.generation != get_generation())
                if get_generation is not None else None
            ),
        )
        utt.timings.asr_end = time.monotonic()
        original_asr = text
        text = strip_nonverbal_labels(text)
        if original_asr != text:
            log.debug("filtered ASR nonverbal labels at %.2fs: %r -> %r", utt.start_s, original_asr, text)

        if caption_match is not None and not caption_match.reliable:
            caption_match = None
        if caption_match is not None and not caption_match.text and is_sound_caption_artifact(text):
            log.debug("Discarding likely ASR hallucination over sound-only caption at %.2fs", utt.start_s)
            continue

        source = "haas_asr"
        if text and caption_match is not None:
            # Keep legacy evidence-based name / one-character spelling repairs
            # even when whole-caption priority is not supported by alignment.
            reconciled = reconcile_caption(text, caption_match.text)
            if caption_match.coverage >= 0.80 and caption_match.overlap_s >= 0.45:
                reconciled = repair_verified_caption(reconciled, caption_match.text)
            if reconciled != text:
                source = "haas_asr+youtube_correction"
                text = reconciled

        caption_first = getattr(
            getattr(asr, "cfg", None), "youtube_captions_first", True,
        )
        if caption_first and caption_match is not None:
            preferred, evidence = prefer_youtube_japanese(text, caption_match)
            if preferred != text:
                log.debug(
                    "YouTube-first Japanese at %.2f-%.2fs (%s, coverage=%.0f%%): %r -> %r",
                    utt.start_s, utt.end_s, evidence,
                    caption_match.coverage * 100, text[:100], preferred[:100],
                )
            text = preferred
            if evidence.startswith("youtube_primary"):
                source = evidence
                if not caption_primary_logged:
                    log.info(
                        "YouTube-first Japanese subtitles active: "
                        "closely aligned captions selected; Haas verifies and fills gaps"
                    )
                    caption_primary_logged = True

        # Weaker but still reliable captions remain a conservative fallback
        # only when Haas genuinely found no words. Live streams without a
        # downloaded caption track remain solely Haas-powered.
        if (
            not text and not original_asr and caption_match is not None and caption_match.text
            and is_usable_transcript(caption_match.text)
            and caption_match.coverage >= 0.80
            and caption_match.overlap_s >= 0.45
            and caption_match.speech_cues == 1
        ):
            text = caption_match.text
            source = "youtube_fallback"
            log.debug(
                "using aligned YouTube caption because Haas ASR returned no text "
                "(%.2f-%.2fs, coverage=%.0f%%): %s",
                utt.start_s, utt.end_s, caption_match.coverage * 100, text,
            )

        if text and rejection_reason(text) == "laughter_only":
            log.debug("Skipping nonverbal laughter at %.2fs", utt.start_s)
            continue

        if not text:
            if not utt.probe:
                empty_voiced_streak += 1
                if empty_voiced_streak % 8 == 0:
                    log.warning(
                        "ASR returned no Japanese for %d successive voiced "
                        "utterances (last %.2f-%.2fs); check speech worker",
                        empty_voiced_streak, utt.start_s, utt.end_s,
                    )
            continue
        empty_voiced_streak = 0
        log.debug("transcript source at %.2f-%.2fs: %s", utt.start_s, utt.end_s, source)

        # Proper names must be repaired before translation and glossary lookup.
        # The VOD title is a stronger signal than kanji auto-caption spelling.
        corrected = correct_asr_name(
            text,
            vod_title=vod_title,
            caption=caption_match.text if caption_match else "",
        )
        if corrected != text:
            log.debug("Watame ASR name correction: %s -> %s", text, corrected)
            text = corrected

        new_name = repair_peko_nickname(
            text, title=vod_title,
            caption=caption_match.text if caption_match else "",
            recent=list(recent_japanese),
        )
        if new_name != text:
            log.debug("Supported nickname correction: %r -> %r", text, new_name)
            text = new_name

        # Filter only complete interjections, after ASR/caption reconciliation
        # and name repair, before any translation request or subtitle output.
        if is_isolated_reaction(text):
            log.debug("Discarding isolated reaction: %s", text)
            continue

        # Mechanical sounds sometimes hallucinate as isolated "うん" -> "Yeah".
        # Protect real acknowledgements: reject only VAD-negative probes or
        # highly stationary tonal audio, unless a tightly aligned caption
        # independently confirms the same Japanese response.
        if (getattr(getattr(asr, "cfg", None), "engine_noise_ack_guard", True)
                and is_short_acknowledgement(text)
                and not caption_confirms_short_acknowledgement(text, caption_match)):
            if utt.probe or is_stationary_mechanical_audio(utt.audio):
                log.debug(
                    "Skipping unverified short acknowledgment over likely game audio "
                    "at %.2f-%.2fs (probe=%s, text=%r)",
                    utt.start_s, utt.end_s, utt.probe, text,
                )
                continue

        if get_generation is not None and utt.generation != get_generation():
            continue

        recent_japanese.append(text)
        out_q.put(
            Transcript(
                utt.start_s,
                utt.end_s,
                text,
                utt.timings,
                generation=utt.generation,
                speech_start_s=utt.speech_start_s,
                speaker_handoff=utt.speaker_handoff,
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
    history_limit = max(context_lines, getattr(translator, "memory_lines", context_lines))
    history: deque[tuple[str, str]] = deque(maxlen=history_limit)
    memory_generation: int | None = None

    def reset_scene() -> None:
        reset = getattr(translator, "reset_scene_context", None)
        if callable(reset):
            reset()

    while True:
        item = in_q.get()

        if item is None:
            break

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            if item.reset or (
                memory_generation is not None
                and item.generation != memory_generation
            ):
                history.clear()
                reset_scene()
            memory_generation = item.generation

            out_q.put(item)
            continue

        tr = item

        if get_generation is not None and tr.generation != get_generation():
            continue

        # Seek resets are normally carried by AudioProgress. Also protect
        # against a new generation reaching this queue before its marker.
        if memory_generation is not None and tr.generation != memory_generation:
            history.clear()
            reset_scene()
        memory_generation = tr.generation

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not call the translator again for the same VOD cue.
        if isinstance(tr, Subtitle):
            # Reused VOD captions bypass ASR; replay their Japanese speech
            # into scene state so the scene is reconstructed after seeks.
            observe_scene = getattr(translator, "observe_scene", None)
            if callable(observe_scene) and tr.ja:
                observe_scene(tr.ja)
            if tr.en and tr.en != "[translation failed]":
                history.append((tr.ja, tr.en))
            tr.timings.mt_start = time.monotonic()
            tr.timings.mt_end = tr.timings.mt_start
            out_q.put(tr)
            continue

        tr.timings.mt_start = time.monotonic()
        en = ""

        if translator is not None:
            # Correct a near-identical repeated Katakana word only when
            # supported by adjacent Japanese; no second ASR/model pass.
            if getattr(translator, "memory_lines", 0):
                repaired_ja = repair_near_repeated_word(tr.ja, list(history))
                if repaired_ja != tr.ja:
                    log.debug("Recent speech repetition: %r -> %r", tr.ja, repaired_ja)
                    tr = replace(tr, ja=repaired_ja)
            context_ja = [ja for ja, _ in list(history)[-8:]]
            title = getattr(translator, "vod_title", "")
            new_ja = repair_gta_carjacking(tr.ja, title, context_ja)
            if new_ja != tr.ja:
                log.debug("GTA carjacking spelling repair: %r -> %r", tr.ja, new_ja)
                tr = replace(tr, ja=new_ja)
            new_ja = repair_ark_spelling(tr.ja, title, context_ja)
            if new_ja != tr.ja:
                log.debug("ARK spelling repair: %r -> %r", tr.ja, new_ja)
                tr = replace(tr, ja=new_ja)
            new_ja = repair_horse_racing_spelling(tr.ja, title, context_ja)
            if new_ja != tr.ja:
                log.debug("Horse-racing spelling repair: %r -> %r", tr.ja, new_ja)
                tr = replace(tr, ja=new_ja)
            try:
                en = translator.translate(tr.ja, list(history))
            except Exception as e:
                log.error("translation failed: %s", e)
                en = "[translation failed]"

            if get_generation is not None and tr.generation != get_generation():
                continue

            if en.strip().upper() == "[SKIP]":
                log.debug("Skipping nonverbal translated utterance: %r", tr.ja)
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
                generation=tr.generation,
                speech_start_s=tr.speech_start_s,
                speaker_handoff=tr.speaker_handoff,
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
    captions: CaptionIndex | PendingCaptions | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Run ingestion, VAD, ASR and translation."""

    # Bounded queues cap queued audio/work during slow ASR or translation.
    # Blocking puts apply backpressure upstream instead of
    # allowing memory use and subtitle age to grow without limit.
    # 64 audio chunks ~= 16.4 s at 4096 samples / 16 kHz; work queues hold
    # at most 16 complete utterances/transcripts/subtitles each.
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    speaker_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    text_q: queue.Queue[Transcript | Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)
    context_q: queue.Queue[Transcript | Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)

    stage_failures: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    queues = {
        "audio": audio_q,
        "utt": utt_q,
        "speaker": speaker_q,
        "text": text_q,
        "context": context_q,
        "subtitle": sub_q,
    }

    threads = [
        threading.Thread(
            target=_guard_stage,
            args=("ingest", ingest.run, stage_failures, audio_q),
            name="ingest",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("vad", run_segmenter, stage_failures,
                  cfg.vad, audio_q, utt_q, lambda: ingest.generation),
            name="vad",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("speaker", run_speaker, stage_failures,
                  cfg.speaker, utt_q, speaker_q, lambda: ingest.generation),
            name="speaker",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "asr", run_asr, stage_failures,
                asr,
                speaker_q,
                text_q,
                lambda: ingest.generation,
                captions,
                vod_title,
                session_cache,
                ingest.source.live,
            ),
            name="asr",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "context", run_contextualize, stage_failures,
                text_q,
                context_q,
                lambda: ingest.generation,
                vod_title,
                cfg.translate.lookahead_s,
                cfg.translate.fragment_join_gap_s,
                cfg.translate.fragment_join_enabled,
            ),
            name="context",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "translate", run_translate, stage_failures,
                translator,
                cfg.translate.context_lines,
                context_q,
                sub_q,
                lambda: ingest.generation,
            ),
            name="translate",
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    stats = Stats()
    logged_subtitle_generations: set[int] = set()
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s
    last_pipeline_event = started
    stall_warn_at = started

    try:
        while not stop.is_set():
            _raise_stage_failure(stage_failures)
            try:
                item = sub_q.get(timeout=0.5)
                last_pipeline_event = time.monotonic()

                if item is None:
                    # An upstream stage can send its final sentinel in a
                    # finally-block, then report a crash to _guard_stage.
                    # Join briefly before declaring normal completion.
                    for thread in threads:
                        thread.join(timeout=0.3)
                    _raise_stage_failure(stage_failures)
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

            _raise_stage_failure(stage_failures)
            for name, q in queues.items():
                stats.max_q[name] = max(
                    stats.max_q.get(name, 0),
                    q.qsize(),
                )

            now = time.monotonic()
            # A stuck worker can leave every thread alive with blocked
            # bounded queues. A quiet stream with no queued work is allowed.
            backlog = {name: q.qsize() for name, q in queues.items() if q.qsize()}
            if backlog and now - last_pipeline_event >= 60.0:
                raise RuntimeError(
                    "translation pipeline stalled for 60s with queued work: "
                    + ", ".join(f"{name}={n}" for name, n in backlog.items())
                    + "; inspect logs/run.log and GPU memory"
                )
            if backlog and now - last_pipeline_event >= 20.0 and now >= stall_warn_at:
                stall_warn_at = now + 20.0
                log.warning(
                    "no pipeline output for %.0fs; backlog %s; stages %s",
                    now - last_pipeline_event,
                    " ".join(f"{name}={n}" for name, n in backlog.items()),
                    " ".join(f"{th.name}={'alive' if th.is_alive() else 'stopped'}"
                             for th in threads),
                )

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
