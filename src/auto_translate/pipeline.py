
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
from .watame import correct_asr_name, reconcile_caption
from .stream_context import (
    repair_peko_nickname, repair_verified_caption, repair_strongly_aligned_caption,
    repair_ark_spelling,
    repair_gta_carjacking, repair_horse_racing_spelling,
)
from .captions import CaptionIndex, PendingCaptions
from .context_memory import repair_near_repeated_word
from .translation_guard import uncertain_asr_text, trusted_merge_candidate
from .text_filter import (
    is_isolated_reaction, is_nonspeech_probe, is_sound_caption_artifact,
    is_usable_transcript, strip_nonverbal_labels, strip_incidental_laughter,
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
) -> None:
    """Transcribe utterances, optionally using aligned captions as a fallback."""
    recent_japanese: deque[str] = deque(maxlen=12)
    last_generation: int | None = None
    empty_voiced_streak = 0
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
                speaker_context_key=utt.speaker_context_key,
            ))
            continue

        if utt.probe and is_nonspeech_probe(utt.audio):
            continue

        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        utt.timings.asr_end = time.monotonic()
        original_asr = text
        text = strip_nonverbal_labels(text)
        if original_asr != text:
            log.debug("filtered ASR nonverbal labels at %.2fs: %r -> %r", utt.start_s, original_asr, text)

        caption_match = (
            captions.match(utt.start_s, utt.end_s)
            if captions is not None
            else None
        )
        if caption_match is not None and not caption_match.reliable:
            caption_match = None
        if caption_match is not None and not caption_match.text and is_sound_caption_artifact(text):
            log.debug("Discarding likely ASR hallucination over sound-only caption at %.2fs", utt.start_s)
            continue

        source = "haas_asr"
        if text and caption_match is not None:
            reconciled = reconcile_caption(text, caption_match.text)
            if caption_match.coverage >= 0.80 and caption_match.overlap_s >= 0.45:
                reconciled = repair_verified_caption(reconciled, caption_match.text)
            # The looser two-typo repair has stricter acoustic alignment
            # requirements. Haas remains primary on genuine disagreements.
            if caption_match.coverage >= 0.90 and caption_match.overlap_s >= 0.75:
                reconciled = repair_strongly_aligned_caption(
                    reconciled, caption_match.text,
                )
            if reconciled != text:
                source = "haas_asr+youtube_correction"
                text = reconciled

        if (
            not text and not original_asr and caption_match is not None and caption_match.text
            and is_usable_transcript(caption_match.text)
        ):
            text = caption_match.text
            source = "youtube_fallback"
            log.debug(
                "using aligned YouTube caption because Haas ASR returned no text "
                "(%.2f-%.2fs, coverage=%.0f%%): %s",
                utt.start_s, utt.end_s, caption_match.coverage * 100, text,
            )

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

        # Keep spoken words but remove obvious laughs before translation or
        # speaker context memory sees the Japanese text. Caption/name repairs
        # have already run, so they cannot reinsert a removed laugh.
        before_laughter = text
        text = strip_incidental_laughter(text)
        if text != before_laughter:
            log.debug(
                "Removed incidental laughter at %.2fs: %r -> %r",
                utt.start_s, before_laughter, text,
            )
        if not text:
            log.debug("Discarding laughter-only utterance at %.2fs", utt.start_s)
            continue

        # Filter only complete interjections, after ASR/caption reconciliation
        # and name repair, before any translation request or subtitle output.
        if is_isolated_reaction(text):
            log.debug("Discarding isolated reaction: %s", text)
            continue

        if get_generation is not None and utt.generation != get_generation():
            continue

        # Haas exposes only text, not acoustic confidence scores.
        # Mark speculative Katakana / strong caption disagreement as uncertain:
        # do not discard real names or overwrite the transcript.
        uncertainty = uncertain_asr_text(
            text,
            caption=caption_match.text if caption_match is not None else "",
            caption_coverage=caption_match.coverage if caption_match is not None else 0.0,
            caption_overlap_s=caption_match.overlap_s if caption_match is not None else 0.0,
        )
        if uncertainty:
            log.debug("ASR text may be uncertain at %.2fs: %r", utt.start_s, text)
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
                speaker_context_key=utt.speaker_context_key,
                asr_uncertain=uncertainty,
            )
        )

    out_q.put(None)


class RecentSpeakerJapanese:
    """Bounded reference text from verified adjacent turns of one voice.

    The speaker-stage key only means consecutive CPU embeddings matched;
    it is never a global speaker ID. Any uncertain voice, interruption, long
    gap or seek clears the reference context.
    """

    def __init__(self, *, max_gap_s: float = 1.25) -> None:
        self.max_gap_s = max_gap_s
        self.lines: deque[tuple[int, float, str]] = deque(maxlen=2)

    def reset(self) -> None:
        self.lines.clear()

    def context_for(self, item: Transcript | Subtitle) -> str | None:
        if not self.lines or item.speaker_context_key is None or item.speaker_handoff:
            return None
        previous_key, previous_end, _ = self.lines[-1]
        gap = item.start_s - previous_end
        if (previous_key != item.speaker_context_key
                or gap < -0.03 or gap > self.max_gap_s):
            return None
        return "。".join(line for _, _, line in self.lines)[-140:]

    def observe(self, item: Transcript | Subtitle) -> None:
        key = item.speaker_context_key
        if key is None or item.speaker_handoff or not item.ja.strip():
            self.reset()
            if key is None or not item.ja.strip():
                return
        elif self.lines:
            previous_key, previous_end, _ = self.lines[-1]
            gap = item.start_s - previous_end
            if previous_key != key or gap < -0.03 or gap > self.max_gap_s:
                self.reset()
        self.lines.append((key, item.end_s, item.ja.strip()[:100]))


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
    speaker_reference = RecentSpeakerJapanese()
    memory_generation: int | None = None
    last_scene_end_s: float | None = None
    translator_cfg = getattr(translator, "cfg", None)
    fragment_enabled = bool(getattr(translator_cfg, "fragment_join_enabled", False))
    fragment_wait_s = min(0.5, max(0.0, float(
        getattr(translator_cfg, "fragment_join_wait_s", 0.5)
    )))
    fragment_gap_s = min(0.5, max(0.0, float(
        getattr(translator_cfg, "fragment_join_gap_s", 0.35)
    )))
    # Preserve an unrelated next queue item (including EOS and seek markers).
    deferred_empty = object()
    deferred: object = deferred_empty
    configured_gap = getattr(getattr(translator, "cfg", None), "scene_gap_reset_s", 30.0)
    scene_gap_reset_s = (
        max(5.0, float(configured_gap))
        if isinstance(configured_gap, (int, float)) else 30.0
    )

    def reset_scene() -> None:
        reset = getattr(translator, "reset_scene_context", None)
        if callable(reset):
            reset()

    while True:
        if deferred is not deferred_empty:
            item, deferred = deferred, deferred_empty
        else:
            item = in_q.get()

        if item is None:
            break

        if (
            fragment_enabled and fragment_wait_s > 0
            and isinstance(item, Transcript)
            and item.speaker_context_key is not None
            and not item.speaker_handoff and not item.asr_uncertain
        ):
            from .stream_context import is_incomplete_fragment
            if is_incomplete_fragment(item.ja):
                try:
                    following = in_q.get(timeout=fragment_wait_s)
                except queue.Empty:
                    following = deferred_empty
                if following is not deferred_empty:
                    if (
                        isinstance(following, Transcript)
                        and following.generation == item.generation
                        and not following.asr_uncertain
                        and trusted_merge_candidate(
                            item.ja, following.ja,
                            first_key=item.speaker_context_key,
                            next_key=following.speaker_context_key,
                            first_end_s=item.end_s,
                            next_start_s=following.start_s,
                            combined_duration_s=following.end_s - item.start_s,
                            next_handoff=following.speaker_handoff,
                            max_gap_s=fragment_gap_s,
                        )
                    ):
                        log.debug(
                            "Joined verified same-speaker fragment %.2f-%.2fs",
                            item.start_s, following.end_s,
                        )
                        item = replace(
                            item,
                            ja=item.ja.rstrip("、 ,") + following.ja.lstrip(),
                            end_s=following.end_s,
                            timings=following.timings,
                        )
                    else:
                        deferred = following

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            if item.reset or (
                memory_generation is not None
                and item.generation != memory_generation
            ):
                history.clear()
                speaker_reference.reset()
                reset_scene()
                last_scene_end_s = None
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
            speaker_reference.reset()
            reset_scene()
            last_scene_end_s = None
        memory_generation = tr.generation

        # A long speech gap indicates an unobserved scene transition.
        # Also guard against a backward timeline discontinuity with no marker.
        # Do not reset merely because a different speaker joins the same scene.
        if last_scene_end_s is not None and (
            tr.start_s - last_scene_end_s > scene_gap_reset_s
            or tr.start_s < last_scene_end_s - 1.0
        ):
            history.clear()
            speaker_reference.reset()
            reset_scene()
        last_scene_end_s = tr.end_s

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not call the translator again for the same VOD cue.
        if isinstance(tr, Subtitle):
            # Reused VOD captions bypass ASR; replay their Japanese speech
            # into scene state so the scene is reconstructed after seeks.
            observe_scene = getattr(translator, "observe_scene", None)
            if callable(observe_scene) and tr.ja:
                observe_scene(tr.ja)
            if tr.en and tr.en != "[translation failed]" and not tr.asr_uncertain:
                history.append((tr.ja, tr.en))
                speaker_reference.observe(tr)
            else:
                # A possibly garbled Japanese fragment must not poison
                # future name repairs, prompt history or scene hints.
                speaker_reference.reset()
            else:
                speaker_reference.reset()
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
                set_context = getattr(translator, "set_reference_context", None)
                if callable(set_context):
                    set_context(None if tr.asr_uncertain else speaker_reference.context_for(tr))
                set_reliability = getattr(translator, "set_source_reliability", None)
                if callable(set_reliability):
                    set_reliability(tr.asr_uncertain)
                en = translator.translate(tr.ja, list(history))
            except Exception as e:
                log.error("translation failed: %s", e)
                en = "[translation failed]"

            if get_generation is not None and tr.generation != get_generation():
                continue

            if en.strip().upper() == "[SKIP]":
                speaker_reference.reset()
                log.debug("Skipping nonverbal translated utterance: %r", tr.ja)
                continue

            if en and en != "[translation failed]" and not tr.asr_uncertain:
                history.append((tr.ja, en))
                speaker_reference.observe(tr)
            else:
                # Still display uncertain dialogue, but never reinforce it.
                speaker_reference.reset()

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
                speaker_context_key=tr.speaker_context_key,
                asr_uncertain=tr.asr_uncertain,
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
    text_q: queue.Queue[Transcript | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)

    stage_failures: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    queues = {
        "audio": audio_q,
        "utt": utt_q,
        "speaker": speaker_q,
        "text": text_q,
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
            ),
            name="asr",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "translate", run_translate, stage_failures,
                translator,
                cfg.translate.context_lines,
                text_q,
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
