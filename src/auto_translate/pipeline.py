
"""Wire the stages together with threads and queues."""

import logging
import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import CancelledError
from contextlib import nullcontext
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Callable

from .asr.base import ASR
from .asr.quality_guard import transcript_problem, supports_japanese_transcript
from .asr_correction import correct_known_asr_terms
from .watame import correct_asr_name, reconcile_caption
from .captions import CaptionIndex, PendingCaptions
from .context_memory import repair_near_repeated_word
from .proper_names import resolve_names
from .text_filter import is_usable_transcript
from .subtitle_clean import (
    clean_english_output, has_generation_loop, normalized_cue,
    looks_like_prompt_echo, sanitize_english_output,
)
from .config import AsrConfig, Config, vad_for_source
from .gui_control import emit_event
from .runtime_diagnostics import update_metrics, increment_counter, record_event
from .ingest import Ingest
from .output.base import Output
from .session_cache import SessionSubtitleCache
from .translate.base import Translator, TranslatorUnavailable
from .types import (
    SAMPLE_RATE,
    AudioChunk,
    AudioProgress,
    SourceEOF,
    Subtitle,
    Transcript,
    Utterance,
)
from .vad import Segmenter, run_segmenter

log = logging.getLogger(__name__)


def _guard_stage(
    name: str,
    fn: Callable,
    failures: "queue.Queue[tuple[str, Exception]]",
    *args,
) -> None:
    """Surface ASR/MT/ingest worker failures in the main runner."""
    try:
        fn(*args)
    except Exception as exc:
        log.exception("pipeline worker %s crashed", name)
        failures.put((name, exc))


def _raise_worker_failure(
    failures: "queue.Queue[tuple[str, Exception]]",
) -> None:
    try:
        name, exc = failures.get_nowait()
    except queue.Empty:
        return
    message = f"pipeline worker '{name}' failed: {exc}"
    emit_event("session_error", message=message)
    raise RuntimeError(message) from exc


def _expected_vod_idle(ingest: Ingest, outputs: list[Output]) -> bool:
    if ingest.source.live:
        return False
    parked = getattr(ingest, "is_vod_audio_lead_limited", None)
    if callable(parked) and parked():
        return True
    for output in outputs:
        manually_paused = getattr(output, "is_manually_paused", None)
        if callable(manually_paused) and manually_paused():
            return True
    return False


def _expired_live_cue(cue, playback_position, late_window_s: float) -> bool:
    """Only skip whole cues MPV already played; never drop raw PCM/progress."""
    if not callable(playback_position):
        return False
    try:
        position = playback_position()
    except (RuntimeError, OSError, ValueError):
        return False
    # Refuse to shed on grossly inconsistent MPV clocks (e.g. bad live
    # timeline anchors); dropping all dialogue would be worse than backlog.
    return (position is not None and math.isfinite(position)
            and max(0.0, late_window_s) < position - cue.end_s <= 120.0)


def run_asr(
    asr: ASR,
    in_q: "queue.Queue[Utterance | AudioProgress | SourceEOF | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | SourceEOF | None]",
    get_generation: Callable[[], int] | None = None,
    captions: CaptionIndex | PendingCaptions | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
    asr_cfg: AsrConfig | None = None,
    live: bool = False,
    playback_position: Callable[[], float | None] | None = None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
    language_filter=None,
    gate=None,
    chat_context=None,
) -> None:
    """Recognize complete VAD utterances, retaining their original media times."""
    import re
    import numpy as np

    name_history = deque(maxlen=4)
    name_generation = None
    last_transcript_end_s = None
    last_text = ""
    repeated = 0
    # Relative live timestamps can be reused after reconnects. A VOD cache
    # must never supply cues or name context for a different live generation.
    if live:
        session_cache = None

    def cancelled(generation):
        return ((stop is not None and stop.is_set())
                or (get_generation is not None and generation != get_generation()))

    while True:
        item = in_q.get()
        if item is None:
            break
        if cancelled(item.generation):
            continue
        if isinstance(item, SourceEOF):
            out_q.put(item)
            continue
        if isinstance(item, AudioProgress):
            if item.reset or item.generation != name_generation:
                name_history.clear()
                last_text, repeated = "", 0
                cancel = getattr(asr, "cancel_pending", None)
                if callable(cancel):
                    cancel()
            name_generation = item.generation
            out_q.put(item)
            continue
        utt = item
        if utt.generation != name_generation:
            name_history.clear()
            last_text, repeated = "", 0
            cancel = getattr(asr, "cancel_pending", None)
            if callable(cancel):
                cancel()
            name_generation = utt.generation
        if live and _expired_live_cue(utt, playback_position, live_late_window_s):
            increment_counter("expired")
            continue
        cached = (session_cache.match(utt.start_s, utt.end_s)
                  if session_cache is not None else None)
        if cached is not None:
            cleaned = sanitize_english_output(cached.en)
            cached = replace(cached, en=cleaned) if cleaned else None
        if cached is not None:
            if cancelled(utt.generation):
                continue
            name_history.append(cached.ja[:160])
            last_transcript_end_s = utt.end_s
            out_q.put(replace(cached, generation=utt.generation, timings=utt.timings))
            continue
        language_decision = None
        if language_filter is not None:
            verdict = language_filter.classify(utt.audio)
            language_decision = verdict
            if verdict.language not in ("ja", "unknown"):
                last_transcript_end_s = None
            if verdict.drop:
                record_event("language_filtered", generation=utt.generation,
                             language=verdict.language, probability=verdict.probability)
                continue
        if cancelled(utt.generation):
            continue
        utt.timings.asr_start = time.monotonic()
        try:
            with (gate.operation("voxtral", current=get_generation, generation=utt.generation)
                  if gate is not None else nullcontext()):
                if live and _expired_live_cue(utt, playback_position, live_late_window_s):
                    increment_counter("expired")
                    continue
                text = asr.transcribe(utt.audio, utt.probe)
        except CancelledError:
            continue
        utt.timings.asr_end = time.monotonic()
        update_metrics(stages={"voxtral_s": utt.timings.asr_end - utt.timings.asr_start})
        record_event("asr_complete", generation=utt.generation,
                     source_start_s=utt.start_s, speech_start_s=utt.speech_start_s,
                     source_end_s=utt.end_s, pcm_start_sample=round(utt.start_s * SAMPLE_RATE),
                     segmented_at=utt.timings.segmented,
                     asr_completed_at=utt.timings.asr_end)
        if cancelled(utt.generation):
            continue

        previous_ja = (name_history[-1] if name_history and last_transcript_end_s is not None
                       and 0 <= utt.start_s - last_transcript_end_s <= 10.0 else "")
        if not isinstance(text, str) or not supports_japanese_transcript(
                text, language_decision=language_decision, video_context=vod_title,
                previous_ja=previous_ja, vad_confidence=utt.vad_confidence,
                duration_s=len(utt.audio) / SAMPLE_RATE, probe=utt.probe):
            continue
        if asr_cfg is not None and any(term in text for term in asr_cfg.blocklist):
            continue
        issue = transcript_problem(text, utt.end_s - utt.start_s)
        if issue:
            increment_counter("dropped")
            record_event("asr_suppressed", reason=issue, generation=utt.generation)
            continue
        compact = re.sub(r"[\W_]+", "", text.casefold())
        repeated = repeated + 1 if compact == last_text else 1
        last_text = compact
        # Retain spoken repeated reactions; only suppress sustained long loops.
        if len(compact) >= 12 and repeated >= 4:
            increment_counter("dropped")
            continue
        # Familiar hallucinated outros on faint, sub-second non-speech are
        # insufficient evidence for a subtitle. Never reject a normal short reply.
        if (text.strip(" 。！？!?") in ("ご視聴ありがとうございました", "チャンネル登録よろしくお願いします")
                and utt.end_s - utt.start_s < 1.3
                and utt.vad_confidence is not None and utt.vad_confidence < .55
                and len(utt.audio) and float(np.mean(np.square(utt.audio))) < .0001):
            increment_counter("dropped")
            continue
        original = text
        caption_match = captions.match(utt.start_s, utt.end_s) if captions is not None else None
        caption_text = caption_match.text if caption_match and caption_match.reliable else ""
        if caption_text:
            text = reconcile_caption(text, caption_text)
        # Captions only corroborate recognized words; an empty recognition is
        # never replaced by dialogue that was not heard in this utterance.
        text = correct_asr_name(text, vod_title=vod_title, caption=caption_text)
        text = correct_known_asr_terms(text, video_context=vod_title, caption=caption_text)
        if not is_usable_transcript(text) or cancelled(utt.generation):
            continue
        name_context = {"japanese": text, "voxtral": original}
        raw_transcript = getattr(asr, "last_raw_transcript", None)
        if isinstance(raw_transcript, str):
            name_context["voxtral_raw"] = raw_transcript
        try:
            mentions = resolve_names(text, previous_ja=" ".join(name_history), video_title=vod_title)
            name_context["mentions"] = [mention.as_dict() for mention in mentions]
        except Exception as exc:
            log.warning("Name resolver unavailable: %s", exc)
            name_context["mentions"] = []
        if chat_context is not None:
            try:
                name_context["chat_evidence"] = chat_context(utt, text)
            except Exception as exc:
                log.debug("Public chat context unavailable: %s", exc)
        if cancelled(utt.generation):
            continue
        name_history.append(text[:160])
        last_transcript_end_s = utt.end_s
        out_q.put(Transcript(utt.start_s, utt.end_s, text, utt.timings,
                             generation=utt.generation, speech_start_s=utt.speech_start_s,
                             name_context=name_context))
    out_q.put(None)


def run_translate(
    translator: Translator | None,
    context_lines: int,
    in_q: "queue.Queue[Transcript | Subtitle | AudioProgress | SourceEOF | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | SourceEOF | None]",
    get_generation: Callable[[], int] | None = None,
    live: bool = False,
    playback_position: Callable[[], float | None] | None = None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
    gate=None,
) -> None:
    """Translate transcripts while retaining recent context and forwarding progress."""
    consecutive_failures = 0
    history_limit = max(context_lines, getattr(translator, "memory_lines", context_lines))
    if getattr(getattr(translator, "cfg", None), "context_profile", None) in ("overlay", "hybrid"):
        # Profiles select three trusted exchanges from a bounded candidate
        # window, including when extended memory is disabled.
        history_limit = 20
    history: deque[tuple[str, str]] = deque(maxlen=history_limit)
    memory_generation: int | None = None

    while True:
        item = in_q.get()

        if item is None:
            break
        if stop is not None and stop.is_set():
            continue  # do not translate queued cues after player shutdown

        if isinstance(item, SourceEOF):
            if get_generation is None or item.generation == get_generation():
                out_q.put(item)
            continue
        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            if item.reset or (
                memory_generation is not None
                and item.generation != memory_generation
            ):
                history.clear()
                consecutive_failures = 0
            memory_generation = item.generation

            out_q.put(item)
            continue

        tr = item

        if get_generation is not None and tr.generation != get_generation():
            continue
        if live and _expired_live_cue(tr, playback_position, live_late_window_s):
            log.info("Skipping expired live PLaMo cue %.2f-%.2fs", tr.start_s, tr.end_s)
            increment_counter("expired")
            continue

        # Seek resets are normally carried by AudioProgress. Also protect
        # against a new generation reaching this queue before its marker.
        if memory_generation is not None and tr.generation != memory_generation:
            history.clear()
            consecutive_failures = 0
        memory_generation = tr.generation

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not call PLaMo again for the same VOD cue.
        if isinstance(tr, Subtitle):
            cleaned = sanitize_english_output(tr.en)
            if not cleaned:
                increment_counter("dropped")
                continue
            tr = replace(tr, en=cleaned)
            if tr.ja:
                history.append((tr.ja, cleaned))
            tr.timings.mt_start = time.monotonic()
            tr.timings.mt_end = tr.timings.mt_start
            out_q.put(tr)
            continue

        tr.timings.mt_start = time.monotonic()
        before_history = tr.ja
        en = ""
        suppressed = False

        if translator is not None:
            # Correct a near-identical repeated Katakana word only when
            # supported by adjacent Japanese; no second ASR/model pass.
            if getattr(translator, "memory_lines", 0):
                repaired_ja = repair_near_repeated_word(tr.ja, list(history))
                if repaired_ja != tr.ja:
                    log.debug("Recent speech repetition: %r -> %r", tr.ja, repaired_ja)
                    tr = replace(tr, ja=repaired_ja)
            try:
                with (gate.operation("plamo", current=get_generation, generation=tr.generation)
                      if gate is not None else nullcontext()):
                    if live and _expired_live_cue(tr, playback_position, live_late_window_s):
                        increment_counter("expired")
                        continue
                    model_started = time.monotonic()
                    set_context = getattr(translator, "set_name_context", None)
                    if callable(set_context) and tr.name_context.get("japanese") == tr.ja:
                        try:
                            set_context(tr.ja, tr.name_context.get("mentions", []))
                        except Exception as exc:
                            log.warning("Translation name context unavailable: %s", exc)
                    en = translator.translate(tr.ja, list(history))
                    update_metrics(stages={"plamo_s": time.monotonic() - model_started})
                    name_diagnostic = getattr(translator, "last_name_diagnostics", {})
                    if (not en and isinstance(name_diagnostic, dict)
                            and name_diagnostic.get("unresolved_name_issue")):
                        suppressed = True
                        log.warning("Name translation abstained at %.2fs: %s", tr.start_s,
                                    name_diagnostic["unresolved_name_issue"])
                consecutive_failures = 0
            except CancelledError:
                continue
            except TranslatorUnavailable:
                if ((stop is not None and stop.is_set()) or
                        (get_generation is not None and tr.generation != get_generation())):
                    continue
                raise
            except Exception as e:
                if ((stop is not None and stop.is_set()) or
                        (get_generation is not None and tr.generation != get_generation())):
                    continue
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise TranslatorUnavailable(f"PLaMo failed on three consecutive cues: {e}") from e
                log.error("translation failed: %s", e)
                suppressed = True

            if get_generation is not None and tr.generation != get_generation():
                continue
            if stop is not None and stop.is_set():
                continue
            if live and _expired_live_cue(tr, playback_position, live_late_window_s):
                increment_counter("expired")
                continue

            if en:
                tcfg = getattr(translator, "cfg", None)
                if (getattr(tcfg, "suppress_runaway_loops", True)
                        and (has_generation_loop(en) or looks_like_prompt_echo(en))):
                    log.warning(
                        "Suppressed repeated/prompt-like PLaMo output: gen=%d source=%.2f-%.2fs",
                        tr.generation, tr.start_s, tr.end_s,
                    )
                    en, suppressed = "", True
                else:
                    en = clean_english_output(
                        en,
                        remove_fillers=getattr(tcfg, "remove_fillers", True),
                        collapse_repetitions=getattr(tcfg, "collapse_repetitions", True),
                    )
                    if not en:
                        suppressed = True

            en = sanitize_english_output(en)
            if not en:
                suppressed = True

            if en:
                history.append((tr.ja, en))

            if not en.strip() and not suppressed:
                log.error(
                    "translator returned an empty English result "
                    "(generation=%d, %.2f-%.2fs, Japanese=%r)",
                    tr.generation,
                    tr.start_s,
                    tr.end_s,
                    tr.ja,
                )

        tr.timings.mt_end = time.monotonic()
        record_event("translation_complete", generation=tr.generation,
                     elapsed_s=tr.timings.mt_end-tr.timings.mt_start,
                     source_start_s=tr.start_s, speech_start_s=tr.speech_start_s,
                     source_end_s=tr.end_s, asr_completed_at=tr.timings.asr_end,
                     mt_completed_at=tr.timings.mt_end,
                     suppressed=suppressed, history_repaired=before_history != tr.ja)
        if suppressed:
            increment_counter("dropped")
            log.info(
                "Translated cue %.2f-%.2fs suppressed by English output guard; "
                "Japanese ASR recognized speech: %r",
                tr.start_s, tr.end_s, tr.ja[:110],
            )
            continue

        out_q.put(
            Subtitle(
                tr.start_s,
                tr.end_s,
                tr.ja,
                en,
                tr.timings,
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
    recent_lag_s: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    deadline_misses: int = 0

    def add(self, sub: Subtitle, deadline_s: float | None = None) -> None:
        tm = sub.timings
        asr_s = max(0.0, tm.asr_end - tm.asr_start)
        mt_s = max(0.0, tm.mt_end - tm.mt_start)
        total_s = max(0.0, tm.mt_end - tm.audio_read)
        self.lines += 1
        self.asr_sum_s += asr_s
        self.mt_sum_s += mt_s
        self.total_sum_s += total_s
        self.recent_lag_s.append(total_s)
        if deadline_s is not None and total_s > deadline_s:
            self.deadline_misses += 1
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

        sorted_lag = sorted(self.recent_lag_s)
        p95 = sorted_lag[min(len(sorted_lag) - 1, int(len(sorted_lag) * 0.95))]
        return (
            f"{self.lines} lines | "
            f"pipeline p95 {p95:.2f}s, deadline misses {self.deadline_misses} | "
            f"{fmt('asr', self.asr_sum_s, self.asr_max_s)} | "
            f"{fmt('mt', self.mt_sum_s, self.mt_max_s)} | "
            f"{fmt('latency', self.total_sum_s, self.total_max_s)} | "
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
    playback_position: Callable[[], float | None] | None = None,
    chat_context=None,
    language_filter=None,
    segmenter=None,
    gate=None,
    diagnostics=None,
) -> None:
    """Run ingestion, VAD, ASR and translation."""

    if ingest.source.live:
        session_cache = None

    vad_cfg = vad_for_source(cfg.vad, live=bool(ingest.source.live))
    log.info(
        "FireRedVAD accuracy-first (%s): threshold=%.2f, min_speech=%dms, "
        "min_silence=%dms, pad=%dms, onset=%dms, soft_cap=%.1fs, hard_cap=%.1fs, "
        "VAD-negative probing=%s",
        "live" if ingest.source.live else "VOD",
        vad_cfg.threshold, vad_cfg.min_speech_ms, vad_cfg.min_silence_ms,
        vad_cfg.speech_pad_ms, vad_cfg.onset_ms,
        vad_cfg.max_utterance_s,
        vad_cfg.max_utterance_s + vad_cfg.max_split_extension_s,
        "on" if vad_cfg.fallback_s > 0 else "off",
    )

    # Bounded queues cap queued audio/work during slow ASR or translation.
    # Blocking puts apply backpressure upstream instead of
    # allowing memory use and subtitle age to grow without limit.
    # 64 audio chunks ~= 16.4 s at 4096 samples / 16 kHz; work queues hold
    # at most 16 complete utterances/transcripts/subtitles each.
    safe = cfg.safe_playback
    work_size = safe.queue_size if safe.enabled else 16
    audio_q: queue.Queue[AudioChunk | AudioProgress | SourceEOF | None] = queue.Queue(maxsize=safe.audio_chunks if safe.enabled else 64)
    utt_q: queue.Queue[Utterance | AudioProgress | SourceEOF | None] = queue.Queue(maxsize=work_size)
    text_q: queue.Queue[Transcript | AudioProgress | SourceEOF | None] = queue.Queue(maxsize=work_size)
    sub_q: queue.Queue[Subtitle | AudioProgress | SourceEOF | None] = queue.Queue(maxsize=work_size)

    queues = {
        "audio": audio_q,
        "utt": utt_q,
        "text": text_q,
        "subtitle": sub_q,
    }
    failures: queue.Queue[tuple[str, Exception]] = queue.Queue()

    threads = [
        threading.Thread(
            target=_guard_stage,
            args=("ingest", ingest.run, failures, audio_q),
            name="ingest",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("vad", run_segmenter, failures, segmenter if segmenter is not None else Segmenter(vad_cfg), audio_q, utt_q,
                  lambda: ingest.generation, None, stop),
            name="vad",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "asr", run_asr, failures, asr,
                utt_q,
                text_q,
                lambda: ingest.generation,
                captions,
                vod_title,
                session_cache,
                cfg.asr,
                bool(ingest.source.live),
                playback_position,
                cfg.mpv.live_late_window_s,
                stop,
                language_filter,
                gate,
                chat_context,
            ),
            name="asr",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=(
                "translate", run_translate, failures, translator,
                cfg.translate.context_lines,
                text_q,
                sub_q,
                lambda: ingest.generation,
                bool(ingest.source.live),
                playback_position,
                cfg.mpv.live_late_window_s,
                stop,
                gate,
            ),
            name="translate",
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    stats = Stats()
    last_subtitle_generation: int | None = None
    observed_generation = ingest.generation
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s
    next_diagnostics = started
    last_activity = started
    last_warning = started
    last_guard_key = ""
    last_guard_end = -float("inf")
    last_guard_generation: int | None = None
    identical_count = 0

    try:
        while not stop.is_set():
            # Generation checks in each worker discard an in-flight old cue.
            # The ASR worker flushes its adapter before the next generation;
            # a delayed main-loop tick must never cancel a new-generation cue.
            if ingest.generation != observed_generation:
                observed_generation = ingest.generation
                record_event("inference_generation_reset", generation=observed_generation)
            _raise_worker_failure(failures)
            if diagnostics is not None and time.monotonic() >= next_diagnostics:
                next_diagnostics = time.monotonic() + cfg.diagnostics.sample_interval_s
                from .windows_subprocess import owned_process_status
                update_metrics(queues={name: q.qsize() for name, q in queues.items()},
                               workers={"threads": {t.name: t.is_alive() for t in threads},
                                        "processes": owned_process_status()})
                emit_event("diagnostics", **diagnostics.snapshot())
            try:
                item = sub_q.get(timeout=0.5)
                if stop.is_set():
                    break
                last_activity = time.monotonic()

                if item is None:
                    for thread in threads:
                        thread.join(timeout=0.2)
                    _raise_worker_failure(failures)
                    for out in outputs:
                        finish = getattr(out, "finish", None)
                        if finish is not None:
                            finish()

                    for out in outputs:
                        out.drain()
                    break

                # Discard subtitles and progress from before the latest seek.
                if item.generation != ingest.generation:
                    increment_counter("dropped")
                    continue

                if isinstance(item, SourceEOF):
                    for out in outputs:
                        finish = getattr(out, "finish", None)
                        if finish is not None:
                            finish(item.generation)
                    continue
                if isinstance(item, AudioProgress):
                    progress_s = item.sample / SAMPLE_RATE
                    if item.reset:
                        last_guard_key, last_guard_end = "", -float("inf")
                        identical_count = 0
                        last_guard_generation = item.generation
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

                if item.generation != last_subtitle_generation:
                    last_subtitle_generation = item.generation
                    log.info(
                        "first translated subtitle for generation=%d at %.2fs",
                        item.generation,
                        item.start_s,
                    )

                if item.en:
                    cleaned = sanitize_english_output(item.en)
                    if not cleaned:
                        increment_counter("dropped")
                        continue
                    item = replace(item, en=cleaned)
                elif translator is not None:
                    increment_counter("dropped")
                    continue

                if item.generation != last_guard_generation:
                    last_guard_key, last_guard_end, identical_count = "", -float("inf"), 0
                    last_guard_generation = item.generation
                key = normalized_cue(item.en) if item.en else ""
                if (key and len(key.split()) >= 2 and key == last_guard_key
                        and -0.2 <= item.start_s - last_guard_end <= 10.0):
                    identical_count += 1
                else:
                    identical_count = 1
                last_guard_key, last_guard_end = key, item.end_s
                if (key and identical_count >
                        max(1, int(cfg.translate.max_identical_cues))):
                    log.info("Suppressed repeat subtitle: gen=%d count=%d text=%r",
                             item.generation, identical_count, item.en[:110])
                    increment_counter("dropped")
                    continue

                stats.add(item, deadline_s=cfg.mpv.delay_s if ingest.source.live else None)
                latency = max(0., time.monotonic() - item.timings.audio_read) if item.timings.audio_read else None
                update_metrics(counters={"latency_s": latency})
                if latency is not None and latency > max(30., cfg.mpv.delay_s):
                    record_event("subtitle_latency_spike", latency_s=round(latency, 3), generation=item.generation)

                if session_cache is not None:
                    session_cache.add(item)

                for out in outputs:
                    out.show(item)

            except queue.Empty:
                pass

            _raise_worker_failure(failures)
            now = time.monotonic()
            if _expected_vod_idle(ingest, outputs):
                # User pause or intentional audio lead cap is not a stall.
                last_activity = now
                last_warning = now
            else:
                backlog = {name: q.qsize() for name, q in queues.items() if q.qsize()}
                if backlog and now - last_activity > 90.0:
                    record_event("pipeline_stall", queues=backlog, elapsed_s=now-last_activity)
                    raise RuntimeError(
                        "Voxtral/PLaMo pipeline stalled with queued work: " + str(backlog)
                    )
                if backlog and now - last_activity > 30.0 and now - last_warning > 30.0:
                    last_warning = now
                    log.warning("Voxtral/PLaMo pipeline waiting with queued work: %s", backlog)

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


        for output in outputs:
            failure = getattr(output, "scheduler_failure", None)
            if isinstance(failure, Exception):
                raise RuntimeError("MPV scheduler crashed") from failure
    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")

    finally:
        # Stop producers and MPV, then interrupt the owned ASR server so a
        # pending request cannot keep the application alive after closing.
        stop.set()
        close_asr = getattr(asr, "close", None)
        if callable(close_asr):
            try:
                close_asr()
            except Exception:
                log.exception("Voxtral interruption failed during shutdown")
        try:
            for out in outputs:
                try:
                    out.close()
                except Exception:
                    log.exception("output shutdown failed")
        finally:
            try:
                ingest.stop()
            except Exception:
                log.exception("ingest shutdown failed")
            finally:
                # Ingest.run emits the audio EOF sentinel, which propagates
                # VAD -> ASR -> MT. Give in-flight VOD requests time to finish.
                grace_s = max(15.0, min(120.0,
                    cfg.asr.response_timeout_s
                    + cfg.translate.timeout_s
                    + cfg.translate.retry_timeout_s))
                from .windows_subprocess import stopping
                if stopping.is_set():
                    from .gui_control import WORKER_DRAIN_S
                    grace_s = WORKER_DRAIN_S
                deadline = time.monotonic() + grace_s
                sentinels = set()
                while time.monotonic() < deadline and any(t.is_alive() for t in threads):
                    for index, (producer, edge) in enumerate(zip(threads, queues.values())):
                        consumer = threads[index + 1] if index + 1 < len(threads) else None
                        if consumer is None or not consumer.is_alive():
                            # A dead consumer cannot release its blocked producer.
                            for _ in range(edge.qsize()):
                                try:
                                    edge.get_nowait()
                                except queue.Empty:
                                    break
                        elif not producer.is_alive() and index not in sentinels:
                            try:
                                edge.put_nowait(None)
                            except queue.Full:
                                pass
                            else:
                                sentinels.add(index)
                    for thread in threads:
                        thread.join(timeout=min(0.025, max(0., deadline - time.monotonic())))
                for thread in threads:
                    if thread.is_alive():
                        thread.join(timeout=max(0., deadline - time.monotonic()))
                active = [thread.name for thread in threads if thread.is_alive()]
                if active:
                    log.warning("shutdown grace expired; worker threads still active: %s", active)
                else:
                    log.info("all pipeline workers exited before model shutdown")
                record_event("pipeline_cleanup", active_threads=active)
                for label, worker in (("Voxtral", asr), ("PLaMo", translator)):
                    close = getattr(worker, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            log.exception("%s shutdown failed", label)
        log.info("done: %s", stats.summary())
