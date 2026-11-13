
"""Wire the stages together with threads and queues."""

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Callable

from .asr.base import ASR
from .asr.quality_guard import (
    foreign_audio_transcript_problem,
    transcript_problem,
)
from .asr_correction import correct_known_asr_terms
from .watame import correct_asr_name, reconcile_caption
from .captions import CaptionIndex, PendingCaptions
from .context_memory import repair_near_repeated_word
from .text_filter import is_usable_transcript
from .subtitle_clean import clean_english_output, has_generation_loop, normalized_cue, looks_like_prompt_echo
from .config import AsrConfig, Config, vad_for_source
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
    raise RuntimeError(f"pipeline worker '{name}' failed: {exc}") from exc


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
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    captions: CaptionIndex | PendingCaptions | None = None,
    vod_title: str = "",
    session_cache: SessionSubtitleCache | None = None,
    asr_cfg: AsrConfig | None = None,
    live: bool = False,
    dual=None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
    language_filter=None,
) -> None:
    """Transcribe Japanese speech; drop only confident English/Indonesian audio."""
    lid_checked = lid_dropped = lid_uncertain = 0
    lid_total_s = 0.0
    while True:
        item = in_q.get()

        if item is None:
            break
        if stop is not None and stop.is_set():
            continue  # discard queued ASR work after player shutdown

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            out_q.put(item)
            continue

        utt = item

        if get_generation is not None and utt.generation != get_generation():
            continue
        if (live and dual is not None and _expired_live_cue(
                utt, getattr(dual, "playback_position", None), live_late_window_s)):
            log.info("Skipping expired live ASR cue %.2f-%.2fs", utt.start_s, utt.end_s)
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

        # Language is determined from ORIGINAL audio, not Japanese-forced ASR.
        # Preserve an independent acoustic verdict for a second, evidence-based
        # guard after dual-ASR reconciliation; never infer EN/ID from text alone.
        verdict = None
        # Never send confidently English or Indonesian speech to Haas/Parakeet,
        # PLaMo, terminal, web or MPV. No English/Indonesian subtitles exist.
        if language_filter is not None:
            lid_started = time.perf_counter()
            verdict = language_filter.classify(utt.audio)
            lid_elapsed = time.perf_counter() - lid_started
            lid_checked += 1
            lid_total_s += lid_elapsed
            lid_dropped += int(verdict.drop)
            windows = getattr(verdict, "windows", ())
            votes = ",".join(
                f"{lang}:{prob:.2f}/ja:{ja_prob:.2f}/margin:{margin:.2f}"
                for lang, prob, ja_prob, margin in windows
            ) or "none"
            suspected_foreign = (
                verdict.language in ("en", "id")
                or any(lang in ("en", "id") for lang, *_ in windows)
            )
            if suspected_foreign and not verdict.drop:
                lid_uncertain += 1
            if verdict.drop:
                log.info(
                    "Filtered non-Japanese audio %.2f-%.2fs: %s "
                    "(confidence=%.3f, Japanese=%.3f, LID=%.3fs, "
                    "windows=%s); no subtitle",
                    utt.start_s, utt.end_s,
                    verdict.language, verdict.probability,
                    verdict.japanese_probability, lid_elapsed, votes,
                )
            elif suspected_foreign or verdict.reason in ("inference_error", "unavailable"):
                log.info(
                    "Language filter PASSED cue %.2f-%.2fs: "
                    "language=%s confidence=%.3f Japanese=%.3f "
                    "LID=%.3fs reason=%s windows=%s",
                    utt.start_s, utt.end_s, verdict.language,
                    verdict.probability, verdict.japanese_probability,
                    lid_elapsed, verdict.reason, votes,
                )
            else:
                log.debug(
                    "Language filter Japanese/short cue %.2f-%.2fs: "
                    "language=%s LID=%.3fs reason=%s",
                    utt.start_s, utt.end_s, verdict.language,
                    lid_elapsed, verdict.reason,
                )
            if lid_checked % 25 == 0:
                log.info(
                    "Language filter summary: checked=%d dropped=%d "
                    "foreign_or_mixed_passed=%d mean_LID=%.3fs",
                    lid_checked, lid_dropped, lid_uncertain,
                    lid_total_s / max(1, lid_checked),
                )
            if verdict.drop:
                continue

        # Strictly Japanese-only output: the legacy English passthrough
        # path has been removed. Classified EN/ID audio is dropped above;
        # uncertain speech remains in the Japanese recognition path.
        probe = None
        if get_generation is not None and utt.generation != get_generation():
            continue
        if stop is not None and stop.is_set():
            continue
        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        primary_elapsed_s = time.monotonic() - utt.timings.asr_start
        # Inspect synced captions before reconciliation, so LFM abstentions
        # can be rescued with independent, sufficiently aligned JP evidence.
        caption_match = (
            captions.match(utt.start_s, utt.end_s)
            if captions is not None else None
        )
        choice = (dual.select_segment(utt, text, vod_title,
                                      lambda: (in_q.qsize() > (1 if live else 5)
                                               or out_q.qsize() > (1 if live else 5)), get_generation,
                                      getattr(asr, "last_stop", None),
                                      primary_elapsed_s,
                                      probe,
                                      caption_match)
                  if dual is not None else None)
        if choice is not None:
            text = choice
            if not text:
                log.info(
                    "VAD-detected speech %.2f-%.2fs omitted by dual-ASR quality selection "
                    "(VAD confidence=%.2f)",
                    utt.start_s, utt.end_s, utt.vad_confidence or 0.0,
                )
                continue
        # Do not re-decode music, low-confidence noise or live audio.
        # Edge silence sometimes helps Haas terminate an omitted short cue.
        duration = utt.end_s - utt.start_s
        if (choice is None and not live and asr_cfg is not None and asr_cfg.retry_empty_speech
                and not getattr(asr, "last_failed", False)
                and not utt.probe and not is_usable_transcript(text)
                and utt.vad_confidence is not None
                and utt.vad_confidence >= asr_cfg.retry_min_confidence
                and asr_cfg.retry_min_duration_s <= duration <= asr_cfg.retry_max_duration_s):
            import numpy as np
            pad = int(0.2 * SAMPLE_RATE)
            candidate = asr.transcribe(np.pad(utt.audio, (pad, pad)), False)
            if is_usable_transcript(candidate):
                text = candidate
                log.info("Recovered VAD-positive silent Haas result at %.2f-%.2fs",
                         utt.start_s, utt.end_s)
            else:
                log.debug("VAD-positive Haas retry produced no usable speech at %.2fs",
                          utt.start_s)
        utt.timings.asr_end = time.monotonic()

        # Caption evidence can correct a proper name regardless of whether
        # the reconciler selected Haas or Parakeet. Previously a non-None
        # dual-ASR selection skipped ALL caption matching and name repairs.
        if text and caption_match is not None and caption_match.reliable:
            text = reconcile_caption(text, caption_match.text)

        if (
            choice is None and not text and caption_match is not None
            and caption_match.reliable
            and is_usable_transcript(caption_match.text)
        ):
            text = caption_match.text
            log.debug(
                "using aligned YouTube caption because Haas returned no text "
                "(%.2f-%.2fs, coverage=%.0f%%): %s",
                utt.start_s, utt.end_s, caption_match.coverage * 100, text,
            )

        if not text or not is_usable_transcript(text):
            log.info(
                "VAD-detected speech %.2f-%.2fs produced no usable Japanese transcript "
                "(Haas/Parakeet/caption stage, VAD confidence=%.2f)",
                utt.start_s, utt.end_s, utt.vad_confidence or 0.0,
            )
            continue
        quality_issue = transcript_problem(text, utt.end_s - utt.start_s)
        if quality_issue:
            log.warning("Suppressed suspect ASR at %.2f-%.2fs: %s", utt.start_s, utt.end_s, quality_issue)
            continue
        # A majority-foreign acoustic verdict plus unmistakably broken
        # Japanese-forced text is stronger evidence than ASR script alone.
        # Preserve all cues with confident Japanese audio or JP captions.
        acoustic_issue = foreign_audio_transcript_problem(
            text, verdict, caption_match,
        )
        if acoustic_issue:
            log.warning(
                "Suppressed foreign-phonetic ASR %.2f-%.2fs: %s "
                "(LID=%s confidence=%.3f; no translation)",
                utt.start_s, utt.end_s, acoustic_issue,
                getattr(verdict, "language", "?"),
                getattr(verdict, "probability", 0.0),
            )
            continue

        # Corrections must apply to the FINAL selected Japanese hypothesis.
        # These rules use explicit caption/video context and do not synthesize
        # new speech; never repair phonetic Watanabe against an explicit caption.
        corrected = correct_asr_name(
            text,
            vod_title=vod_title,
            caption=caption_match.text if caption_match else "",
        )
        if corrected != text:
            log.info("Watame ASR name correction: %s -> %s", text, corrected)
            text = corrected

        repaired = correct_known_asr_terms(
            text,
            video_context=vod_title,
            caption=caption_match.text if caption_match and caption_match.reliable else "",
        )
        if repaired != text:
            log.info("Verified context ASR term: %r -> %r", text, repaired)
            text = repaired

        if get_generation is not None and utt.generation != get_generation():
            continue

        if dual is not None:
            dual.remember(utt.generation, text)
        out_q.put(
            Transcript(
                utt.start_s,
                utt.end_s,
                text,
                utt.timings,
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
    live: bool = False,
    playback_position: Callable[[], float | None] | None = None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
) -> None:
    """Translate transcripts while retaining recent context and forwarding progress."""
    history_limit = max(context_lines, getattr(translator, "memory_lines", context_lines))
    history: deque[tuple[str, str]] = deque(maxlen=history_limit)
    memory_generation: int | None = None

    while True:
        item = in_q.get()

        if item is None:
            break
        if stop is not None and stop.is_set():
            continue  # do not translate queued cues after player shutdown

        if isinstance(item, AudioProgress):
            if get_generation is not None and item.generation != get_generation():
                continue

            if item.reset or (
                memory_generation is not None
                and item.generation != memory_generation
            ):
                history.clear()
            memory_generation = item.generation

            out_q.put(item)
            continue

        tr = item

        if get_generation is not None and tr.generation != get_generation():
            continue
        if live and _expired_live_cue(tr, playback_position, live_late_window_s):
            log.info("Skipping expired live PLaMo cue %.2f-%.2fs", tr.start_s, tr.end_s)
            continue

        # Seek resets are normally carried by AudioProgress. Also protect
        # against a new generation reaching this queue before its marker.
        if memory_generation is not None and tr.generation != memory_generation:
            history.clear()
        memory_generation = tr.generation

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not call PLaMo again for the same VOD cue.
        if isinstance(tr, Subtitle):
            if tr.ja and tr.en and tr.en != "[translation failed]":
                history.append((tr.ja, tr.en))
            tr.timings.mt_start = time.monotonic()
            tr.timings.mt_end = tr.timings.mt_start
            out_q.put(tr)
            continue

        # Guard the translation stage as well: imported/external Transcript
        # objects must not bypass the existing deterministic ASR checks.
        issue = transcript_problem(tr.ja, max(0., tr.end_s - tr.start_s))
        if issue:
            log.warning(
                "Skipping PLaMo for invalid transcript %.2f-%.2fs: %s",
                tr.start_s, tr.end_s, issue,
            )
            continue

        tr.timings.mt_start = time.monotonic()
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
                en = translator.translate(tr.ja, list(history))
            except Exception as e:
                log.error("translation failed: %s", e)
                en = "[translation failed]"

            if get_generation is not None and tr.generation != get_generation():
                continue

            if en and en != "[translation failed]":
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

            if en and en != "[translation failed]":
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
        if suppressed:
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
    dual=None,
    language_filter=None,
) -> None:
    """Run ingestion, VAD, ASR and translation."""

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
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    text_q: queue.Queue[Transcript | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)

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
            args=("vad", run_segmenter, failures, vad_cfg, audio_q, utt_q,
                  lambda: ingest.generation),
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
                dual,
                cfg.mpv.live_late_window_s,
                stop,
                language_filter,
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
                getattr(dual, "playback_position", None),
                cfg.mpv.live_late_window_s,
                stop,
            ),
            name="translate",
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    stats = Stats()
    last_subtitle_generation: int | None = None
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s
    last_activity = started
    last_warning = started
    last_guard_key = ""
    last_guard_end = -float("inf")
    last_guard_generation: int | None = None
    identical_count = 0

    try:
        while not stop.is_set():
            _raise_worker_failure(failures)
            try:
                item = sub_q.get(timeout=0.5)
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
                    continue

                stats.add(item, deadline_s=cfg.mpv.delay_s if ingest.source.live else None)

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
                    raise RuntimeError(
                        "Haas/PLaMo pipeline stalled with queued work: " + str(backlog)
                    )
                if backlog and now - last_activity > 30.0 and now - last_warning > 30.0:
                    last_warning = now
                    log.warning("Haas/PLaMo pipeline waiting with queued work: %s", backlog)

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
        # Stop producers and MPV first; never close Haas or PLaMo while
        # an ASR/MT thread is still actively using the model.
        stop.set()
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
                    + cfg.dual_asr.vod_timeout_s
                    + cfg.reconciler.vod_timeout_s
                    + cfg.translate.timeout_s
                    + cfg.translate.retry_timeout_s))
                deadline = time.monotonic() + grace_s
                # Main loop is no longer rendering subtitles: keep draining
                # completed cues so MT cannot block forever on a full queue.
                # Stop-aware ASR/MT stages discard pending work and propagate
                # their existing EOF sentinel when producers exit.
                while time.monotonic() < deadline and any(t.is_alive() for t in threads):
                    try:
                        sub_q.get(timeout=0.10)
                    except queue.Empty:
                        pass
                for thread in threads:
                    if thread.is_alive():
                        thread.join(timeout=max(0., deadline - time.monotonic()))
                active = [thread.name for thread in threads if thread.is_alive()]
                if active:
                    log.warning("shutdown grace expired; worker threads still active: %s", active)
                else:
                    log.info("all pipeline workers exited before model shutdown")
                for label, worker in (("Haas", asr), ("PLaMo", translator)):
                    close = getattr(worker, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            log.exception("%s shutdown failed", label)
        log.info("done: %s", stats.summary())
