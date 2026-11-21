
"""Wire the stages together with threads and queues."""

import logging
import math
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

from .asr.base import ASRBackend
from .asr.quality_guard import transcript_problem
from .config import Config, vad_for_source
from .gui_control import emit_event
from .ingest import Ingest
from .output.base import Output
from .runtime_diagnostics import increment_counter, record_event, update_metrics
from .session_cache import SessionSubtitleCache
from .subtitle_clean import (
    clean_english_output,
    english_output_issue,
    has_generation_loop,
    looks_like_prompt_echo,
    normalized_cue,
    sanitize_english_output,
)
from .translate.base import TranslatorUnavailable
from .translate.gemma4 import useful_context
from .types import (
    SAMPLE_RATE,
    AudioChunk,
    AudioProgress,
    SourceEOF,
    Subtitle,
    Transcript,
    Utterance,
)
from .vad import run_segmenter

log = logging.getLogger(__name__)




def _create_vad_segmenter(vad_cfg, vad_backend_name: str):
    """Create a VAD segmenter using the configured backend via registry.
    
    Falls back to direct Segmenter instantiation if the registry is unavailable.
    """
    try:
        from .registry import get_default_registry
        registry = get_default_registry()
        return registry.create_vad(vad_backend_name, vad_cfg)
    except (ValueError, ImportError):
        # Registry unavailable or backend not registered — fall back to direct creation
        log.warning(
            "VAD registry unavailable for '%s', falling back to direct Segmenter",
            vad_backend_name,
        )
        from .vad import Segmenter
        return Segmenter(vad_cfg)


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


def _rejection_reason(asr, text) -> str:
    reason = getattr(asr, "last_rejection", None)
    if isinstance(reason, str) and reason:
        return reason
    return "empty" if not isinstance(text, str) or not text.strip() else "unsupported_language"


def _raw_transcript(asr) -> str:
    raw = getattr(asr, "last_raw_transcript", "")
    return raw if isinstance(raw, str) else ""


ASR_CONTEXT_GAP_S = 15.0
_SENTENCE_END = re.compile(r"[。．.！!？?…‥」』）)】♪〜～ー]\s*$")
MERGE_TICK_S = 0.25
_TICK = object()


class WorkerState:
    """Thread-safe explicit inference state for a pipeline worker.

    Replaces heuristic stall detection (inferring from queue depth and
    last_activity timestamps) with actual worker-reported state.

    Each worker (ASR, translation) has one instance. The worker calls
    enter_inference() before starting inference and leave_inference() when
    done. The watchdog reads the state to distinguish:
    - Active inference within its deadline
    - Active inference exceeding its deadline (stall)
    - Idle worker with queued work (worker not consuming)
    - Idle worker with no work (natural silence)
    """

    __slots__ = (
        "_lock",
        "active",
        "generation",
        "inference_start",
        "last_completion",
        "last_queue_get",
        "utterance_id",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self.utterance_id: int | None = None
        self.generation: int | None = None
        self.inference_start: float | None = None
        self.last_completion: float | None = None
        self.last_queue_get: float | None = None

    def enter_inference(self, utterance_id: int, generation: int) -> None:
        """Called before the worker begins ASR or translation inference."""
        with self._lock:
            self.active = True
            self.utterance_id = utterance_id
            self.generation = generation
            self.inference_start = time.monotonic()

    def leave_inference(self, success: bool = True) -> None:
        """Called after inference completes (success or failure)."""
        with self._lock:
            self.active = False
            self.utterance_id = None
            self.generation = None
            self.inference_start = None
            if success:
                self.last_completion = time.monotonic()

    def record_queue_get(self, utterance_id: int) -> None:
        """Called when the worker dequeues an item (before inference)."""
        with self._lock:
            self.last_queue_get = time.monotonic()
            self.utterance_id = utterance_id

    def is_stalled(self, deadline_s: float) -> bool:
        """Check if the worker is stuck in inference past its deadline."""
        with self._lock:
            if not self.active:
                return False
            if self.inference_start is None:
                return False
            elapsed = time.monotonic() - self.inference_start
            return elapsed > deadline_s

    def stale_generation(self, current_gen: int) -> bool:
        """Check if the worker is processing a stale generation."""
        with self._lock:
            if not self.active:
                return False
            return self.generation is not None and self.generation != current_gen

    def summary(self) -> dict:
        """Return a snapshot of the current state for diagnostics."""
        with self._lock:
            result: dict = {
                "active": self.active,
                "utterance_id": self.utterance_id,
                "generation": self.generation,
            }
            if self.inference_start is not None:
                result["inference_elapsed_s"] = round(
                    time.monotonic() - self.inference_start, 2
                )
            if self.last_completion is not None:
                result["last_completion_ago_s"] = round(
                    time.monotonic() - self.last_completion, 2
                )
            if self.last_queue_get is not None:
                result["last_queue_get_ago_s"] = round(
                    time.monotonic() - self.last_queue_get, 2
                )
            return result


def _ticking(in_q: queue.Queue, interval_s: float):
    """Queue items, with a tick whenever none arrives within the interval."""
    while True:
        try:
            yield in_q.get(timeout=interval_s)
        except queue.Empty:
            yield _TICK


def _join_fragments(items, max_gap_s: float, max_span_s: float, due=None, on_hold=None):
    """Hold a transcript that ends mid-sentence and join it with a closely following one.

    Progress markers that could precede the continuation are parked behind the
    held cue, so no marker ever claims time the held cue still covers. With
    ``due``, a tick releases the held cue once ``due(held)`` says it cannot wait.
    When a ``SourceEOF`` arrives, any held fragment is yielded immediately.
    When ``on_hold`` is provided, it is called when a fragment begins being held.
    """
    held: Transcript | None = None
    parked: list[AudioProgress] = []
    for item in items:
        if item is _TICK:
            if held is not None:
                age_s = time.monotonic() - held.timings.audio_read if held.timings.audio_read else None
                if due is not None and due(held):
                    reason = "timeout" if age_s is not None and age_s > 3.0 else "playback_near"
                    log.debug(
                        "Releasing held fragment (%s): %.2f-%.2fs age=%.1fs ja=%r",
                        reason, held.start_s, held.end_s,
                        age_s if age_s is not None else -1, held.ja[:80],
                    )
                    increment_counter("released_fragments")
                    yield held
                    yield from parked
                    parked.clear()
                    held = None
            continue
        if isinstance(item, SourceEOF):
            # Flush any held fragment on EOF so it reaches translation
            if held is not None:
                log.debug(
                    "Flushing held fragment on EOF: %.2f-%.2fs ja=%r",
                    held.start_s, held.end_s, held.ja[:80],
                )
                increment_counter("released_fragments_eof")
                yield held
                yield from parked
                parked.clear()
                held = None
            yield item
            continue
        if held is not None:
            if (isinstance(item, AudioProgress) and not item.reset and item.generation == held.generation
                    and item.sample / SAMPLE_RATE <= held.end_s + max_gap_s):
                parked.append(item)
                continue
            if (type(item) is Transcript and item.generation == held.generation
                    and item.start_s - held.end_s <= max_gap_s and item.end_s - held.start_s <= max_span_s):
                item = Transcript(held.start_s, item.end_s, held.ja + item.ja, held.timings,
                                  generation=held.generation, speech_start_s=held.speech_start_s)
                increment_counter("merged_fragments")
            else:
                yield held
                yield from parked
                parked.clear()
            held = None
        if (type(item) is Transcript and not _SENTENCE_END.search(item.ja)
                and not (due is not None and due(item))):
            held = item
            if on_hold is not None:
                on_hold(held)
            continue
        yield item
        yield from parked
        parked.clear()


def run_asr(
    asr: ASRBackend,
    in_q: "queue.Queue[Utterance | AudioProgress | SourceEOF | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | SourceEOF | None]",
    get_generation: Callable[[], int] | None = None,
    session_cache: SessionSubtitleCache | None = None,
    live: bool = False,
    playback_position: Callable[[], float | None] | None = None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
    language_filter=None,
    gate=None,
    record_speech_cb: Callable[[float, float, int], int | None] | None = None,
    name_fusion=None,
    worker_state: "WorkerState | None" = None,
) -> None:
    """Recognize complete VAD utterances, retaining their original media times."""
    import numpy as np

    current_generation = None
    last_accepted_end_s: float | None = None
    remember = getattr(asr, "remember", None)
    forget = getattr(asr, "forget", None)
    set_context = getattr(asr, "set_context", None)

    def refresh_name_hints():
        if name_fusion is not None and callable(set_context):
            # These are candidates for decoding, not recognized transcript text.
            set_context(name_fusion.prompt())

    refresh_name_hints()
    # Relative live timestamps can be reused after reconnects. A VOD cache
    # must never supply cues for a different live generation.
    if live:
        session_cache = None

    def cancelled(generation):
        return ((stop is not None and stop.is_set())
                or (get_generation is not None and generation != get_generation()))

    def reset_generation(generation):
        nonlocal current_generation, last_accepted_end_s
        last_accepted_end_s = None
        cancel = getattr(asr, "cancel_pending", None)
        if callable(cancel):
            cancel()
        if name_fusion is not None:
            name_fusion.reset()
            refresh_name_hints()
        current_generation = generation

    def accepted(text, end_s, *, vad_confidence=None, probe=False):
        nonlocal last_accepted_end_s
        last_accepted_end_s = end_s
        if callable(remember):
            remember(text)
        # Name memory needs evidence independent of model output: only a
        # non-probe VAD-positive utterance with adequate speech confidence
        # may count toward the repeated-observation threshold. A cached or
        # low-confidence ASR guess cannot seed self-reinforcing prompts.
        if (name_fusion is not None and not probe
                and vad_confidence is not None and vad_confidence >= 0.35
                and name_fusion.observe(text)):
            refresh_name_hints()

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
            if item.reset or item.generation != current_generation:
                reset_generation(item.generation)
            out_q.put(item)
            continue
        utt = item
        if utt.generation != current_generation:
            reset_generation(utt.generation)
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
            accepted(cached.ja, utt.end_s, vad_confidence=utt.vad_confidence,
                     probe=utt.probe)
            # Record this speech region as pending so the watermark knows
            # about it. The cached subtitle must complete this exact pending
            # entry, not some other utterance with similar timestamps.
            cached_speech_id = None
            if record_speech_cb is not None:
                result = record_speech_cb(utt.start_s, utt.end_s, utt.generation)
                if result is not None:
                    cached_speech_id = result
            out_q.put(replace(cached, generation=utt.generation, timings=utt.timings,
                              speech_id=cached_speech_id))
            continue
        if cancelled(utt.generation):
            continue
        if (last_accepted_end_s is not None
                and utt.start_s - last_accepted_end_s > ASR_CONTEXT_GAP_S):
            if callable(forget):
                forget()
            if name_fusion is not None:
                name_fusion.reset()
                refresh_name_hints()
            last_accepted_end_s = None
        # Record queue wait time: time from segmented to ASR start (waiting for inference slot)
        utt.timings.asr_queue_wait = time.monotonic() - utt.timings.segmented
        utt.timings.asr_start = time.monotonic()
        # Report to watchdog: this utterance is now in-flight for ASR
        if worker_state is not None:
            worker_state.record_queue_get(id(utt))
            worker_state.enter_inference(id(utt), utt.generation)
        asr_max_retries = 2
        asr_attempt = 0
        text = ""
        reviewed = False
        while True:
            try:
                with (gate.operation("asr", current=get_generation, generation=utt.generation)
                      if gate is not None else nullcontext()):
                    if cancelled(utt.generation):
                        break
                    if live and _expired_live_cue(utt, playback_position, live_late_window_s):
                        increment_counter("expired")
                        break
                    text = asr.transcribe(utt.audio, utt.probe)
                    reason = _rejection_reason(asr, text)
                    if not text and reason in ("context_echo", "repeated_output"):
                        reviewed = True
                        remaining = asr.cfg.response_timeout_s - (time.monotonic() - utt.timings.asr_start)
                        retry = (asr.cfg.context_retry_limit and remaining > 0
                                 and not cancelled(utt.generation)
                                 and not (live and _expired_live_cue(utt, playback_position, live_late_window_s)))
                        log.info("ASR_LEAK_DETECTED reason=%s segment_start=%.2f segment_end=%.2f "
                                 "match_score=%.2f action=%s", reason, utt.start_s, utt.end_s,
                                 asr.last_match_score, "retry_without_context" if retry else "discard")
                        if retry:
                            text = asr.recover_context_echo(utt.audio, utt.probe, utt.vad_confidence, remaining)
                        if cancelled(utt.generation) or (live and _expired_live_cue(
                                utt, playback_position, live_late_window_s)):
                            text = ""
                        log.info("ASR_LEAK_OUTCOME segment_start=%.2f segment_end=%.2f outcome=%s reason=%s",
                                 utt.start_s, utt.end_s, "accepted" if text else "discarded",
                                 "audio_recovery" if text else getattr(asr, "last_rejection", None) or "stale")
            except CancelledError:
                break
            if cancelled(utt.generation):
                break
            if text and text.strip():
                break
            reason = _rejection_reason(asr, text)
            last_rejection = getattr(asr, "last_rejection", None)
            if (not reviewed and asr_attempt < asr_max_retries
                    and isinstance(last_rejection, str)
                    and last_rejection == "error"):
                asr_attempt += 1
                record_event("asr_retry", generation=utt.generation, attempt=asr_attempt,
                             source_start_s=utt.start_s, source_end_s=utt.end_s,
                             reason=reason)
                time.sleep(0.25 * asr_attempt)
                continue
            increment_counter("asr_rejected")
            record_event("asr_rejected", generation=utt.generation, reason=reason,
                         raw_text="" if reviewed else _raw_transcript(asr)[:160], vad_confidence=utt.vad_confidence,
                         source_start_s=utt.start_s, source_end_s=utt.end_s,
                         retries=asr_attempt)
            log.debug("Rejected ASR %.2f-%.2fs (%s): %s",
                      utt.start_s, utt.end_s, reason, _raw_transcript(asr)[:160])
            break
        if cancelled(utt.generation):
            if worker_state is not None:
                worker_state.leave_inference(success=False)
            continue
        # Report to watchdog: ASR inference finished
        if worker_state is not None:
            worker_state.leave_inference(success=bool(text and text.strip()))
        utt.timings.asr_end = time.monotonic()
        if not (isinstance(text, str) and text.strip()):
            continue
        if language_filter is not None:
            duration_s = utt.end_s - utt.start_s
            if not language_filter.filter(text, duration_s):
                record_event("language_filtered", generation=utt.generation,
                             source_start_s=utt.start_s, source_end_s=utt.end_s)
                continue
        # Record client-side HTTP timing from Haas for stall diagnosis
        last_timing = getattr(asr, "last_timing", None)
        if last_timing is not None:
            utt.timings.asr_http_s = last_timing.get("http_request_s", 0.0)
        update_metrics(stages={"asr_s": utt.timings.asr_end - utt.timings.asr_start})
        record_event("asr_complete", generation=utt.generation,
                     source_start_s=utt.start_s, speech_start_s=utt.speech_start_s,
                     source_end_s=utt.end_s, pcm_start_sample=round(utt.start_s * SAMPLE_RATE),
                     segmented_at=utt.timings.segmented,
                     asr_completed_at=utt.timings.asr_end)
        issue = transcript_problem(text, utt.end_s - utt.start_s)
        if issue:
            increment_counter("dropped")
            record_event("asr_suppressed", reason=issue, generation=utt.generation)
            continue
        # Familiar hallucinated outros on faint, sub-second non-speech are
        # insufficient evidence for a subtitle. Never reject a normal short reply.
        if (text.strip(" 。！？!?") in ("ご視聴ありがとうございました", "チャンネル登録よろしくお願いします")
                and utt.end_s - utt.start_s < 1.3
                and utt.vad_confidence is not None and utt.vad_confidence < .55
                and len(utt.audio) and float(np.mean(np.square(utt.audio))) < .0001):
            increment_counter("dropped")
            continue
        if cancelled(utt.generation):
            continue
        accepted(text, utt.end_s, vad_confidence=utt.vad_confidence,
                 probe=utt.probe)
        out_q.put(Transcript(utt.start_s, utt.end_s, text, utt.timings,
                             generation=utt.generation, speech_start_s=utt.speech_start_s))
    out_q.put(None)


def run_translate(
    translator: object | None,
    context_lines: int,
    in_q: "queue.Queue[Transcript | Subtitle | AudioProgress | SourceEOF | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | SourceEOF | None]",
    get_generation: Callable[[], int] | None = None,
    live: bool = False,
    playback_position: Callable[[], float | None] | None = None,
    live_late_window_s: float = 7.0,
    stop: threading.Event | None = None,
    gate=None,
    merge: tuple[float, float] | None = None,
    live_merge_margin_s: float = 1.5,
    record_speech_cb: Callable[[float, float, int], None] | None = None,
    complete_speech_cb: Callable[[float, float, int], None] | None = None,
    worker_state: "WorkerState | None" = None,
) -> None:
    """Translate transcripts while retaining recent context and forwarding progress.

    ``merge`` is ``(max_gap_s, max_span_s)`` for joining mid-sentence fragments.
    Live releases a held fragment once playback is within the margin of its onset.
    """
    consecutive_failures = 0
    history: deque[tuple[str, str]] = deque(maxlen=max(1, context_lines))
    memory_generation: int | None = None
    last_speech_end_s: float | None = None
    context_max_gap_s = getattr(getattr(translator, "cfg", None), "context_max_gap_s", 15.0)
    set_position = getattr(translator, "set_position", None)
    # VOD merge timeout: release held fragments after this many seconds of real-time waiting
    _vod_timeout_raw = getattr(getattr(translator, "cfg", None), "vod_merge_timeout_s", None)
    vod_merge_timeout_s = 5.0
    if isinstance(_vod_timeout_raw, (int, float)) and 0 < _vod_timeout_raw <= 30:
        vod_merge_timeout_s = float(_vod_timeout_raw)
    items = iter(in_q.get, object())
    if merge is not None and live and playback_position is not None:
        # A stalled/discontinuous MPV clock must not hold a translated fragment
        # indefinitely while AudioProgress piles up behind it.
        live_held_id: int | None = None
        live_held_at = 0.0

        def live_on_hold(held: Transcript) -> None:
            nonlocal live_held_id, live_held_at
            live_held_id = id(held)
            live_held_at = time.monotonic()

        def due(held: Transcript) -> bool:
            position = playback_position()
            onset = held.speech_start_s if held.speech_start_s is not None else held.start_s
            if position is None or position >= onset - live_merge_margin_s:
                return True
            return (id(held) == live_held_id
                    and time.monotonic() - live_held_at >= max(3.0, vod_merge_timeout_s))

        items = _join_fragments(
            _ticking(in_q, MERGE_TICK_S), *merge, due=due, on_hold=live_on_hold,
        )
    elif merge is not None:
        # VOD mode: use ticking with a real-time timeout so held fragments
        # are released when no continuation arrives within vod_merge_timeout_s.
        # This prevents the circular deadlock where a held fragment blocks
        # progress markers, which exhausts translation lead, which auto-pauses
        # MPV, which blocks FFmpeg, which never produces the continuation.
        held_since: dict[int, float] = {}  # generation -> monotonic time when held
        def vod_due(held: Transcript) -> bool:
            ts = held_since.get(held.generation)
            if ts is None:
                return False
            return time.monotonic() - ts >= vod_merge_timeout_s
        def vod_on_hold(held: Transcript) -> None:
            held_since[held.generation] = time.monotonic()
        items = _join_fragments(_ticking(in_q, MERGE_TICK_S), *merge, due=vod_due, on_hold=vod_on_hold)

    for item in items:
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
                last_speech_end_s = None
                consecutive_failures = 0
            memory_generation = item.generation

            out_q.put(item)
            continue

        tr = item

        if get_generation is not None and tr.generation != get_generation():
            continue
        if live and _expired_live_cue(tr, playback_position, live_late_window_s):
            log.info("Skipping expired live cue %.2f-%.2fs", tr.start_s, tr.end_s)
            increment_counter("expired")
            continue

        # Seek resets are normally carried by AudioProgress. Also protect
        # against a new generation reaching this queue before its marker.
        if memory_generation is not None and tr.generation != memory_generation:
            history.clear()
            last_speech_end_s = None
            consecutive_failures = 0
        memory_generation = tr.generation
        if last_speech_end_s is not None and tr.start_s - last_speech_end_s > context_max_gap_s:
            history.clear()
        last_speech_end_s = tr.end_s

        # A cached Subtitle already includes its Japanese transcript and
        # English translation; do not translate the same VOD cue again.
        if isinstance(tr, Subtitle):
            cleaned = sanitize_english_output(tr.en)
            if not cleaned:
                increment_counter("dropped")
                continue
            tr = replace(tr, en=cleaned)
            if useful_context(tr.ja):
                # Cached subtitles already passed validation, safe to add to history
                history.append((tr.ja, cleaned))
            tr.timings.mt_start = time.monotonic()
            tr.timings.mt_end = tr.timings.mt_start
            out_q.put(tr)
            continue

        # Record queue wait time: time from ASR completion to translation start
        tr.timings.mt_queue_wait = time.monotonic() - tr.timings.asr_end
        tr.timings.mt_start = time.monotonic()
        en = ""
        suppressed = False

        # Record this speech region as pending translation so the watermark
        # knows not to advance past it until the subtitle is published.
        speech_id = None
        if record_speech_cb is not None:
            result = record_speech_cb(tr.start_s, tr.end_s, tr.generation)
            if result is not None:
                speech_id = result

        def release_pending() -> None:
            """Finalize a reserved interval when this cue will not be published."""
            if speech_id is not None and complete_speech_cb is not None:
                complete_speech_cb(
                    tr.start_s, tr.end_s, tr.generation, speech_id=speech_id,
                )

        if translator is not None:
            # Report to watchdog: this transcript is now in-flight for translation
            if worker_state is not None:
                worker_state.record_queue_get(id(tr))
                worker_state.enter_inference(id(tr), tr.generation)
            try:
                with (gate.operation("mt", current=get_generation, generation=tr.generation)
                      if gate is not None else nullcontext()):
                    if live and _expired_live_cue(tr, playback_position, live_late_window_s):
                        increment_counter("expired")
                        if worker_state is not None:
                            worker_state.leave_inference(success=False)
                        release_pending()
                        continue
                    model_started = time.monotonic()
                    if callable(set_position):
                        set_position(tr.start_s)
                    en = translator.translate(tr.ja, list(history) if context_lines else [])
                    update_metrics(stages={"mt_s": time.monotonic() - model_started})
                consecutive_failures = 0
            except CancelledError:
                if worker_state is not None:
                    worker_state.leave_inference(success=False)
                release_pending()
                continue
            except TranslatorUnavailable:
                if worker_state is not None:
                    worker_state.leave_inference(success=False)
                release_pending()
                if ((stop is not None and stop.is_set()) or
                        (get_generation is not None and tr.generation != get_generation())):
                    continue
                raise
            except Exception as e:
                if worker_state is not None:
                    worker_state.leave_inference(success=False)
                if ((stop is not None and stop.is_set()) or
                        (get_generation is not None and tr.generation != get_generation())):
                    release_pending()
                    continue
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    release_pending()
                    raise TranslatorUnavailable(f"Translation failed on three consecutive cues: {e}") from e
                log.error("translation failed: %s", e)
                suppressed = True

            # Translation inference finished (success, suppressed, or stale)
            if worker_state is not None:
                worker_state.leave_inference(success=not suppressed)
            if get_generation is not None and tr.generation != get_generation():
                release_pending()
                continue
            if stop is not None and stop.is_set():
                release_pending()
                continue
            if live and _expired_live_cue(tr, playback_position, live_late_window_s):
                increment_counter("expired")
                release_pending()
                continue

            # Track suppression reason for better diagnostics
            suppression_reason = None
            tcfg = getattr(translator, "cfg", None)
            allow_japanese = getattr(tcfg, "suppress_japanese_in_english", True) is False
            suppress_empty = getattr(tcfg, "suppress_empty_translation", True)
            if en:
                if has_generation_loop(en) or looks_like_prompt_echo(en):
                    suppression_reason = "generation_loop_or_prompt_echo"
                    log.warning(
                        "Suppressed repeated/prompt-like translation output: gen=%d source=%.2f-%.2fs raw=%r",
                        tr.generation, tr.start_s, tr.end_s, en[:120],
                    )
                    en, suppressed = "", True
                else:
                    en_cleaned = clean_english_output(
                        en,
                        remove_fillers=getattr(tcfg, "remove_fillers", True),
                        collapse_repetitions=getattr(tcfg, "collapse_repetitions", True),
                        allow_japanese=allow_japanese,
                    )
                    if not en_cleaned:
                        suppression_reason = "cleaned_to_empty"
                        log.debug(
                            "Translation cleaned to empty: gen=%d source=%.2f-%.2fs raw=%r",
                            tr.generation, tr.start_s, tr.end_s, en[:120],
                        )
                    en = en_cleaned

            en_sanitized = sanitize_english_output(en, allow_japanese=allow_japanese)
            if not en_sanitized and not suppression_reason:
                issue = english_output_issue(en, allow_japanese=allow_japanese)
                suppression_reason = issue or "sanitize_empty"
                log.debug(
                    "Translation rejected by sanitize (%s): gen=%d source=%.2f-%.2fs raw=%r",
                    suppression_reason, tr.generation, tr.start_s, tr.end_s, en[:120],
                )
            en = en_sanitized
            if not en:
                if suppress_empty:
                    suppressed = True
                else:
                    # Translation failed — do NOT fall back to Japanese as English.
                    # Showing Japanese in the English field is worse than no subtitle.
                    # The config option suppress_empty_translation=false originally
                    # intended to preserve content, but Japanese-in-English confuses
                    # viewers and breaks the MPV output module. Instead, suppress
                    # with a clear diagnostic so the failure is visible.
                    suppressed = True
                    log.info(
                        "Translation failed - suppressing subtitle (not falling back to Japanese): "
                        "gen=%d source=%.2f-%.2fs ja=%r",
                        tr.generation, tr.start_s, tr.end_s, tr.ja[:110],
                    )

            # Recognized Japanese stays useful context even when its English failed.
            # Only add clean translations to history — contaminated translations
            # would pollute future context. Empty English (from failures) still
            # gets recorded so the Japanese context is preserved.
            if useful_context(tr.ja):
                if en and english_output_issue(en, allow_japanese=allow_japanese):
                    # Translation was rejected — keep the Japanese context but
                    # use an empty English placeholder so future prompts don't
                    # recycle the bad translation.
                    history.append((tr.ja, ""))
                else:
                    # Clean translation or empty from failure — safe for history
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
                     suppressed=suppressed)
        if suppressed:
            increment_counter("dropped")
            reason_str = f" ({suppression_reason})" if suppression_reason else ""
            log.info(
                "Translated cue %.2f-%.2fs suppressed by English output guard%s; "
                "Japanese ASR recognized speech: %r",
                tr.start_s, tr.end_s, reason_str, tr.ja[:110],
            )
            # Clear pending speech so the watermark can advance past the suppressed region
            if complete_speech_cb is not None:
                complete_speech_cb(tr.start_s, tr.end_s, tr.generation, speech_id=speech_id)
            continue

        # Japanese fallback is no longer used; translation failures are suppressed.

        out_q.put(
            Subtitle(
                tr.start_s,
                tr.end_s,
                tr.ja,
                en,
                tr.timings,
                generation=tr.generation,
                speech_start_s=tr.speech_start_s,
                speech_id=speech_id,
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
    # ASR HTTP timing for stall diagnosis
    asr_http_sum_s: float = 0.0
    asr_http_max_s: float = 0.0
    asr_stall_count: int = 0  # utterances where HTTP request > 2s
    # Queue wait time tracking
    asr_queue_wait_sum_s: float = 0.0
    mt_queue_wait_sum_s: float = 0.0
    # Expired subtitle tracking
    expired_subtitles: int = 0
    # Latency percentiles
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    latency_p99_s: float = 0.0
    latency_median_s: float = 0.0

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
        # Fix deadline miss reporting: use a configurable default of 10s
        # when no explicit deadline is provided
        effective_deadline = deadline_s if deadline_s is not None else 10.0
        if total_s > effective_deadline:
            self.deadline_misses += 1
        self.asr_max_s = max(self.asr_max_s, asr_s)
        self.mt_max_s = max(self.mt_max_s, mt_s)
        self.total_max_s = max(self.total_max_s, total_s)
        # Track ASR HTTP timing for stall diagnosis
        asr_http_s = tm.asr_http_s
        if asr_http_s > 0:
            self.asr_http_sum_s += asr_http_s
            self.asr_http_max_s = max(self.asr_http_max_s, asr_http_s)
            if asr_http_s > 2.0:
                self.asr_stall_count += 1
        # Track queue wait times
        self.asr_queue_wait_sum_s += tm.asr_queue_wait
        self.mt_queue_wait_sum_s += tm.mt_queue_wait

    def _percentiles(self) -> tuple[float, float, float, float]:
        """Calculate latency percentiles from recent_lag_s."""
        if not self.recent_lag_s:
            return (0.0, 0.0, 0.0, 0.0)
        sorted_lag = sorted(self.recent_lag_s)
        n = len(sorted_lag)
        p50 = sorted_lag[n // 2]
        p95 = sorted_lag[min(n - 1, int(n * 0.95))]
        p99 = sorted_lag[min(n - 1, int(n * 0.99))]
        median = sorted_lag[n // 2] if n % 2 == 1 else (sorted_lag[n // 2 - 1] + sorted_lag[n // 2]) / 2
        return (p50, p95, p99, median)

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

        p50, p95, p99, median = self._percentiles()
        # Update percentile fields for diagnostics
        self.latency_p50_s = p50
        self.latency_p95_s = p95
        self.latency_p99_s = p99
        self.latency_median_s = median

        parts = [
            f"{self.lines} lines",
            f"latency median {median:.2f}s p95 {p95:.2f}s p99 {p99:.2f}s, deadline misses {self.deadline_misses}",
            fmt('asr', self.asr_sum_s, self.asr_max_s),
            fmt('mt', self.mt_sum_s, self.mt_max_s),
            fmt('latency', self.total_sum_s, self.total_max_s),
        ]
        # Queue wait times
        if self.asr_queue_wait_sum_s > 0 or self.mt_queue_wait_sum_s > 0:
            parts.append(
                f"queue_wait asr {self.asr_queue_wait_sum_s / self.lines:.2f}s "
                f"mt {self.mt_queue_wait_sum_s / self.lines:.2f}s"
            )
        if self.asr_http_sum_s > 0:
            parts.append(
                f"asr_http avg {self.asr_http_sum_s / self.lines:.2f}s "
                f"max {self.asr_http_max_s:.2f}s, stalls {self.asr_stall_count}"
            )
        if self.expired_subtitles > 0:
            parts.append(f"expired subtitles {self.expired_subtitles}")
        parts.append(f"max queue {maxq}")
        return " | ".join(parts)


def run(
    ingest: Ingest,
    cfg: Config,
    asr: ASRBackend,
    translator: object | None,
    outputs: list[Output],
    stop: threading.Event,
    session_cache: SessionSubtitleCache | None = None,
    playback_position: Callable[[], float | None] | None = None,
    language_filter=None,
    segmenter=None,
    gate=None,
    diagnostics=None,
    name_fusion=None,
) -> None:
    """Run ingestion, VAD, ASR and translation."""

    if ingest.source.live:
        session_cache = None

    vad_cfg = vad_for_source(cfg.vad, live=bool(ingest.source.live))
    vad_backend_name = cfg.models.vad_backend
    log.info(
        "VAD backend '%s' (%s): threshold=%.2f, min_speech=%dms, "
        "min_silence=%dms, pad=%dms, onset=%dms, soft_cap=%.1fs, hard_cap=%.1fs, "
        "VAD-negative probing=%s",
        vad_backend_name,
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

    # Build the record_speech callback for the contiguous watermark.
    # Needed by both run_asr (cached subtitles) and run_translate (new translations).
    record_speech_cb = None
    complete_speech_cb = None
    for out in outputs:
        rs = getattr(out, "record_speech", None)
        if rs is not None and record_speech_cb is None:
            record_speech_cb = rs
        cs = getattr(out, "complete_speech", None)
        if cs is not None and complete_speech_cb is None:
            complete_speech_cb = cs

    # Explicit worker state tracking for ASR and translation.
    # The watchdog reads these to distinguish active inference from idle/silence.
    asr_worker_state = WorkerState()
    mt_worker_state = WorkerState()

    threads = [
        threading.Thread(
            target=_guard_stage,
            args=("ingest", ingest.run, failures, audio_q),
            name="ingest",
            daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("vad", run_segmenter, failures,
                  segmenter if segmenter is not None else _create_vad_segmenter(vad_cfg, vad_backend_name),
                  audio_q, utt_q,
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
                session_cache,
                bool(ingest.source.live),
                playback_position,
                cfg.mpv.live_late_window_s,
                stop,
                language_filter,
                gate,
                record_speech_cb,
                name_fusion,
                asr_worker_state,
            ),
            name="asr",
            daemon=True,
        ),
    ]

    threads.append(
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
                # Live holds a fragment only while playback is still short of its onset.
                ((cfg.translate.merge_max_gap_s, cfg.translate.merge_max_s)
                 if cfg.translate.merge_fragments
                 and (not ingest.source.live or playback_position is not None) else None),
                cfg.translate.live_merge_margin_s,
                record_speech_cb,
                complete_speech_cb,
                mt_worker_state,
            ),
            name="translate",
            daemon=True,
        )
    )

    for thread in threads:
        thread.start()

    def release_discarded_subtitle(sub: Subtitle) -> None:
        """Avoid leaving coverage blocked when an output guard drops a cue."""
        if sub.speech_id is not None and complete_speech_cb is not None:
            complete_speech_cb(
                sub.start_s, sub.end_s, sub.generation, speech_id=sub.speech_id,
            )

    stats = Stats()
    last_subtitle_generation: int | None = None
    observed_generation = ingest.generation
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s
    next_diagnostics = started
    last_activity = started
    last_subtitle_at = started  # tracks last published subtitle, independent of AudioProgress
    last_warning = started
    last_guard_key = ""
    last_guard_start = -float("inf")
    last_guard_end = -float("inf")
    last_guard_generation: int | None = None
    identical_count = 0
    last_vod_diag = 0.0  # Throttle VOD diagnostics logging

    try:
        while not stop.is_set():
            # Generation checks in each worker discard an in-flight old cue.
            # The ASR worker flushes its adapter before the next generation;
            # a delayed main-loop tick must never cancel a new-generation cue.
            if ingest.generation != observed_generation:
                # A seek invalidates every in-flight utterance, subtitle and
                # dedup window from the previous generation. Drop stale state
                # so the new generation starts clean and old subtitles cannot
                # bleed into the new seek position.
                observed_generation = ingest.generation
                last_guard_key = ""
                last_guard_start = -float("inf")
                last_guard_end = -float("inf")
                last_guard_generation = None
                identical_count = 0
                last_subtitle_generation = None
                last_subtitle_at = time.monotonic()
                record_event("inference_generation_reset", generation=observed_generation)
                record_event("seek_reset", generation=observed_generation)
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
                    tcfg = getattr(translator, "cfg", None)
                    allow_jp = tcfg is not None and getattr(tcfg, "suppress_japanese_in_english", True) is False
                    cleaned = sanitize_english_output(item.en, allow_japanese=allow_jp)
                    if not cleaned:
                        release_discarded_subtitle(item)
                        increment_counter("dropped")
                        continue
                    item = replace(item, en=cleaned)
                elif translator is not None:
                    release_discarded_subtitle(item)
                    increment_counter("dropped")
                    continue

                if item.generation != last_guard_generation:
                    last_guard_key, last_guard_start, last_guard_end, identical_count = "", -float("inf"), -float("inf"), 0
                    last_guard_generation = item.generation
                key = normalized_cue(item.en) if item.en else ""
                # Timestamp-aware dedup: overlapping chunks re-publishing the
                # same speech are duplicates, but genuinely repeated speech in
                # separate (non-overlapping) utterances -- e.g. "go, go, go" --
                # must survive. Same text only counts as a repeat when the time
                # ranges overlap; identical intervals are a subset of that.
                overlap = bool(
                    key and last_guard_key
                    and key == last_guard_key
                    and item.start_s < last_guard_end - 0.05
                    and last_guard_start < item.end_s + 0.05,
                )
                if overlap:
                    identical_count += 1
                else:
                    identical_count = 1
                last_guard_key, last_guard_start, last_guard_end = key, item.start_s, item.end_s
                if (key and identical_count >
                        max(1, int(cfg.translate.max_identical_cues))):
                    log.info(
                        "Suppressed repeat subtitle: gen=%d count=%d overlap=%s text=%r",
                        item.generation, identical_count, overlap, item.en[:110],
                    )
                    release_discarded_subtitle(item)
                    increment_counter("dropped")
                    continue

                stats.add(item, deadline_s=cfg.mpv.delay_s if ingest.source.live else None)
                latency = max(0., time.monotonic() - item.timings.audio_read) if item.timings.audio_read else None
                update_metrics(counters={"latency_s": latency})
                # Track subtitle latency with full breakdown
                if latency is not None:
                    record_event("subtitle_latency", latency_s=round(latency, 3),
                                 generation=item.generation,
                                 asr_queue_wait=round(item.timings.asr_queue_wait, 3),
                                 mt_queue_wait=round(item.timings.mt_queue_wait, 3),
                                 asr_s=round(max(0, item.timings.asr_end - item.timings.asr_start), 3),
                                 mt_s=round(max(0, item.timings.mt_end - item.timings.mt_start), 3),
                                 asr_http_s=round(item.timings.asr_http_s, 3))
                    # Track late subtitles (> 10s)
                    if latency > 10.0:
                        increment_counter("subtitle_late")
                if latency is not None and latency > max(30., cfg.mpv.delay_s):
                    record_event("subtitle_latency_spike", latency_s=round(latency, 3), generation=item.generation)

                if session_cache is not None:
                    session_cache.add(item)

                record_event("subtitle_published", generation=item.generation,
                             speech_start_s=item.speech_start_s,
                             pcm_start_s=item.start_s, speech_end_s=item.end_s,
                             asr_completed_at=item.timings.asr_end,
                             translation_completed_at=item.timings.mt_end,
                             ja=item.ja[:110], en=item.en[:110])
                log.info(
                    "Published subtitle gen=%d %.2f-%.2fs ja=%r en=%r",
                    item.generation, item.start_s, item.end_s,
                    item.ja[:80], item.en[:80],
                )
                last_subtitle_at = time.monotonic()

                for out in outputs:
                    out.show(item)
                    # Advance the translation watermark to cover this published subtitle
                    advance_to_sub = getattr(out, "advance_to_subtitle", None)
                    if advance_to_sub is not None:
                        advance_to_sub(item.end_s, item.generation, speech_id=item.speech_id,
                                       subtitle_start_s=item.start_s)

            except queue.Empty:
                pass

            _raise_worker_failure(failures)
            now = time.monotonic()
            if _expected_vod_idle(ingest, outputs):
                # Pacing is intentional; do not suppress independent worker-health
                # diagnostics by constantly resetting last_warning.
                last_activity = now
            else:
                backlog = {name: q.qsize() for name, q in queues.items() if q.qsize()}
                if backlog and now - last_activity > 90.0:
                    record_event("pipeline_stall", queues=backlog, elapsed_s=now-last_activity)
                    raise RuntimeError(
                        "ASR/translation pipeline stalled with queued work: " + str(backlog)
                    )
                if backlog and now - last_activity > 30.0 and now - last_warning > 30.0:
                    last_warning = now
                    log.warning("ASR/translation pipeline waiting with queued work: %s", backlog)

            # Subtitle starvation watchdog: uses explicit WorkerState to detect
            # stalls without false positives from natural silence.
            # Works for both VOD and livestream.
            no_sub_s = now - last_subtitle_at
            if no_sub_s > 15.0 and ingest.generation == observed_generation:
                # Compute backlog once for all checks below.
                subtitle_backlog = {name: q.qsize() for name, q in queues.items() if q.qsize()}
                has_backlog = bool(subtitle_backlog)

                # Use explicit worker state to determine stall type.
                # WorkerState tracks actual inference entry/exit, not heuristics.
                asr_stalled = asr_worker_state.is_stalled(cfg.asr.response_timeout_s + 5.0)
                mt_stalled = mt_worker_state.is_stalled(cfg.translate.timeout_s + 5.0)
                asr_stale_gen = asr_worker_state.stale_generation(ingest.generation)
                mt_stale_gen = mt_worker_state.stale_generation(ingest.generation)
                any_worker_active = asr_worker_state.active or mt_worker_state.active

                # Classify the stall condition:
                # 1. Queued backlog — workers not consuming
                # 2. In-flight stall — worker stuck during inference
                # 3. Stale generation — worker processing old seek position
                # 4. Natural silence — no work, no active inference (NOT a stall)
                is_queued_stall = has_backlog and not any_worker_active
                # A hung inference can coexist with a full upstream queue.
                is_in_flight_stall = asr_stalled or mt_stalled
                is_stale_gen_stall = asr_stale_gen or mt_stale_gen
                # Natural silence: no backlog, no active worker, audio may or may
                # not be flowing. This is NOT a stall — do nothing.
                is_natural_silence = (
                    not has_backlog
                    and not any_worker_active
                    and not asr_stalled
                    and not mt_stalled
                )

                # Rate-limit: emit diagnostics at most every 30s per stall type
                can_diag = now - last_warning > 30.0

                confirmed_stall = is_in_flight_stall or is_stale_gen_stall or is_queued_stall
                if no_sub_s > 30.0 and confirmed_stall and can_diag:
                    stall_type = (
                        "in_flight" if is_in_flight_stall else
                        "stale_generation" if is_stale_gen_stall else "queued"
                    )
                    last_warning = now
                    record_event(
                        "subtitle_starvation",
                        queues=subtitle_backlog,
                        no_subtitle_for_s=round(no_sub_s, 1),
                        generation=ingest.generation,
                        stall_type=stall_type,
                        asr_state=asr_worker_state.summary(),
                        mt_state=mt_worker_state.summary(),
                    )
                    if is_in_flight_stall:
                        log.warning(
                            "Subtitle starvation (in-flight): no subtitle for %.0fs; "
                            "ASR stalled=%s MT stalled=%s; queued=%s",
                            no_sub_s, asr_stalled, mt_stalled, subtitle_backlog,
                        )
                    elif is_queued_stall:
                            log.warning(
                                "Subtitle starvation (queued): no subtitle for %.0fs "
                                "with queued work %s; workers idle",
                                no_sub_s, subtitle_backlog,
                            )
                    elif is_stale_gen_stall:
                            log.warning(
                                "Subtitle starvation (stale gen): no subtitle for %.0fs; "
                                "ASR stale=%s MT stale=%s",
                                no_sub_s, asr_stale_gen, mt_stale_gen,
                            )

                # Recovery watchdog: VOD deadlock recovery when auto-pause is active
                # and there is genuine work to process (not natural silence).
                # Unpausing MPV cannot unblock an inference call stuck inside
                # its model/backend. Reserve the pacing escape for idle workers
                # with queued input; never play untranslated video to mask a hang.
                needs_recovery = (
                    not ingest.source.live and is_queued_stall
                    and not is_in_flight_stall and not is_natural_silence
                )
                if needs_recovery:
                    auto_paused = False
                    for out in outputs:
                        if getattr(out, "_vod_auto_paused", False):
                            auto_paused = True
                            break
                    if auto_paused:
                        record_event(
                            "vod_deadlock_recovery",
                            queues=subtitle_backlog,
                            elapsed_s=no_sub_s,
                            generation=ingest.generation,
                            stall_type="queued" if is_queued_stall else "in_flight",
                        )
                        log.warning(
                            "VOD deadlock detected: no subtitle for %.0fs with "
                            "%s stall and auto-pause active; attempting recovery",
                            no_sub_s,
                            "queued" if is_queued_stall else "in-flight",
                        )
                        for out in outputs:
                            if getattr(out, "_vod_auto_paused", False):
                                resume = getattr(out, "resume_from_auto_pause", None)
                                if callable(resume):
                                    resume()
                                else:
                                    try:
                                        out._player_command("set_property", "pause", False)
                                        out._vod_auto_paused = False
                                        log.info("Recovery: forced MPV resume")
                                    except Exception:
                                        log.exception("Recovery: failed to resume MPV")
                        last_warning = now

            # Watermark diagnostics: log translation progress every 10s during VOD
            if not ingest.source.live and now - last_vod_diag > 10.0:
                last_vod_diag = now
                # Collect watermark positions from outputs via CoverageTracker API
                tu = None
                vp = None
                ps = None
                for out in outputs:
                    coverage = getattr(out, "_coverage", None)
                    if coverage is not None:
                        tu = coverage.translated_until
                        vp = coverage.vad_progress
                        ps = coverage.pending_count()
                depths = {name: q.qsize() for name, q in queues.items()}
                log.info(
                    "VOD diagnostics: translated_until=%.1fs vad_progress=%.1fs "
                    "pending_speech=%d queues=%s",
                    tu if tu is not None else -1,
                    vp if vp is not None else -1,
                    ps if ps is not None else -1,
                    depths,
                )
                # Pending speech is finalized by its own terminal processing path
                # (published/suppressed/expired/cancelled) or cleared on seek.
                # Never abandon valid speech based only on VAD media position.

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
                raise RuntimeError("MPV scheduler crashed") from failure  # noqa: TRY004
    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")

    finally:
        # Stop producers and MPV, then interrupt the owned ASR server so a
        # pending request cannot keep the application alive after closing.
        stop.set()
        for name, interrupt in (("ASR", getattr(asr, "close", None)),
                                ("Translation", getattr(translator, "interrupt", None))):
            if callable(interrupt):
                try:
                    interrupt()
                except Exception:
                    log.exception("%s interruption failed during shutdown", name)
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
                    + cfg.translate.timeout_s * 2))
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
                for label, worker in (("ASR", asr), ("Translation", translator)):
                    close = getattr(worker, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            log.exception("%s shutdown failed", label)
        log.info("done: %s", stats.summary())
