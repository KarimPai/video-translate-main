"""Single-stage Whisper Large-v3 direct Japanese-audio-to-English AST.

Retains FireRedVAD, optional audio processing, VOD cache, seek-safe
playback synchronization, audio pacing and manual-pause watchdog fixes.
"""

from __future__ import annotations

import logging
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .config import Config
from .context import RecentSpeechContext
from .subtitle_style import concise_english
from .whisper_v3 import WhisperLargeV3
from .ingest import Ingest
from .output.base import Output
from .session_cache import SessionSubtitleCache
from .speaker import run_speaker
from .types import SAMPLE_RATE, AudioChunk, AudioProgress, Subtitle, Utterance
from .vad import run_segmenter

log = logging.getLogger(__name__)

# Whisper can hallucinate stock YouTube outro captions on ambiguous speech.
# Only suppress repeats of these complete phrases; never discard their first
# occurrence, or suppress ordinary, genuine conversational thanks.
_OUTRO = re.compile(
    r"^(?:thank you|thanks)(?: very much)? for watching"
    r"(?: (?:until )?the end)?$",
    re.IGNORECASE,
)
# Some weak Japanese gaming utterances decode into stock isolated thanks.
# Never filter normal contextual expressions (e.g., \"Thanks for the help\").
_STOCK_THANKS = re.compile(
    r"^(?:thank you(?: very much)?|thanks(?: a lot)?)$",
    re.IGNORECASE,
)


def _is_stock_thanks(text: str) -> bool:
    normalized = " ".join(re.sub(r"[^a-zA-Z\s]", "", text).split())
    return bool(_STOCK_THANKS.fullmatch(normalized))


def _credible_repeated_thanks(first: object, second: object,
                               vad_confidence: float | None) -> bool:
    """Require corroboration and strong speech evidence on repeated stock thanks."""
    if not _is_stock_thanks(getattr(first, "text", "")) or not _is_stock_thanks(
        getattr(second, "text", "")
    ):
        return False
    if vad_confidence is None or not math.isfinite(float(vad_confidence)):
        return False
    if float(vad_confidence) < 0.65:
        return False
    for decoded in (first, second):
        lp = getattr(decoded, "avg_logprob", None)
        ns = getattr(decoded, "no_speech_prob", None)
        if (lp is None or ns is None or not math.isfinite(float(lp))
                or not math.isfinite(float(ns)) or float(lp) < -0.55
                or float(ns) > 0.25 or getattr(decoded, "suspicious", False)):
            return False
    return True



def _is_outro_phrase(text: str) -> bool:
    """Detect *standalone* stock closing phrases, not conversational thanks."""
    normalized = " ".join(re.sub(r"[^\w\s]", "", text).split())
    return bool(_OUTRO.fullmatch(normalized))


def _suppress_generic_outro(
    text: str,
    end_s: float,
    *,
    cfg: object | None,
    source_info: dict | None,
) -> bool:
    """Guard against stock-outro hallucinations in normal stream content.

    Only allow standalone closing phrases within the last 90s of a completed
    VOD when a trusted duration is available. Unknown-duration/live streams
    cannot prove an ending; the safeguard can be disabled via config.toml.
    """
    if not getattr(cfg, "suppress_generic_outros", True):
        return False
    if not _is_outro_phrase(text):
        return False
    info = source_info or {}
    if not info.get("live", False):
        duration = info.get("duration_s")
        if duration is not None:
            try:
                duration = float(duration)
                allow_s = max(0.0, min(
                    300.0, float(getattr(cfg, "outro_allow_near_vod_end_s", 90.0))
                ))
                if math.isfinite(duration) and duration > 0:
                    if duration - allow_s <= end_s <= duration + 1.0:
                        return False
            except (TypeError, ValueError):
                pass
    return True


def _is_repeated_outro(
    text: str, timestamp_s: float, recent: list[float], cooldown_s: float
) -> bool:
    normalized = re.sub(r"[^\w\s]", "", text).strip()
    normalized = " ".join(normalized.split())
    if not _OUTRO.fullmatch(normalized):
        return False
    # Timestamps are from the video; clear on seeks, and accept after expiry.
    recent[:] = [t for t in recent if 0 <= timestamp_s - t <= cooldown_s]
    if recent:
        return True
    recent.append(timestamp_s)
    return False


def _guard_stage(
    name: str,
    target: Callable,
    failures: "queue.Queue[tuple[str, BaseException]]",
    *args,
) -> None:
    """Propagate worker exceptions instead of silently hanging playback."""
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


def _discard_probe(audio: np.ndarray) -> bool:
    """Reject clearly non-speech VAD-negative probes, without a text ASR."""
    if audio.size < SAMPLE_RATE // 4:
        return True
    peak = float(np.max(np.abs(audio)))
    if peak < (1 / 32768):
        return True
    # Avoid sending isolated gunshot/click impulses to the speech translator.
    frames = audio[:audio.size // 320 * 320].reshape(-1, 320)
    if frames.size and np.max((np.abs(frames) > peak * 0.05).sum(axis=1)) <= 16:
        return True
    return False



def _stable_weak_speech(first: object, second: object, vad_confidence: float | None,
                        cfg: object | None) -> str:
    """Conservatively recover quiet real speech rejected by both passes.

    Exact word agreement is required, not a fuzzy semantic match. This does
    not promote a single hallucination, weak VAD evidence, or likely silence.
    """
    if not getattr(cfg, "rescue_stable_weak_speech", True):
        return ""
    if vad_confidence is None or not math.isfinite(float(vad_confidence)):
        return ""
    if float(vad_confidence) < 0.78:
        return ""
    if getattr(first, "text", "") or getattr(second, "text", ""):
        return ""

    tokens: list[list[str]] = []
    for decoded in (first, second):
        candidate = getattr(decoded, "weak_text", "").strip()
        avg = getattr(decoded, "weak_avg_logprob", None)
        ns = getattr(decoded, "weak_no_speech_prob", None)
        if (not candidate or avg is None or ns is None
                or not math.isfinite(float(avg)) or not math.isfinite(float(ns))
                or float(avg) < -1.60 or float(ns) > 0.35
                or _is_outro_phrase(candidate) or _is_stock_thanks(candidate)):
            return ""
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", candidate.casefold())
        if len(words) < 2 or sum(map(len, words)) < 8:
            return ""
        tokens.append(words)
    return first.weak_text if tokens[0] == tokens[1] else ""


def run_whisper(
    whisper: WhisperLargeV3,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    session_cache: SessionSubtitleCache | None = None,
    audio_processing: dict[str, bool] | None = None,
    source_info: dict | None = None,
) -> None:
    """Emit each accepted utterance once; never shift its source timestamps."""
    empty_count = 0
    recent_outros: list[float] = []
    recent_thanks: list[float] = []
    last_generation: int | None = None
    cfg = getattr(whisper, "cfg", None)
    context = RecentSpeechContext(cfg)
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            return
        if get_generation is not None and item.generation != get_generation():
            continue
        if last_generation is None or item.generation != last_generation:
            recent_outros.clear()
            recent_thanks.clear()
            context.reset()
            last_generation = item.generation
        if isinstance(item, AudioProgress):
            if item.reset:
                recent_outros.clear()
                recent_thanks.clear()
                context.reset()
            out_q.put(item)
            continue

        utt = item
        context.prepare(utt)
        cached = session_cache.match(utt.start_s, utt.end_s) if session_cache is not None else None
        if cached is not None:
            if _suppress_generic_outro(
                cached.en, utt.end_s, cfg=cfg, source_info=source_info,
            ):
                log.info(
                    "Suppressed cached generic Whisper outro (gen=%d %.2f-%.2fs): %r",
                    utt.generation, utt.start_s, utt.end_s, cached.en,
                )
                continue
            if _is_stock_thanks(cached.en):
                recent_thanks[:] = [t for t in recent_thanks
                                    if 0 <= utt.start_s - t <= 120.0]
                recent_thanks.append(utt.start_s)
            context.accept(utt, cached.en)
            out_q.put(replace(
                cached, generation=utt.generation, timings=utt.timings,
                speaker_handoff=utt.speaker_handoff or cached.speaker_handoff,
            ))
            continue
        # VAD-negative noise probes are never safe for direct AST.
        if utt.probe:
            continue
        if not len(utt.audio) or not np.isfinite(utt.audio).all():
            continue
        if float(np.max(np.abs(utt.audio))) < 1 / 32768:
            continue

        utt.timings.asr_start = time.monotonic()
        style_prompt = ""
        if getattr(cfg, "concise_english", False):
            style_prompt = str(getattr(cfg, "style_prompt", ""))[:120].strip()
        previous_prompt = context.prompt(utt)
        combined_prompt = " ".join(
            item for item in (style_prompt, previous_prompt) if item
        ) or None
        retried = False
        retry_reason = ""
        selected = None
        if hasattr(whisper, "decode_audio"):
            original = whisper.decode_audio(
                utt.audio, prompt=combined_prompt,
                hotwords=context.hotwords(),
            )
            selected = original
            english = original.text
            generic_outro = _is_outro_phrase(english)
            # Only isolated generic thanks are checked, not meaningful
            # sentences. Respect speaker handoffs and reset history on seek.
            recent_thanks[:] = [t for t in recent_thanks
                                if 0 <= utt.start_s - t <= 120.0]
            recurring_thanks = (
                _is_stock_thanks(english) and bool(recent_thanks)
                and not utt.speaker_handoff
                and getattr(cfg, "confirm_repeated_stock_thanks", True)
            )
            retry_reason = (
                "recurring_thanks" if recurring_thanks
                else "outro" if generic_outro and getattr(cfg, "retry_on_outro", True)
                else "uncertain" if original.suspicious else ""
            )
            silent_reject = (
                not english and original.discarded_segments > 0
                and original.no_speech_prob is not None
                and original.no_speech_prob >= float(getattr(
                    cfg, "no_speech_retry_block_threshold", 0.85,
                ))
            )
            if (retry_reason and getattr(cfg, "retry_uncertain", True)
                    and not silent_reject):
                retried = True
                # Same utterance only: never concatenate a prior speech tail
                # or feed old subtitles back into this decoder pass.
                second = whisper.decode_audio(
                    utt.audio, prompt=style_prompt or None,
                    hotwords=context.hotwords(),
                    temperature=float(getattr(cfg, "retry_temperature", 0.2)),
                )
                candidate = second.text.strip()
                first_logp = original.avg_logprob
                second_logp = second.avg_logprob
                if (candidate and not second.suspicious
                        and (first_logp is None or second_logp is None
                             or second_logp >= first_logp - 0.15)
                        and (not generic_outro or not _is_outro_phrase(candidate))
                        and (not recurring_thanks or not _is_stock_thanks(candidate)
                             or _credible_repeated_thanks(
                                 original, second, utt.vad_confidence))):
                    english = candidate
                    selected = second
                    log.info(
                        "Whisper AST retry accepted (%s, gen=%d %.2f-%.2fs)",
                        retry_reason, utt.generation, utt.start_s, utt.end_s,
                    )
                elif recurring_thanks:
                    # A repeated isolated thank-you was not independently
                    # supported by strong speech/model evidence. The first
                    # occurrence is untouched; do not spam later weak ones.
                    log.info(
                        "Suppressed unconfirmed recurring generic thanks "
                        "(gen=%d %.2f-%.2fs vad=%s first_lp=%s second_lp=%s)",
                        utt.generation, utt.start_s, utt.end_s,
                        utt.vad_confidence, original.avg_logprob,
                        second.avg_logprob,
                    )
                    english = ""
                elif not english and not candidate:
                    rescued = _stable_weak_speech(
                        original, second, utt.vad_confidence, cfg,
                    )
                    if rescued:
                        english = rescued
                        selected = second
                        log.info(
                            "Whisper AST recovered matching weak speech "
                            "(gen=%d %.2f-%.2fs vad=%.2f)",
                            utt.generation, utt.start_s, utt.end_s,
                            utt.vad_confidence,
                        )
        else:
            # Existing test doubles and older API-compatible adapters.
            english = whisper.translate_audio(utt.audio)

        utt.timings.asr_end = time.monotonic()
        raw_english = english
        english = concise_english(
            english,
            remove_fillers=bool(getattr(cfg, "remove_fillers", True)),
            collapse_repetitions=bool(
                getattr(cfg, "collapse_repetitions", True)
            ),
        ) if getattr(cfg, "concise_english", True) else english
        if raw_english != english:
            log.debug(
                "Concise subtitle cleanup: before=%r after=%r",
                raw_english[:160], english[:160],
            )
        if get_generation is not None and utt.generation != get_generation():
            continue

        if getattr(cfg, "diagnostics", False):
            rms = float(np.sqrt(np.mean(np.square(
                np.asarray(utt.audio, dtype=np.float64)
            )))) if utt.audio.size else 0.0
            peak = float(np.max(np.abs(utt.audio))) if utt.audio.size else 0.0
            log.info(
                "Whisper diagnostic: gen=%d speech=%.2f-%.2fs %.2fs "
                "rms=%.4f peak=%.3f vad_confidence=%s gtcrn=%s "
                "sherpa=%s speaker_split=%s speaker_handoff=%s "
                "model=%.3fs retry=%s reason=%s "
                "avg_logprob=%s no_speech=%s compression=%s "
                "discarded=%s output=%r",
                utt.generation, utt.start_s, utt.end_s,
                utt.end_s - utt.start_s, rms, peak,
                utt.vad_confidence,
                (audio_processing or {}).get("gtcrn", "unknown"),
                (audio_processing or {}).get("speaker", "unknown"),
                utt.speaker_split, utt.speaker_handoff,
                utt.timings.asr_end - utt.timings.asr_start,
                retried, retry_reason,
                getattr(selected, "avg_logprob", None),
                getattr(selected, "no_speech_prob", None),
                getattr(selected, "compression_ratio", None),
                getattr(selected, "discarded_segments", None),
                english[:100],
            )

        if (re.search(r"[\u3040-\u30ff\u3400-\u9fff]", english)
                and not re.search(r"[A-Za-z]", english)):
            log.info("Skipped Japanese-only Whisper AST output")
            continue
        if not english.strip():
            empty_count += 1
            if empty_count % 8 == 0:
                log.warning(
                    "Whisper returned empty English for %d utterances "
                    "(latest %.2f-%.2fs)",
                    empty_count, utt.start_s, utt.end_s,
                )
            continue
        empty_count = 0
        if _suppress_generic_outro(
            english, utt.end_s, cfg=cfg, source_info=source_info,
        ):
            log.info(
                "Suppressed generic Whisper outro during gameplay "
                "(gen=%d %.2f-%.2fs live=%s duration=%s): %r",
                utt.generation, utt.start_s, utt.end_s,
                (source_info or {}).get("live", "unknown"),
                (source_info or {}).get("duration_s", "unknown"), english,
            )
            continue
        if (
            getattr(cfg, "suppress_repeated_signoffs", True) and
            _is_repeated_outro(
                english, utt.start_s, recent_outros,
                getattr(cfg, "signoff_cooldown_s", 1200.0),
            )
        ):
            log.info(
                "Suppressed repeated generic Whisper outro (%.2fs): %r",
                utt.start_s, english,
            )
            continue

        if _is_stock_thanks(english):
            recent_thanks[:] = [t for t in recent_thanks
                                if 0 <= utt.start_s - t <= 120.0]
            recent_thanks.append(utt.start_s)
        context.accept(utt, english)
        out_q.put(Subtitle(
            start_s=utt.start_s,
            end_s=utt.end_s,
            ja="",
            en=english,
            timings=utt.timings,
            generation=utt.generation,
            speech_start_s=utt.speech_start_s,
            speaker_handoff=utt.speaker_handoff,
        ))


@dataclass
class Stats:
    lines: int = 0
    sum_s: float = 0.0
    max_s: float = 0.0
    total_sum_s: float = 0.0
    total_max_s: float = 0.0
    max_q: dict[str, int] = field(default_factory=dict)

    def add(self, sub: Subtitle) -> None:
        t = sub.timings
        decode_s = max(0.0, t.asr_end - t.asr_start)
        total_s = max(0.0, t.asr_end - t.audio_read)
        self.lines += 1
        self.sum_s += decode_s
        self.max_s = max(self.max_s, decode_s)
        self.total_sum_s += total_s
        self.total_max_s = max(self.total_max_s, total_s)

    def summary(self) -> str:
        if not self.lines:
            return "no Japanese-to-English subtitles yet"
        q = " ".join(f"{k}={v}" for k, v in self.max_q.items())
        return (
            f"{self.lines} lines | Whisper AST avg {self.sum_s / self.lines:.2f}s "
            f"max {self.max_s:.2f}s | audio-to-English "
            f"avg {self.total_sum_s / self.lines:.2f}s "
            f"max {self.total_max_s:.2f}s | max queue {q}"
        )


def _expected_vod_idle(ingest: Ingest, outputs: list[Output]) -> bool:
    """True only if a VOD is deliberately idle, NOT a failed worker.

    The read-ahead cap parks FFmpeg when audio gets 75s ahead of MPV.
    A manual MPV pause may then leave a partial utterance or one queued item
    with no subsequent audio progress. This is normal; the watchdog must not
    kill the session after 60 seconds. Worker exceptions are always checked
    independently via _raise_stage_failure.
    """
    if getattr(ingest.source, "live", False):
        return False

    lead_limited = getattr(ingest, "is_vod_audio_lead_limited", None)
    if callable(lead_limited) and lead_limited():
        return True

    for output in outputs:
        is_manually_paused = getattr(output, "is_manually_paused", None)
        if callable(is_manually_paused) and is_manually_paused():
            return True

    return False


def run(
    ingest: Ingest,
    cfg: Config,
    whisper: WhisperLargeV3,
    outputs: list[Output],
    stop: threading.Event,
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Run direct Whisper AST after FireRedVAD."""
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    speaker_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)
    failures: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    queues = {
        "audio": audio_q, "vad": utt_q, "speaker": speaker_q,
        "subtitle": sub_q,
    }
    generation = lambda: ingest.generation

    threads = [
        threading.Thread(
            target=_guard_stage, args=("ingest", ingest.run, failures, audio_q),
            name="ingest", daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("vad", run_segmenter, failures, cfg.vad, audio_q, utt_q, generation),
            name="vad", daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("speaker", run_speaker, failures, cfg.speaker, utt_q, speaker_q, generation),
            name="speaker", daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("whisper-ast", run_whisper, failures,
                  whisper, speaker_q, sub_q, generation, session_cache,
                  {"gtcrn": cfg.gtcrn.enabled, "speaker": cfg.speaker.enabled},
                  {"live": ingest.source.live, "duration_s": ingest.source.duration}),
            name="whisper-ast", daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    stats = Stats()
    first_logged: set[int] = set()
    processed_generation = ingest.generation
    processed_from = ingest.source.start_s
    start = time.monotonic()
    next_stats = start + cfg.stats_interval_s
    last_event = start
    last_warn = start
    was_expected_idle = False
    try:
        while not stop.is_set():
            _raise_stage_failure(failures)
            try:
                item = sub_q.get(timeout=0.5)
                last_event = time.monotonic()
                if item is None:
                    for thread in threads:
                        thread.join(timeout=0.3)
                    _raise_stage_failure(failures)
                    for output in outputs:
                        finish = getattr(output, "finish", None)
                        if finish is not None:
                            finish()
                    for output in outputs:
                        output.drain()
                    break
                if item.generation != ingest.generation:
                    continue
                if isinstance(item, AudioProgress):
                    position = item.sample / SAMPLE_RATE
                    if item.reset:
                        processed_generation = item.generation
                        processed_from = position
                    elif session_cache is not None:
                        if processed_generation != item.generation:
                            processed_generation = item.generation
                            processed_from = position
                        session_cache.record_processed(processed_from, position)
                        processed_from = position
                    for output in outputs:
                        method = getattr(output, "reset_buffer" if item.reset else "advance", None)
                        if method is not None:
                            method(position)
                    continue
                if item.generation not in first_logged:
                    first_logged.add(item.generation)
                    log.info("first Whisper AST English subtitle for generation=%d at %.2fs",
                             item.generation, item.start_s)
                stats.add(item)
                if session_cache is not None:
                    session_cache.add(item)
                for output in outputs:
                    output.show(item)
            except queue.Empty:
                pass

            _raise_stage_failure(failures)
            now = time.monotonic()
            for name, q in queues.items():
                stats.max_q[name] = max(stats.max_q.get(name, 0), q.qsize())
            backlog = {k: q.qsize() for k, q in queues.items() if q.qsize()}
            expected_idle = _expected_vod_idle(ingest, outputs)
            if expected_idle:
                # A user pause or intentional 75s audio lead cap must not
                # count as a stall. Reset the clock throughout the pause:
                # otherwise the old timestamp would cause an immediate crash
                # on the first scheduler tick after playback resumes.
                if not was_expected_idle:
                    log.info(
                        "Whisper AST watchdog: expected VOD inactivity "
                        "(manual pause or read-ahead limit); retaining queues %s",
                        backlog,
                    )
                last_event = now
                last_warn = now
            else:
                if was_expected_idle:
                    log.info(
                        "Whisper AST watchdog active again after VOD resumes"
                    )
                if backlog and now - last_event >= 60:
                    raise RuntimeError(
                        "Whisper AST pipeline stalled with queued work: "
                        + str(backlog) + "; inspect logs/run.log"
                    )
                if backlog and now - last_event >= 20 and now >= last_warn:
                    last_warn = now + 20
                    log.warning("Whisper AST pipeline backlog: %s", backlog)
            was_expected_idle = expected_idle
            if now >= next_stats:
                next_stats = now + cfg.stats_interval_s
                log.info("[%.0fs] %s", now - start, stats.summary())
                for name, q in queues.items():
                    if q.qsize() >= max(1, int(q.maxsize * 0.75)):
                        log.warning("Whisper backpressure: %s queue %d/%d",
                                    name, q.qsize(), q.maxsize)
    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")
    finally:
        for output in outputs:
            output.close()
        ingest.stop()
        log.info("done: %s", stats.summary())
