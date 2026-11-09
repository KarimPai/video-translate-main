"""Direct streaming speech translation: audio -> Whisper Large-v3 -> English subtitles.

Retains GTCRN, FireRedVAD, Sherpa speaker boundaries, MPV timing and VOD
seek caching. No intermediate Japanese ASR, YouTube transcripts, or MT model.
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
from .context import RecentSpeechContext, strip_context_echo
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
            context.reset()
            last_generation = item.generation
        if isinstance(item, AudioProgress):
            if item.reset:
                recent_outros.clear()
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
            context.accept(utt, cached.en)
            out_q.put(replace(
                cached, generation=utt.generation, timings=utt.timings,
                speaker_handoff=utt.speaker_handoff or cached.speaker_handoff,
            ))
            continue
        if utt.probe and _discard_probe(utt.audio):
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

            retry_reason = (
                "outro" if generic_outro and getattr(cfg, "retry_on_outro", True)
                else "uncertain" if original.suspicious else ""
            )
            if retry_reason and getattr(cfg, "retry_uncertain", True):
                max_samples = round(max(1.0, float(
                    getattr(cfg, "retry_max_audio_s", 12.0)
                )) * SAMPLE_RATE)
                tail = context.audio_context(utt)
                if tail is not None and utt.audio.size + tail.size > max_samples:
                    tail = None
                samples = (
                    np.concatenate((tail, utt.audio))
                    if tail is not None else utt.audio
                )
                retried = True
                second = whisper.decode_audio(
                    samples, prompt=style_prompt or None,
                    hotwords=context.hotwords(),
                    temperature=float(getattr(cfg, "retry_temperature", 0.2)),
                )
                # Translation is not word-aligned. Do not publish previous
                # dialogue again just because we included its audio.
                candidate = (
                    strip_context_echo(context.last_text, second.text)
                    if tail is not None else second.text
                )
                if candidate and not second.suspicious:
                    first_logp = (
                        original.avg_logprob if original.avg_logprob is not None
                        else -1.0
                    )
                    second_logp = (
                        second.avg_logprob if second.avg_logprob is not None
                        else first_logp
                    )
                    if (
                        not english or
                        (original.suspicious and
                         second_logp >= first_logp - 0.15) or
                        (generic_outro and
                         second_logp >= first_logp - 0.15 and
                         not _OUTRO.fullmatch(
                             " ".join(re.sub(r"[^\w\s]", "", candidate).split())
                         ))
                    ):
                        english = candidate
                        selected = second
                        log.info(
                            "Whisper retry accepted (%s, context_audio=%.2fs, "
                            "gen=%d %.2f-%.2fs)",
                            retry_reason,
                            0.0 if tail is None else tail.size / SAMPLE_RATE,
                            utt.generation, utt.start_s, utt.end_s,
                        )
                    else:
                        log.info(
                            "Whisper retry discarded: failed quality comparison "
                            "(gen=%d %.2f-%.2fs)",
                            utt.generation, utt.start_s, utt.end_s,
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
        utt.timings.mt_start = utt.timings.asr_end
        utt.timings.mt_end = utt.timings.asr_end
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
            return "no English subtitles yet"
        q = " ".join(f"{k}={v}" for k, v in self.max_q.items())
        return (
            f"{self.lines} lines | Whisper avg {self.sum_s / self.lines:.2f}s "
            f"max {self.max_s:.2f}s | audio-to-English "
            f"avg {self.total_sum_s / self.lines:.2f}s "
            f"max {self.total_max_s:.2f}s | max queue {q}"
        )


def run(
    ingest: Ingest,
    cfg: Config,
    whisper: WhisperLargeV3,
    outputs: list[Output],
    stop: threading.Event,
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Run four streaming workers with bounded queues and reliable teardown."""
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    speaker_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)
    failures: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    queues = {"audio": audio_q, "vad": utt_q, "speaker": speaker_q, "subtitle": sub_q}
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
            args=("whisper", run_whisper, failures, whisper, speaker_q, sub_q,
                  generation, session_cache,
                  {"gtcrn": cfg.gtcrn.enabled, "speaker": cfg.speaker.enabled},
                  {"live": ingest.source.live, "duration_s": ingest.source.duration}),
            name="whisper", daemon=True,
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
                    log.info("first Whisper English subtitle for generation=%d at %.2fs",
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
            if backlog and now - last_event >= 60:
                raise RuntimeError(
                    "Whisper pipeline stalled with queued work: " + str(backlog)
                    + "; inspect logs/run.log and logs/run.log"
                )
            if backlog and now - last_event >= 20 and now >= last_warn:
                last_warn = now + 20
                log.warning("Whisper pipeline backlog: %s", backlog)
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
