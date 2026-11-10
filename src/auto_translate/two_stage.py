"""Japanese-only Whisper ASR -> Japanese-to-English PLaMo translation.

This is the active pipeline for the whisper-v3 branch. The original direct
Whisper translation function is not called by run() in this configuration.
All timestamps belong to the source audio and remain unchanged at both stages.
"""
from __future__ import annotations

import logging
import queue
import re
import time
from collections import deque
from dataclasses import replace
from typing import Callable

import numpy as np

from .asr_quality import is_standalone_laughter, formulaic_japanese_outro_key
from .translation_integrity import nearby_japanese_fragments
from .types import SAMPLE_RATE, AudioProgress, Subtitle, Transcript, Utterance
from .whisper_v3 import WhisperLargeV3
from .session_cache import SessionSubtitleCache
from .translate.base import Translator

log = logging.getLogger(__name__)

# Whisper's language='ja' is a decoder hint, not an output-language validator.
# Reject English-only hallucinations and spoken English in this Japanese-only
# experimental pipeline. Any Japanese kana or Kanji allows normal code-switching.
_JAPANESE_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def has_japanese_script(text: str) -> bool:
    return bool(_JAPANESE_SCRIPT.search(text))


def run_whisper_asr(
    whisper: WhisperLargeV3,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Recognize Japanese ONLY; route cache hits without another model call."""
    last_generation: int | None = None
    recent_outro: tuple[str, float] | None = None
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            return
        if get_generation is not None and item.generation != get_generation():
            continue
        if isinstance(item, AudioProgress):
            if item.reset:
                last_generation = item.generation
                recent_outro = None
            out_q.put(item)
            continue

        utt = item
        if last_generation is None or utt.generation != last_generation:
            last_generation = utt.generation
            recent_outro = None

        if session_cache is not None:
            cached = session_cache.match(utt.start_s, utt.end_s)
            if cached is not None and cached.ja and cached.en:
                out_q.put(replace(
                    cached, timings=utt.timings,
                    generation=utt.generation,
                    speaker_handoff=utt.speaker_handoff or cached.speaker_handoff,
                ))
                continue

        if utt.probe:  # VAD-negative probes have a high hallucination risk.
            continue
        pcm = np.asarray(utt.audio, dtype=np.float32)
        if not pcm.size or not np.isfinite(pcm).all():
            continue
        if float(np.max(np.abs(pcm))) < 1.0 / 32768:
            continue

        utt.timings.asr_start = time.monotonic()
        # A transcription prompt containing English subtitle history risks
        # contaminating the Japanese ASR, particularly across speakers.
        # Avoid long-window audio overlaps and English-name hints by default.
        original = whisper.decode_audio(utt.audio, prompt=None, hotwords=None)
        chosen = original
        retry = False
        cfg = whisper.cfg
        if (
            original.suspicious
            and getattr(cfg, "retry_uncertain", True)
            and not utt.probe
        ):
            retry = True
            second = whisper.decode_audio(
                utt.audio, prompt=None, hotwords=None,
                temperature=float(getattr(cfg, "retry_temperature", 0.2)),
            )
            original_logp = (
                original.avg_logprob if original.avg_logprob is not None
                else -100.0
            )
            second_logp = (
                second.avg_logprob if second.avg_logprob is not None
                else -100.0
            )
            if (
                second.text and not second.suspicious
                and second_logp >= original_logp - 0.15
            ):
                chosen = second
        utt.timings.asr_end = time.monotonic()
        japanese = chosen.text.strip()
        if not japanese:
            continue
        if not has_japanese_script(japanese):
            log.info(
                "Skipped Whisper Latin-only ASR result (generation=%d "
                "%.2f-%.2fs): %r",
                utt.generation, utt.start_s, utt.end_s, japanese[:120],
            )
            continue
        if getattr(cfg, "diagnostics", False):
            log.info(
                "Whisper Japanese ASR: gen=%d %.2f-%.2fs %.3fs retry=%s "
                "avg_logprob=%s no_speech=%s discarded=%s JP=%r",
                utt.generation, utt.start_s, utt.end_s,
                utt.timings.asr_end - utt.timings.asr_start,
                retry, chosen.avg_logprob, chosen.no_speech_prob,
                chosen.discarded_segments, japanese[:120],
            )
        if getattr(cfg, "suppress_standalone_laughter", False) and is_standalone_laughter(japanese):
            log.info(
                "Skipped standalone laughter (gen=%d %.2f-%.2f): %r",
                utt.generation, utt.start_s, utt.end_s, japanese[:100],
            )
            continue
        outro_key = formulaic_japanese_outro_key(japanese)
        if (getattr(cfg, "suppress_repeated_japanese_outros", False)
                and outro_key is not None):
            recent = recent_outro
            cooldown = max(0.0, float(getattr(
                cfg, "japanese_outro_cooldown_s", 120.0,
            )))
            if (recent is not None and recent[0] == outro_key
                    and 0 <= utt.start_s - recent[1] <= cooldown):
                log.info(
                    "Skipped repeated stock outro (gen=%d %.2f-%.2f): %r",
                    utt.generation, utt.start_s, utt.end_s, japanese[:100],
                )
                continue
            recent_outro = (outro_key, utt.start_s)
        if get_generation is not None and utt.generation != get_generation():
            continue
        out_q.put(Transcript(
            utt.start_s, utt.end_s, japanese, utt.timings,
            generation=utt.generation,
            speech_start_s=utt.speech_start_s,
            speaker_handoff=utt.speaker_handoff,
        ))


def run_plamo_translate(
    translator: Translator,
    in_q: "queue.Queue[Transcript | Subtitle | AudioProgress | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
) -> None:
    """Translate only accepted current Japanese, with bounded prior context."""
    cfg = getattr(translator, "cfg", None)
    context_lines = max(0, int(getattr(cfg, "context_lines", 8)))
    memory_lines = max(0, int(getattr(cfg, "memory_lines", 24)))
    history: deque[tuple[str, str]] = deque(maxlen=max(1, context_lines, memory_lines))
    recent_ja: deque[tuple[float, float, str]] = deque(maxlen=4)
    last_generation: int | None = None

    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            return

        if get_generation is not None and item.generation != get_generation():
            continue

        if isinstance(item, AudioProgress):
            if item.reset or (
                last_generation is not None and item.generation != last_generation
            ):
                history.clear()
                recent_ja.clear()
            last_generation = item.generation
            out_q.put(item)
            continue

        tr = item
        if last_generation is None or last_generation != tr.generation:
            history.clear()
            recent_ja.clear()
        last_generation = tr.generation
        if tr.speaker_handoff:
            recent_ja.clear()

        if isinstance(tr, Subtitle):
            # Subtitles reused from VOD cache already contain JA and EN.
            if tr.en and tr.en != "[translation failed]":
                history.append((tr.ja, tr.en))
            recent_ja.append((tr.start_s, tr.end_s, tr.ja))
            tr.timings.mt_start = tr.timings.asr_end
            tr.timings.mt_end = tr.timings.asr_end
            out_q.put(tr)
            continue

        tr.timings.mt_start = time.monotonic()
        # Translation context must not silently rewrite raw Japanese detected
        # by Whisper. The homophone hints now live in PLaMo context only.
        japanese = tr.ja
        fragments = nearby_japanese_fragments(
            list(recent_ja), tr.start_s, japanese,
            max_fragments=int(getattr(cfg, "fragment_context_lines", 2)),
            max_gap_s=float(getattr(cfg, "fragment_max_gap_s", 1.8)),
        )
        try:
            english = translator.translate(
                japanese, list(history), fragment_context=fragments
            )
        except Exception:
            log.exception(
                "PLaMo translation failed: generation=%s %.2f-%.2fs JP=%r",
                tr.generation, tr.start_s, tr.end_s, japanese,
            )
            english = "[translation failed]"
        tr.timings.mt_end = time.monotonic()

        if get_generation is not None and tr.generation != get_generation():
            continue
        if english and english != "[translation failed]":
            history.append((japanese, english))
        recent_ja.append((tr.start_s, tr.end_s, japanese))
        out_q.put(Subtitle(
            tr.start_s, tr.end_s, japanese, english, tr.timings,
            generation=tr.generation,
            speech_start_s=tr.speech_start_s,
            speaker_handoff=tr.speaker_handoff,
        ))
