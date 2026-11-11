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

from .config import Config, vad_for_source
from .context import RecentSpeechContext
from .subtitle_style import concise_english
from .neutral_subtitles import needs_reference_evidence, neutralize_with_japanese
from .whisper_v3 import WhisperLargeV3
from .ingest import Ingest
from .output.base import Output
from .session_cache import SessionSubtitleCache
from .types import SAMPLE_RATE, AudioChunk, AudioProgress, Subtitle, Utterance
from .vad import run_segmenter

log = logging.getLogger(__name__)

def _words(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)*", text.casefold().replace("’", "'"))


# Conservative full-cue patterns: do not match ordinary dialogue containing
# these words. English is the normal output of the direct JA->EN AST pipeline.
_PHANTOM_PHRASE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "outro": (
        re.compile(r"(?:thank you|thanks)(?: very much)? for watching"),
    ),
    "credits": (
        re.compile(r"(?:subtitles?|captions?) (?:by|provided by|created by|made by)(?: [\w.'/-]+){0,6}"),
        re.compile(r"(?:subtitled|captioned) by(?: [\w.'/-]+){0,6}"),
    ),
    "subscribe": (
        re.compile(r"(?:please )?subscribe to (?:my|our|the) channel"),
        re.compile(r"(?:please )?(?:like and )?subscribe"),
        re.compile(r"(?:don'?t|do not) forget to (?:like and )?subscribe"),
        re.compile(r"(?:please )?(?:hit|click) the subscribe button"),
        re.compile(r"(?:please )?like,? share,? and subscribe"),
    ),
}


def _phantom_phrase_kind(text: str) -> str:
    """Return the matched kind for a standalone stock phrase, else empty."""
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = normalized.strip(" .!?…。、，:;\"'“”’").casefold()
    if not normalized:
        return ""
    for kind, patterns in _PHANTOM_PHRASE_PATTERNS.items():
        if any(pattern.fullmatch(normalized) for pattern in patterns):
            return kind
    return ""


def _is_stock_outro(text: str) -> bool:
    """Retain the existing focused-outro matcher for compatibility."""
    return _phantom_phrase_kind(text) == "outro"


def _japanese_supports_outro(text: str) -> bool:
    """Require language-specific evidence for the *watching* claim.

    A generic spoken 'arigatou' is evidence for 'thank you', not for an
    invented 'for watching'. Checking the Japanese transcription remains a
    heuristic: two passes from one model are not independent ground truth.
    """
    normalized = re.sub(r"\s+", "", text).casefold()
    if "thankyouforwatching" in normalized or "thanksforwatching" in normalized:
        return True  # Japanese speakers occasionally say this in English.
    watching = any(term in normalized for term in ("視聴", "見て", "観て", "ご覧", "見に", "観に"))
    gratitude = any(term in normalized for term in ("ありがと", "感謝", "サンキュー"))
    return watching and gratitude


def _japanese_supports_phantom(kind: str, text: str) -> bool:
    """Require matching Japanese content, not another English decode alone."""
    if kind == "outro":
        return _japanese_supports_outro(text)
    normalized = re.sub(r"\s+", "", text).casefold()
    if kind == "credits":
        return (
            any(word in normalized for word in ("字幕", "テロップ", "キャプション"))
            and any(word in normalized for word in (
                "作成", "制作", "提供", "担当", "協力", "による", "by",
            ))
        )
    if kind == "subscribe":
        return (
            "チャンネル登録" in normalized
            or ("subscribe" in normalized and "channel" in normalized)
            or ("登録" in normalized and any(word in normalized for word in (
                "お願いします", "よろしく", "してください",
            )))
        )
    return False


def _phantom_timing_implausible(text: str, duration_s: float) -> bool:
    """Use only egregious rates: English words != Japanese speech timing."""
    if duration_s < 0.28:
        return True
    words = len(_words(text))
    return bool(words >= 3 and duration_s < 1.5 and words / duration_s > 8.0)


def _phantom_weak_audio(vad_confidence: float | None, decoded: object) -> bool:
    """Require combined low VAD and high no-speech evidence for an early drop."""
    ns = getattr(decoded, "no_speech_prob", None)
    if vad_confidence is None or ns is None:
        return False
    if not math.isfinite(float(vad_confidence)) or not math.isfinite(float(ns)):
        return False
    if vad_confidence < 0.60 and ns >= 0.85:
        return True
    logprob = getattr(decoded, "avg_logprob", None)
    return bool(
        vad_confidence < 0.50 and ns >= 0.60
        and logprob is not None and math.isfinite(float(logprob))
        and logprob < -0.80
    )


def _structural_issue(text: str, seconds: float) -> str:
    words = _words(text)
    if len(words) > max(12, 8 * seconds + 4):
        return "length"
    for width in range(2, 7):
        for end in range(4 * width, len(words) + 1):
            phrase = words[end-width:end]
            if len(set(phrase)) > 1 and words[end-4*width:end] == phrase * 4:
                return "loop"
    return ""


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
                or float(avg) < -1.60 or float(ns) > 0.35):
            return ""
        words = _words(candidate)
        if not words:
            return ""
        tokens.append(words)
    return first.weak_text if tokens[0] == tokens[1] else ""


def _recover_background_probe(
    whisper: WhisperLargeV3,
    utt: Utterance,
    cfg: object | None,
) -> tuple[str, str]:
    """Evidence-gated VOD-only recovery of low-VAD, otherwise-skipped audio.

    Japanese detection alone is not enough: require two *unprompted* English
    AST passes to agree exactly, with strong model confidence and ordinary
    cadence. Background speech overlapping foreground speech is NOT solved
    by this path; it inspects only VAD-negative audio gaps.
    """
    duration = len(utt.audio) / SAMPLE_RATE
    if (not getattr(cfg, "background_probe_enabled", True)
            or not hasattr(whisper, "decode_audio")
            or duration < 1.5
            or duration > float(getattr(cfg, "background_probe_max_audio_s", 5.0))
            or utt.vad_confidence is None
            or utt.vad_confidence < float(getattr(
                cfg, "background_probe_min_vad_confidence", 0.20))
            or _discard_probe(utt.audio)
            or float(np.sqrt(np.mean(np.square(
                utt.audio.astype(np.float64))))) < 0.003):
        return "", ""
    verifier = getattr(whisper, "transcribe_japanese_evidence", None)
    if not callable(verifier):
        return "", ""
    try:
        japanese = verifier(utt.audio)
        if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]{3,}", japanese):
            return "", japanese
        candidates = [
            whisper.decode_audio(
                utt.audio, prompt=None, hotwords=None,
                temperature=temperature,
            )
            for temperature in (0.0, float(getattr(cfg, "retry_temperature", 0.2)))
        ]
    except Exception as exc:
        log.warning("Whisper background probe failed: %s", exc)
        return "", ""
    for decoded in candidates:
        text = decoded.text.strip()
        if (not text or decoded.suspicious
                or decoded.avg_logprob is None or decoded.avg_logprob < -0.35
                or decoded.no_speech_prob is None or decoded.no_speech_prob > 0.20
                or (decoded.compression_ratio is not None
                    and decoded.compression_ratio > 2.0)
                or _structural_issue(text, duration)
                or _phantom_phrase_kind(text)):
            return "", japanese
    first, second = candidates
    if (len(_words(first.text)) < 2
            or _words(first.text) != _words(second.text)):
        return "", japanese
    return first.text.strip(), japanese


def run_whisper(
    whisper: WhisperLargeV3,
    in_q: "queue.Queue[Utterance | AudioProgress | None]",
    out_q: "queue.Queue[Subtitle | AudioProgress | None]",
    get_generation: Callable[[], int] | None = None,
    session_cache: SessionSubtitleCache | None = None,
) -> None:
    """Emit each accepted utterance once; never shift its source timestamps."""
    empty_count = 0
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
            context.reset()
            last_generation = item.generation
        if isinstance(item, AudioProgress):
            if item.reset:
                context.reset()
            out_q.put(item)
            continue

        utt = item
        if utt.probe:
            # Never treat an unverified gap as dialogue. Unlike ordinary
            # speech, these low-VAD windows require JA evidence + two
            # agreeing EN passes. Do not feed their English into context.
            utt.timings.asr_start = time.monotonic()
            english, jp = _recover_background_probe(whisper, utt, cfg)
            if english and getattr(cfg, "neutral_subtitles", True):
                english = neutralize_with_japanese(english, jp)
            if english and (getattr(cfg, "remove_fillers", True)
                            or getattr(cfg, "collapse_repetitions", True)):
                english = concise_english(
                    english,
                    remove_fillers=getattr(cfg, "remove_fillers", True),
                    collapse_repetitions=getattr(cfg, "collapse_repetitions", True),
                )
            utt.timings.asr_end = time.monotonic()
            if english and (get_generation is None
                            or utt.generation == get_generation()):
                log.info(
                    "Recovered VAD-negative background speech: %.2f-%.2fs "
                    "vad=%.3f JA=%r EN=%r",
                    utt.start_s, utt.end_s, utt.vad_confidence,
                    jp[:100], english[:100],
                )
                out_q.put(Subtitle(
                    start_s=utt.start_s,
                    end_s=utt.end_s,
                    ja="", en=english, timings=utt.timings,
                    generation=utt.generation,
                    speech_start_s=utt.start_s,
                ))
            continue
        context.prepare(utt)
        cached = session_cache.match(utt.start_s, utt.end_s) if session_cache is not None else None
        if cached is not None:
            context.accept(utt, cached.en)
            out_q.put(replace(
                cached, generation=utt.generation, timings=utt.timings,
                speaker_handoff=utt.speaker_handoff or cached.speaker_handoff,
            ))
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
        japanese_evidence = ""
        if hasattr(whisper, "decode_audio"):
            original = whisper.decode_audio(
                utt.audio, prompt=combined_prompt,
                hotwords=context.hotwords(),
            )
            selected = original
            english = original.text
            seconds = len(utt.audio) / SAMPLE_RATE
            issue = _structural_issue(english or original.weak_text, seconds)
            weak_speech = (
                utt.vad_confidence is not None and utt.vad_confidence < 0.60
                and original.no_speech_prob is not None and original.no_speech_prob >= 0.60
                and original.avg_logprob is not None and original.avg_logprob < -0.35
            )
            retry_reason = issue or ("weak_speech" if weak_speech else
                                     "uncertain" if original.suspicious else "")
            silent_reject = (
                not english and original.discarded_segments > 0
                and original.no_speech_prob is not None
                and original.no_speech_prob >= float(getattr(
                    cfg, "no_speech_retry_block_threshold", 0.85,
                ))
            )
            verified = False
            if (retry_reason and getattr(cfg, "retry_uncertain", True)
                    and not silent_reject and seconds <= getattr(cfg, "retry_max_audio_s", 12.0)
                    and (get_generation is None or utt.generation == get_generation())):
                retried = True
                # Same utterance only: never concatenate a prior speech tail
                # or feed old subtitles back into this decoder pass.
                second = whisper.decode_audio(
                    utt.audio, prompt=None, hotwords=None,
                    temperature=float(getattr(cfg, "retry_temperature", 0.2)),
                )
                candidate = second.text.strip()
                first_words = _words(english or original.weak_text)
                second_words = _words(candidate)
                agrees = bool(first_words) and first_words == second_words
                second_issue = _structural_issue(candidate, seconds)
                if candidate and not second.suspicious:
                    # Preserve corroborated chants with plausible pacing. Only
                    # shorten a loop if both passes say exactly the same words.
                    repeated = (issue == "loop" and second_words
                                and first_words == second_words * (len(first_words) // len(second_words)))
                    speech_supported = (not weak_speech or
                                        (second.no_speech_prob is not None and second.no_speech_prob < 0.60))
                    plausible = speech_supported and (not second_issue or (second_issue == "loop" and agrees))
                    if plausible and (agrees or repeated or not first_words):
                        english = candidate
                        selected = second
                        verified = True
                elif not english and not candidate and not issue:
                    english = _stable_weak_speech(original, second, utt.vad_confidence, cfg)
            if (issue or weak_speech) and not verified:
                english = ""

            # Accuracy-first, EMPTY-only recovery. Whisper sometimes returns
            # zero segments for short but genuine Japanese speech. Do not
            # retry a rejected high-no-speech segment or any other uncertain
            # non-speech audio. At most 3 model passes: AST / JP / AST.
            should_recover_empty = (
                not english and not original.text
                and original.decoded_segments == 0
                and original.discarded_segments == 0
                and getattr(cfg, "recover_empty_speech", True)
                and getattr(cfg, "retry_uncertain", True)
                and utt.vad_confidence is not None
                and utt.vad_confidence >= float(getattr(
                    cfg, "recover_min_vad_confidence", 0.88))
                # A truly zero-segment decode usually has NO silence score.
                # Permit that absent metric only after very strong VAD, a
                # nonempty Japanese check and a high-confidence English retry.
                and (original.no_speech_prob is None
                     or original.no_speech_prob <= float(getattr(
                         cfg, "recover_max_no_speech_prob", 0.45)))
                and 0.45 <= seconds <= float(getattr(
                    cfg, "recover_max_audio_s", 8.0))
                and (get_generation is None or utt.generation == get_generation())
            )
            if should_recover_empty:
                verifier = getattr(whisper, "transcribe_japanese_evidence", None)
                if callable(verifier):
                    try:
                        japanese_evidence = verifier(utt.audio)
                        # A single model doing two tasks is not independent
                        # ground truth, so require specific Japanese text,
                        # strong VAD, and confident non-looping EN retry.
                        jp_supported = bool(re.search(
                            r"[぀-ヿ㐀-鿿]{2,}",
                            japanese_evidence,
                        ))
                        if jp_supported:
                            retried = True
                            retry_reason = "empty_ast_with_japanese_evidence"
                            rescue = whisper.decode_audio(
                                utt.audio, prompt=None, hotwords=None,
                                temperature=float(getattr(cfg, "retry_temperature", 0.2)),
                            )
                            candidate = rescue.text.strip()
                            confident = (
                                bool(candidate)
                                and not rescue.suspicious
                                and rescue.avg_logprob is not None
                                and rescue.avg_logprob >= -0.45
                                and rescue.no_speech_prob is not None
                                and rescue.no_speech_prob <= 0.35
                                and (rescue.compression_ratio is None
                                     or rescue.compression_ratio <= 2.0)
                                and not _structural_issue(candidate, seconds)
                                and not _phantom_phrase_kind(candidate)
                            )
                            if confident:
                                english = candidate
                                selected = rescue
                                log.info(
                                    "Recovered missed Japanese speech: gen=%d "
                                    "segment=%.2f-%.2f vad=%.3f ja=%r en=%r",
                                    utt.generation, utt.start_s, utt.end_s,
                                    utt.vad_confidence, japanese_evidence[:100],
                                    english[:100],
                                )
                    except Exception as exc:
                        log.warning("Whisper empty-speech recovery failed: %s", exc)

            phantom_kind = _phantom_phrase_kind(english)
            phantom_enabled = (
                getattr(cfg, "verify_stock_outros", True)
                if phantom_kind == "outro"
                else getattr(cfg, "filter_phantom_phrases", True)
            )
            if phantom_kind and phantom_enabled:
                duration_s = len(utt.audio) / SAMPLE_RATE
                bad_timing = _phantom_timing_implausible(english, duration_s)
                weak_audio = _phantom_weak_audio(utt.vad_confidence, selected)
                if weak_audio:
                    log.info(
                        "Suppressed Whisper phantom phrase (weak audio): "
                        "kind=%s duration=%.2fs vad=%s en=%r",
                        phantom_kind, duration_s, utt.vad_confidence,
                        english[:120],
                    )
                    english = ""
                else:
                    # Decode the identical PCM once more, unprompted, into
                    # Japanese. It is corroboration, not independent proof.
                    japanese = ""
                    verifier = getattr(whisper, "transcribe_japanese_evidence", None)
                    if callable(verifier):
                        try:
                            japanese = verifier(utt.audio)
                        except Exception as exc:
                            log.warning("Whisper phantom verification failed: %s", exc)
                    corroborated = _japanese_supports_phantom(phantom_kind, japanese)
                    # Do not reject a genuine very fast phrase solely from the
                    # English-word/JA-duration ratio when VAD is convincing.
                    timing_reject = bad_timing and (
                        utt.vad_confidence is None
                        or utt.vad_confidence < 0.85
                    )
                    if not corroborated or timing_reject:
                        log.info(
                            "Suppressed uncorroborated Whisper phantom phrase: "
                            "kind=%s duration=%.2fs timing=%s vad=%s en=%r ja_check=%r",
                            phantom_kind, duration_s, bad_timing,
                            utt.vad_confidence, english[:120], japanese[:120],
                        )
                        english = ""

        else:
            # Existing test doubles and older API-compatible adapters.
            english = whisper.translate_audio(utt.audio)

        # Whisper's normal AST response supplies English only. An additional
        # Japanese decode, when warranted, checks whether the audio actually
        # named an owner or a gender; no blind pronoun substitution.
        if (english and getattr(cfg, "neutral_subtitles", True)
                and needs_reference_evidence(english)
                and len(utt.audio) / SAMPLE_RATE <= float(
                    getattr(cfg, "neutral_verify_max_audio_s", 8.0))
                and (get_generation is None or utt.generation == get_generation())):
            verifier = getattr(whisper, "transcribe_japanese_evidence", None)
            if callable(verifier):
                try:
                    if not japanese_evidence:
                        japanese_evidence = verifier(utt.audio)
                    revised = neutralize_with_japanese(english, japanese_evidence)
                    if revised != english:
                        log.info(
                            "Whisper neutral subtitle: %r -> %r (ja=%r)",
                            english[:120], revised[:120], japanese_evidence[:100],
                        )
                        english = revised
                except Exception as exc:
                    log.warning("Whisper neutral-language check failed: %s", exc)

        utt.timings.asr_end = time.monotonic()
        raw_english = english
        remove_fillers = bool(getattr(cfg, "remove_fillers", True))
        collapse_repetitions = bool(getattr(cfg, "collapse_repetitions", True))
        # Nonverbal/filler cleanup is independent of optional concise-style
        # prompting. The default concise_english=False must not disable it.
        if remove_fillers or collapse_repetitions:
            english = concise_english(
                english,
                remove_fillers=remove_fillers,
                collapse_repetitions=collapse_repetitions,
            )
        if raw_english != english:
            log.debug(
                "Subtitle cleanup: before=%r after=%r",
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
                "rms=%.4f peak=%.3f vad_confidence=%s "
                "model=%.3fs retry=%s reason=%s "
                "avg_logprob=%s no_speech=%s compression=%s "
                "discarded=%s output=%r",
                utt.generation, utt.start_s, utt.end_s,
                utt.end_s - utt.start_s, rms, peak,
                utt.vad_confidence,
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
    vad_cfg = vad_for_source(cfg.vad, live=bool(ingest.source.live))
    log.info(
        "Whisper speech-recall VAD (%s): threshold=%.2f, min_speech=%dms, "
        "pad=%dms, silence=%dms, onset=%dms, soft_cap=%.1fs, hard_cap=%.1fs, "
        "background_probes=%s; beam=%d",
        "live" if ingest.source.live else "VOD",
        vad_cfg.threshold, vad_cfg.min_speech_ms, vad_cfg.speech_pad_ms,
        vad_cfg.min_silence_ms, vad_cfg.onset_ms, vad_cfg.max_utterance_s,
        vad_cfg.max_utterance_s + vad_cfg.max_split_extension_s,
        "on" if vad_cfg.fallback_s > 0 and cfg.whisper.background_probe_enabled else "off",
        cfg.whisper.beam_size,
    )
    audio_q: queue.Queue[AudioChunk | AudioProgress | None] = queue.Queue(maxsize=64)
    utt_q: queue.Queue[Utterance | AudioProgress | None] = queue.Queue(maxsize=16)
    sub_q: queue.Queue[Subtitle | AudioProgress | None] = queue.Queue(maxsize=16)
    failures: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    queues = {
        "audio": audio_q, "vad": utt_q,
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
            args=("vad", run_segmenter, failures, vad_cfg, audio_q, utt_q, generation),
            name="vad", daemon=True,
        ),
        threading.Thread(
            target=_guard_stage,
            args=("whisper-ast", run_whisper, failures,
                  whisper, utt_q, sub_q, generation, session_cache),
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
