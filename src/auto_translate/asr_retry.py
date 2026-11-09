"""Selective, evidence-based Haas re-decoding without an extra ASR model.

Repeated deterministic decodes on unchanged inputs do not constitute a vote.
Try at most one small boundary variant and one unboosted-keyword variant, only
for uncertain recognized speech. Captions are advisory only when closely aligned.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np

from .text_filter import rejection_reason, strip_nonverbal_labels

log = logging.getLogger(__name__)

_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[\s\u3000、。，．・!?！？…〜～.,:;:「」『』（）()\[\]【】ー]+")


@dataclass(frozen=True)
class Candidate:
    text: str
    attempt: int
    score: float


def compact(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text).casefold())


def _strong_caption(caption: Any) -> str:
    if (caption is not None
            and getattr(caption, "reliable", False)
            and getattr(caption, "coverage", 0.0) >= 0.80
            and getattr(caption, "overlap_s", 0.0) >= 0.45):
        return str(getattr(caption, "text", "") or "")
    return ""


def _caption_agreement(text: str, caption_text: str) -> float:
    a, b = compact(text), compact(caption_text)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def quality_score(text: str, *, duration_s: float, caption: Any = None) -> float:
    """Absolute textual plausibility plus independent caption agreement.

    Text length alone never awards points; this avoids favoring hallucinated
    verbose decodes over an accurate short answer.
    """
    text = strip_nonverbal_labels(text)
    normalized = compact(text)
    if not normalized or rejection_reason(text) is not None:
        return -8.0
    score = 0.0
    if _JAPANESE.search(normalized):
        score += 1.0
    else:
        score -= 0.8  # English words can be real, so not an automatic rejection
    # Two-character results on long (>2s) voiced utterances are suspect.
    if duration_s >= 2.0 and len(normalized) <= 2:
        score -= 2.2
    elif duration_s >= 4.0 and len(normalized) <= 5:
        score -= 1.0
    # Decoder artifacts often insert isolated Latin letters into Japanese.
    if _JAPANESE.search(normalized):
        if re.search(r"(?<![a-z])[a-z](?![a-z])", normalized):
            score -= 1.0
        elif _LATIN.search(normalized):
            score -= 0.2  # Do not reject real GTA / OK / JP-English speech
    verified = _strong_caption(caption)
    if verified:
        score += 3.0 * _caption_agreement(text, verified)
    return score


def needs_retry(text: str, *, duration_s: float, caption: Any = None) -> bool:
    normalized = compact(strip_nonverbal_labels(text))
    if not normalized or rejection_reason(text) is not None:
        return True
    if duration_s >= 2.0 and len(normalized) <= 2:
        return True
    if duration_s >= 4.0 and len(normalized) <= 5:
        return True
    if (_JAPANESE.search(normalized)
            and re.search(r"(?<![a-z])[a-z](?![a-z])", normalized)):
        return True
    verified = _strong_caption(caption)
    return bool(verified and len(compact(verified)) >= 6
                and _caption_agreement(text, verified) < 0.55)


def select_best(candidates: list[str], *, duration_s: float, caption: Any = None) -> Candidate:
    """Keep first decode on ties; require meaningful score improvement."""
    first = Candidate(
        strip_nonverbal_labels(candidates[0]), 1,
        quality_score(candidates[0], duration_s=duration_s, caption=caption),
    )
    best = first
    for number, text in enumerate(candidates[1:], start=2):
        cleaned = strip_nonverbal_labels(text)
        score = quality_score(cleaned, duration_s=duration_s, caption=caption)
        if score > best.score + 0.35:
            best = Candidate(cleaned, number, score)
    return best


def choose_hypothesis(
    asr: Any,
    audio: np.ndarray,
    original: str,
    *,
    start_s: float,
    end_s: float,
    speech_start_s: float | None = None,
    probe: bool = False,
    caption: Any = None,
    live: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """Return the strongest available decode before the retry deadline.

    Initial recognition is performed by the caller and retains its existing
    response timeout. Only *extra* attempts share the strict budget. A worker
    timed out by speech.cpp may still finish internally; this function stops
    sending additional retry requests immediately if a timeout occurs.
    """
    cfg = getattr(asr, "cfg", None)
    bounded = getattr(asr, "transcribe_bounded", None)
    duration = max(0.0, end_s - start_s)
    if (probe or cfg is None or not callable(bounded)
            or not getattr(cfg, "retry_uncertain_enabled", False)
            or duration > getattr(cfg, "retry_max_audio_s", 8.0)
            or duration < 1.0
            or not needs_retry(original, duration_s=duration, caption=caption)):
        return original

    limit = int(getattr(cfg, "retry_max_passes", 3))
    if live:
        limit = min(limit, int(getattr(cfg, "retry_live_max_passes", 2)))
    limit = max(1, min(limit, 3))
    if limit <= 1:
        return original
    budget = float(
        getattr(cfg, "retry_live_budget_s", 1.5) if live
        else getattr(cfg, "retry_budget_s", 3.0)
    )
    if budget <= 0:
        return original

    # Prefer modifications that retain speech: VAD's 200ms preroll can
    # safely lose 60ms; otherwise trim at most 60ms at the utterance end,
    # and do not trim the hard-cut longform segments.
    variants: list[tuple[np.ndarray, bool, str]] = []
    onset_padding = max(0.0, (speech_start_s or start_s) - start_s)
    if onset_padding >= 0.10 and len(audio) > 1.0 * 16000:
        variants.append((audio[960:], True, "shorter-preroll"))
    elif 1.0 < duration < 6.0 and len(audio) > 1.0 * 16000 + 960:
        variants.append((audio[:-960], True, "shorter-tail"))
    # When keywords are available, removing them is genuinely different
    # from repeating the same deterministic decoding request.
    if bool(getattr(asr, "keywords", "")):
        variants.append((audio, False, "unboosted-keywords"))
    # Do not perform redundant decodes when no meaningful variant is available.
    if not variants:
        return original

    candidates = [original]
    retry_deadline = time.monotonic() + budget
    for variant_audio, keywords, description in variants[:limit - 1]:
        if cancelled is not None and cancelled():
            break  # seek or new stream generation: no extra ASR work
        remaining = retry_deadline - time.monotonic()
        if remaining < 0.08:
            break
        try:
            candidate = bounded(
                variant_audio, timeout_s=remaining, use_keywords=keywords,
            )
            candidates.append(candidate)
            log.debug("Haas retry %s returned %r", description, candidate[:90])
        except TimeoutError:
            log.info("Haas multi-pass deadline reached; keeping best completed transcript")
            break
        except Exception as exc:
            # A retry is optional; never discard an otherwise usable first pass.
            log.warning("Optional Haas retry %s failed: %s", description, exc)
            break

        if not needs_retry(candidate, duration_s=duration, caption=caption):
            # Only stop early if the candidate is clearly better than the
            # first hypothesis, not merely different.
            current = select_best(candidates, duration_s=duration, caption=caption)
            if current.attempt > 1:
                break

    chosen = select_best(candidates, duration_s=duration, caption=caption)
    if chosen.attempt != 1:
        log.debug(
            "Haas selected pass %d/%d (quality %.2f): %r -> %r",
            chosen.attempt, len(candidates), chosen.score,
            original[:100], chosen.text[:100],
        )
    return chosen.text
