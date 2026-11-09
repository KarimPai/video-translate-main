"""Japanese YouTube captions (manual or auto), parsing and alignment."""

import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from difflib import SequenceMatcher
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from .config import IngestConfig
from .text_filter import (is_isolated_reaction, is_usable_transcript,
                          strip_nonverbal_labels)

log = logging.getLogger(__name__)

_TIMESTAMP = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
_TIMING_LINE = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)")
_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")


def _seconds(value: str) -> float | None:
    match = _TIMESTAMP.fullmatch(value.strip())
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def _clean_caption(lines: list[str]) -> str:
    value = " ".join(lines)
    value = _TAG.sub("", value)
    value = html.unescape(value)
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return _SPACE.sub(" ", value).strip()


@dataclass(frozen=True)
class CaptionCue:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class CaptionMatch:
    text: str
    overlap_s: float
    coverage: float
    reliable: bool
    speech_cues: int = 1


class CaptionIndex:
    """Small immutable VTT index matched against source-timeline timestamps."""

    def __init__(self, cues: list[CaptionCue]) -> None:
        self.cues = sorted(cues, key=lambda cue: (cue.start_s, cue.end_s))
        self._starts = [cue.start_s for cue in self.cues]
        # Ends can be out of order when a long cue overlaps shorter ones.
        self._latest_ends = list(accumulate((cue.end_s for cue in self.cues), max))

    @classmethod
    def from_vtt(cls, content: str) -> "CaptionIndex":
        cues: list[CaptionCue] = []
        for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n")):
            lines = block.splitlines()
            timing_index = next(
                (i for i, line in enumerate(lines) if "-->" in line),
                None,
            )
            if timing_index is None:
                continue
            timing = _TIMING_LINE.match(lines[timing_index])
            if timing is None:
                continue
            start = _seconds(timing.group(1))
            end = _seconds(timing.group(2))
            if start is None or end is None or end <= start:
                continue
            payload = lines[timing_index + 1:]
            # YouTube rolling VTT can retain the previous line above a new
            # timestamped line. Discard it only when it matches the last cue.
            if (cues and len(payload) > 1 and cues[-1].end_s >= start - 0.05
                    and re.search(r"<(?:\d+:)?\d{2}:\d{2}[.,]\d{3}>", " ".join(payload[1:]))
                    and _clean_caption(payload[:1]) == cues[-1].text):
                payload = payload[1:]
            text = _clean_caption(payload)
            if not text:
                continue
            cues.append(CaptionCue(start, end, text))
        return cls(cues)

    @classmethod
    def from_file(cls, path: Path) -> "CaptionIndex":
        return cls.from_vtt(path.read_text(encoding="utf-8-sig"))

    def match(
        self,
        start_s: float,
        end_s: float,
        tolerance_s: float = 0.15,
    ) -> CaptionMatch | None:
        duration = max(0.01, end_s - start_s)
        matched: list[tuple[CaptionCue, float]] = []
        first = bisect_left(self._latest_ends, start_s - tolerance_s)
        stop = bisect_right(self._starts, end_s + tolerance_s)
        for index in range(first, stop):
            cue = self.cues[index]
            if cue.end_s < start_s - tolerance_s:
                continue
            overlap = max(
                0.0,
                min(end_s, cue.end_s) - max(start_s, cue.start_s),
            )
            if overlap > 0:
                matched.append((cue, overlap))

        if not matched:
            return None

        # Stage directions cannot lend their overlap to a sliver of dialogue.
        cleaned = [(cue, overlap, strip_nonverbal_labels(cue.text)) for cue, overlap in matched]
        speech = [entry for entry in cleaned if entry[2]]
        evidence = speech or cleaned
        texts: list[str] = []
        covered_to = start_s
        overlap_s = 0.0
        well_aligned = True
        for cue, overlap, line in evidence:
            left, right = max(start_s, cue.start_s), min(end_s, cue.end_s)
            overlap_s += max(0.0, right - max(left, covered_to))
            covered_to = max(covered_to, right)
            # A short segment inside a long cue cannot verify the whole line.
            if overlap < 0.40 * (cue.end_s - cue.start_s):
                well_aligned = False
            if not line:
                continue
            if texts and line.startswith(texts[-1]):
                texts[-1] = line
            elif not texts or texts[-1] != line:
                texts.append(line)

        coverage = overlap_s / duration
        return CaptionMatch(
            text=" ".join(texts),
            overlap_s=overlap_s,
            coverage=coverage,
            reliable=well_aligned and coverage >= 0.60 and overlap_s >= 0.25,
            speech_cues=len(texts),
        )


# Strict agreement-based caption priority. YouTube's autogenerated captions
# can lag, overlap voices, repeat text or omit words; timestamps alone never
# establish that the caption is the same speaker/utterance as the VAD segment.
_COMPACT = re.compile(r"[\s\u3000、。，．・!?！？…〜～.,:;:「」『』（）()\[\]【】ー]+")
_JP = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_NEGATION = re.compile(r"ない|無い|ません|じゃない|ではない|できない|出来ない|ずに|なかった")


def _caption_compact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch
        for ch in value
    )
    return _COMPACT.sub("", value)


def prefer_youtube_japanese(
    haas: str,
    match: CaptionMatch | None,
) -> tuple[str, str]:
    """Prefer tightly aligned Japanese captions when Haas supports the reading.

    Returns (Japanese text, evidence label). ASR is always decoded independently
    and remains authoritative for missing/delayed captions, clear disagreement,
    uncertain negation and extra speech outside a shorter caption. In contrast,
    a closely matching caption fixes homophones and proper-name spelling.
    """
    if match is None or not match.reliable or match.coverage < 0.85 or match.overlap_s < 0.45:
        return haas, "haas_asr"
    caption = strip_nonverbal_labels(match.text)
    if (not caption or not is_usable_transcript(caption)
            or is_isolated_reaction(caption) or not _JP.search(caption)):
        return haas, "haas_asr"

    a = _caption_compact(haas)
    c = _caption_compact(caption)
    if not c:
        return haas, "haas_asr"

    # Empty/obviously nonlexical recognition may be recovered when the
    # caption is strongly aligned. Do not replace short, meaningful replies
    # from another speaker by an unrelated multiword caption.
    if not a or not is_usable_transcript(haas) or is_isolated_reaction(haas):
        if len(c) >= 4 and match.speech_cues == 1:
            return caption, "youtube_primary_haas_recovery"
        return haas, "haas_asr"

    if a == c:
        return caption, "youtube_primary_haas_agreed"

    # A single aligned caption may complete a Haas transcript cut short
    # at the end. Require a meaningful shared four-character prefix rather
    # than guessing based on timestamps alone.
    if (match.speech_cues == 1 and len(a) >= 4
            and c.startswith(a) and len(c) >= len(a) + 3
            and bool(_NEGATION.search(haas)) == bool(_NEGATION.search(caption))):
        return caption, "youtube_primary_haas_completed"

    # A two-cue match could include another speaker. Never switch the
    # entire transcript unless both recognize essentially the same words.
    agreement = SequenceMatcher(None, a, c, autojunk=False).ratio()
    min_agreement = 0.88 if match.speech_cues != 1 else 0.68
    if agreement < min_agreement:
        return haas, "haas_asr_disagreement"

    # ASR might contain a second spoken clause omitted from YouTube captions.
    if a.startswith(c) and len(a) >= len(c) + 3:
        return haas, "haas_asr_extra_speech"
    # A negation that exists only on one side may invert meaning; with no
    # independent ground truth, prefer Haas over caption replacement.
    if bool(_NEGATION.search(haas)) != bool(_NEGATION.search(caption)):
        return haas, "haas_asr_polarity_guard"
    return caption, "youtube_primary_haas_verified"


class PendingCaptions:
    """Publish a complete index when the background VTT download finishes."""

    def __init__(self) -> None:
        self.index: CaptionIndex | None = None

    def match(self, start_s: float, end_s: float) -> CaptionMatch | None:
        index = self.index
        return index.match(start_s, end_s) if index is not None else None


def _caption_cache_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "auto-translate" / "timelines"


def download_vod_captions(
    video_id: str,
    url: str,
    cfg: IngestConfig,
) -> CaptionIndex | None:
    """Download/cache Japanese captions; never raises for unavailable captions."""
    cache_dir = _caption_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{video_id}.ja.vtt"

    if cached.is_file():
        try:
            index = CaptionIndex.from_file(cached)
            if index.cues:
                log.info(
                    "Japanese YouTube captions loaded from cache (%d cues)",
                    len(index.cues),
                )
                return index
        except (OSError, UnicodeError):
            log.warning("could not read cached captions: %s", cached)

    command = [
        "yt-dlp",
        "--no-warnings",
        "--no-call-home",
        "--skip-download",
        "--write-subs",  # use creator-provided Japanese captions when available
        "--write-auto-subs",  # otherwise use Japanese auto-captions
        "--sub-langs", "ja.*,ja",
        "--sub-format", "vtt/best",
        "--output", str(cache_dir / f"{video_id}.%(ext)s"),
    ]
    if cfg.cookies_from_browser:
        command += ["--cookies-from-browser", cfg.cookies_from_browser]
    command.append(url)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"captions-{video_id}-",
            dir=str(cache_dir),
        ) as temp_dir:
            command[command.index("--output") + 1] = str(
                Path(temp_dir) / f"{video_id}.%(ext)s"
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            candidates = sorted(Path(temp_dir).glob("*.vtt"))
            if result.returncode != 0 and not candidates:
                detail = (result.stderr or result.stdout).strip().splitlines()
                log.info(
                    "Japanese YouTube captions unavailable: %s",
                    detail[-1] if detail else f"yt-dlp exit code {result.returncode}",
                )
                return None
            if not candidates:
                log.info("this VOD has no downloadable Japanese captions")
                return None

            # Prefer the Japanese track, but accept a Japanese regional variant.
            candidates.sort(
                key=lambda path: (
                    0 if ".ja." in path.name else 1,
                    path.name,
                )
            )
            shutil.copy2(candidates[0], cached)

    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Japanese caption download failed; continuing with Haas ASR: %s", exc)
        return None

    try:
        index = CaptionIndex.from_file(cached)
    except (OSError, UnicodeError) as exc:
        log.warning("could not parse downloaded captions; continuing with Haas ASR: %s", exc)
        return None

    if not index.cues:
        log.info("downloaded Japanese caption file contained no usable cues")
        return None

    log.info("Japanese YouTube captions ready (%d cues)", len(index.cues))
    return index
