"""Qt-independent, session-scoped transcript review and diagnostic tailing."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def cue_key(start: float, end: float) -> tuple[int, int]:
    return round(start * 1000), round(end * 1000)


@dataclass
class ReviewCue:
    start_s: float
    end_s: float
    ja: str
    en: str
    generation: int = 0
    revisions: list[tuple[str, str]] = field(default_factory=list)
    diagnostic: dict = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        d = self.diagnostic
        return bool(d.get("flags") or d.get("translation_flags") or d.get("fallback_reason") or
                    "uncertain" in str(d.get("chosen_by", "")) or
                    str(d.get("chosen_by", "")).startswith("abstain"))


class TranscriptReview:
    def __init__(self):
        self.cues: dict[tuple[int, int], ReviewCue] = {}
        self.diagnostics: dict[tuple[int, int], dict] = {}

    def add(self, data: dict) -> tuple[int, int]:
        start, end = data["start_s"], data["end_s"]
        if (type(start) not in (float, int) or type(end) not in (float, int)
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            raise ValueError("Invalid subtitle timestamps")
        ja, en = data.get("ja", ""), data.get("en", "")
        if not isinstance(ja, str) or not isinstance(en, str) or max(len(ja), len(en)) > 100000:
            raise ValueError("Invalid subtitle text")
        key = cue_key(start, end)
        old = self.cues.get(key)
        if old and old.revisions:
            return key  # Automated replay must never overwrite a human correction.
        self.cues[key] = ReviewCue(start, end, ja, en, data.get("generation", 0),
                                   diagnostic=dict(self.diagnostics.get(key, {})))
        return key

    def add_diagnostic(self, record: dict) -> None:
        try:
            start, end = float(record["start_s"]), float(record["end_s"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                return
            key = cue_key(start, end)
        except (KeyError, TypeError, ValueError):
            return
        # Subtitle/translation records enrich metrics, preserving ASR evidence.
        data = self.diagnostics.setdefault(key, {})
        data.update(record)
        if key in self.cues:
            self.cues[key].diagnostic = dict(data)
        elif str(record.get("chosen_by", "")).startswith("abstain"):
            self.add({"start_s": start, "end_s": end,
                      "ja": record.get("haas", ""), "en": "",
                      "generation": record.get("generation", 0)})

    def correct(self, key, ja: str, en: str) -> ReviewCue:
        if not isinstance(ja, str) or not isinstance(en, str) or not (ja.strip() or en.strip()):
            raise ValueError("Enter Japanese or English subtitle text")
        cue = self.cues[key]
        if (ja, en) != (cue.ja, cue.en):
            cue.revisions.append((cue.ja, cue.en))
            cue.ja, cue.en = ja.strip(), en.strip()
        return cue

    def filtered(self, query: str = "", needs_review: bool = False):
        query = query.casefold().strip()
        return [(key, cue) for key, cue in sorted(self.cues.items())
                if (not needs_review or cue.needs_review)
                and (not query or query in (cue.ja + " " + cue.en + " " +
                     str(cue.diagnostic)).casefold())]

    def snapshot(self):
        from .types import Subtitle
        return [Subtitle(c.start_s, c.end_s, c.ja, c.en, generation=c.generation)
                for _, c in sorted(self.cues.items())]


class DiagnosticTail:
    """Read only complete new JSONL lines; survive file truncation/replacement."""
    def __init__(self, path: Path, *, from_end: bool = True):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if from_end and self.path.exists() else 0
        self.identity = None

    def read(self, limit: int = 500) -> list[dict]:
        records = []
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if stat.st_size < self.offset or (self.identity is not None and identity != self.identity):
                self.offset = 0
            self.identity = identity
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                for _ in range(limit):
                    line = stream.readline(2_000_001)
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            records.append(record)
                    except (ValueError, UnicodeError):
                        continue
        except OSError:
            pass
        return records
