"""Automated Japanese ASR disagreement triage; no manual Japanese required.

This is NOT an accuracy score. Consensus can be wrong, and unmatched cue
boundaries can create apparent disagreements.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEET = ROOT / "benchmarks/reference/working/hologta_30_review.csv"
DEFAULT_OUT = ROOT / "benchmarks/reference/working/analysis"
MODELS = ("haas", "qwen", "whisper")
FIELDS = (
    "id", "start_s", "end_s", "original_stream_time", "focus", "priority",
    "signals", "disagreement", "present_models", "missing_models",
    "boundary_models", "haas_chars", "qwen_chars", "whisper_chars",
    "haas_ja", "qwen_ja", "whisper_ja",
)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in value if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (letter != other)))
        previous = current
    return previous[-1]


def difference(a: str, b: str) -> float:
    """Normalized distance, NOT character error rate."""
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))


def clock(value: float) -> str:
    hours, remain = divmod(int(value), 3600)
    minutes, seconds = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def analyze_row(row: dict[str, str], original_start_s: float = 5100.0) -> dict:
    texts = {name: normalize(row.get(f"{name}_ja", "")) for name in MODELS}
    usable = [name for name in MODELS
              if row.get(f"{name}_boundary", "") != "source_unavailable"]
    present = [name for name in usable if texts[name]]
    absent = [name for name in usable if not texts[name]]
    boundaries = [name for name in usable
                  if row.get(f"{name}_boundary", "") == "partial_cue"]
    pairs = [(a, b) for i, a in enumerate(usable) for b in usable[i + 1:]]
    disagreement = (statistics.mean(difference(texts[a], texts[b]) for a, b in pairs)
                    if pairs else None)
    signals = []
    if len(usable) < len(MODELS):
        signals.append("incomplete_model_data")
    if not present:
        signals.append("speech_presence_unverified")
    elif len(present) == 1 and len(usable) >= 2:
        signals.append("only_one_model_emits")
    elif absent and len(present) >= 2 and max(len(texts[n]) for n in present) >= 8:
        signals.append("one_or_more_models_empty")
    if disagreement is not None:
        if disagreement >= .70:
            signals.append("very_high_disagreement")
        elif disagreement >= .45:
            signals.append("high_disagreement")
    if boundaries:
        signals.append("cue_boundary_mismatch")
    focus = row.get("focus", "")
    if any(tag in focus.casefold() for tag in
           ("name", "lamy", "miko", "lui", "marine", "nickname")):
        signals.append("name_sensitive_window")
    if "phonetic-loop" in focus or "Whisper-collapse" in focus:
        signals.append("known_tricky_audio")
    weight = (5 * ("one_or_more_models_empty" in signals)
              + 4 * ("only_one_model_emits" in signals)
              + 4 * ("very_high_disagreement" in signals)
              + 2 * ("high_disagreement" in signals)
              + 2 * ("name_sensitive_window" in signals)
              + 2 * ("known_tricky_audio" in signals)
              + 1 * ("cue_boundary_mismatch" in signals))
    priority = "high" if weight >= 5 else "medium" if weight >= 2 else "low"
    return {
        "id": row["id"], "start_s": row["start_s"], "end_s": row["end_s"],
        "original_stream_time": clock(original_start_s + float(row["start_s"])),
        "focus": focus, "priority": priority, "signals": ";".join(signals),
        "disagreement": round(disagreement, 3) if disagreement is not None else "",
        "present_models": ";".join(present), "missing_models": ";".join(absent),
        "boundary_models": ";".join(boundaries),
        **{f"{name}_chars": len(texts[name]) for name in MODELS},
        **{f"{name}_ja": row.get(f"{name}_ja", "") for name in MODELS},
    }


def analyze(rows: list[dict[str, str]], offset: float = 5100.0) -> list[dict]:
    return [analyze_row(row, offset) for row in rows]


def summary(results: list[dict]) -> dict:
    priorities = {name: sum(row["priority"] == name for row in results)
                  for name in ("high", "medium", "low")}
    numbers = [float(row["disagreement"]) for row in results
               if row["disagreement"] != ""]
    return {
        "windows": len(results),
        "priority_counts": priorities,
        "missing_hypothesis_windows": {
            name: sum(name in row["missing_models"].split(";") for row in results)
            for name in MODELS
        },
        "unreviewed_speech_presence": sum(
            "speech_presence_unverified" in row["signals"] for row in results
        ),
        "boundary_mismatch_windows": sum(bool(row["boundary_models"]) for row in results),
        "mean_pairwise_disagreement": round(statistics.mean(numbers), 3) if numbers else None,
        "warning": "Model disagreement only; NOT CER, accuracy, confidence, or verified silence.",
    }


def report(results: list[dict], stats: dict, limit: int = 15) -> str:
    ordered = sorted(
        results,
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2}[row["priority"]],
            -(float(row["disagreement"]) if row["disagreement"] != "" else -1),
            row["id"],
        ),
    )
    lines = [
        "# holoGTA ASR disagreement report — no Japanese transcription required", "",
        "**This is NOT an accuracy ranking.** Models can agree on the same wrong",
        "transcript. Segment-boundary differences also affect disagreement scores.",
        "All-empty model output does not establish that an audio window is silent.", "",
        f"- Windows analyzed: **{stats['windows']}**",
        f"- High-priority: **{stats['priority_counts']['high']}**",
        f"- Medium-priority: **{stats['priority_counts']['medium']}**",
        f"- Low-priority: **{stats['priority_counts']['low']}**",
        f"- Unverified speech-presence: **{stats['unreviewed_speech_presence']}**",
        f"- Cue-boundary warnings: **{stats['boundary_mismatch_windows']}**",
        f"- Average pairwise disagreement (not CER): **{stats['mean_pairwise_disagreement']}**", "",
        "## Windows most useful for regression tests", "",
        "Times are local to the 20-minute clip; stream times assume clip offset 01:25:00.", "",
        "| Window | Clip time | Original stream | Risk flags | Empty model outputs |",
        "|---|---|---|---|---|",
    ]
    for row in ordered[:max(limit, 0)]:
        begin, end = clock(float(row["start_s"])), clock(float(row["end_s"]))
        signals = row["signals"].replace("_", " ").replace(";", ", ") or "low disagreement"
        empty = row["missing_models"].replace(";", ", ") or "none"
        lines.append(
            f"| {row['id']} | {begin}–{end} | {row['original_stream_time']} | {signals} | {empty} |"
        )
    lines += [
        "", "## Reading the report", "",
        "- A model with no text may have dropped speech, or other models may hallucinate.",
        "- High disagreement means different transcriptions, **not** verified errors.",
        "- Name-sensitive windows warrant cautious nickname handling.",
        "- Unverified speech-presence windows must not be classified as silence.", "",
        "## Next engineering step", "",
        "Keep Haas + Silero (VAD 0.28, 14s) as the provisional baseline. Replay",
        "high-disagreement windows to investigate missing speech and named entities.",
        "Real CER requires Japanese-grounded transcriptions, validated subtitles,",
        "or a Japanese-fluent reviewer. The CSV retains Japanese hypotheses for",
        "developers; you do not need to read Japanese to use this English report.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stream-offset-s", type=float, default=5100.0,
                        help="original stream offset for the clip (default 01:25:00)")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    if not args.sheet.is_file():
        parser.error(f"Review worksheet missing: {args.sheet}. First run prepare.")
    with args.sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "start_s", "end_s", "haas_ja", "qwen_ja", "whisper_ja"}
    if not rows or not required.issubset(rows[0]):
        parser.error("Review worksheet is empty or missing model columns.")
    results = analyze(rows, args.stream_offset_s)
    stats = summary(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "disagreements.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    (args.outdir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (args.outdir / "report.md").write_text(report(results, stats, args.top), encoding="utf-8")
    print(f"Windows: {len(results)} | high={stats['priority_counts']['high']} "
          f"medium={stats['priority_counts']['medium']} "
          f"low={stats['priority_counts']['low']}")
    print(f"English report: {args.outdir / 'report.md'}")
    print("NOTE: disagreement is not accuracy; no Japanese ground truth is assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
