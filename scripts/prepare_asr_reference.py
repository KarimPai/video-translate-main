"""Prepare audio/review worksheets and score human-verified JA transcripts.

Reuses existing 20-minute ASR benchmark JSONL files; no GPU inference here.
CSV rows are NOT ground truth until verification_status is 'verified'.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "reference" / "hologta_30.csv"
FIELDS = (
    "id", "start_s", "end_s", "focus", "verification_status", "speech_present",
    "reference_ja", "review_note", "haas_ja", "qwen_ja", "whisper_ja",
    "haas_boundary", "qwen_boundary", "whisper_boundary",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cues(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_matched_run(root: Path, model: str, clip_size: int) -> Path | None:
    suffix = {"haas": "-haas-gemma4.jsonl", "qwen": "-qwen3-asr-gemma4.jsonl",
              "whisper": "-whisper-gemma4.jsonl"}[model]
    for run_dir in sorted(root.glob("*"), reverse=True):
        manifest = run_dir / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            config = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (config.get("clip_bytes") != clip_size
                or config.get("vad_max_utterance_s") != 14
                or config.get("vad_threshold") != 0.28):
            continue
        matches = sorted(run_dir.glob("*" + suffix))
        if matches:
            return matches[0]
    return None


def aligned_text(cues: list[dict], start: float, end: float) -> tuple[str, str]:
    """Align full-run cues by midpoint; flag boundary-spanning cues.

    WARNING: a midpoint-aligned hypothesis is an exploratory comparison, NOT
    equivalent to running the ASR on the exact cropped audio window.
    """
    selected = [cue for cue in cues
                if start <= (float(cue["start_s"]) + float(cue["end_s"])) / 2 < end]
    boundary = any(float(cue["start_s"]) < start - 0.15
                   or float(cue["end_s"]) > end + 0.15 for cue in selected)
    return " ".join(str(cue.get("ja", "")).strip() for cue in selected if cue.get("ja")), (
        "partial_cue" if boundary else ""
    )


def prepare(args: argparse.Namespace) -> int:
    clip = args.clip.resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"Clip not found: {clip}")
    rows = read_csv(args.manifest)
    if len(rows) != 30 or len({r["id"] for r in rows}) != 30:
        raise ValueError("Expected 30 unique reference windows")
    paths = {}
    cues = {}
    for model in ("haas", "qwen", "whisper"):
        provided = getattr(args, model)
        path = provided or latest_matched_run(args.runs_root, model, clip.stat().st_size)
        if path is None:
            raise FileNotFoundError(
                f"No matched {model} benchmark JSONL. Pass --{model} PATH, or run "
                "the model at VAD 14s, threshold 0.28 on the same clip."
            )
        print(f"{model}: {path}")
        paths[model] = str(path)
        cues[model] = load_cues(path)

    result = []
    for row in rows:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not (0 <= start < end <= 1201):
            raise ValueError(f"Invalid clip-local window {row['id']}: {start}-{end}")
        item = dict(row)
        item.update(verification_status="pending", speech_present="", reference_ja="",
                    review_note="")
        for model in ("haas", "qwen", "whisper"):
            item[f"{model}_ja"], item[f"{model}_boundary"] = aligned_text(
                cues[model], start, end)
        result.append(item)

    args.outdir.mkdir(parents=True, exist_ok=True)
    review_path = args.outdir / "hologta_30_review.csv"
    if review_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite corrected review sheet: {review_path}")
    write_csv(review_path, result, FIELDS)
    (args.outdir / "sources.json").write_text(
        json.dumps({"clip": str(clip), "clip_bytes": clip.stat().st_size,
                    "hypotheses": paths, "warning": "AI transcripts are NOT ground truth"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Review CSV: {review_path}")
    if args.extract_audio:
        audio = args.outdir / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        for row in result:
            target = audio / f"{row['id']}.wav"
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                       "-ss", str(row["start_s"]), "-i", str(clip),
                       "-t", str(round(float(row["end_s"]) - float(row["start_s"]), 3)),
                       "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
            subprocess.run(command, check=True)
        print(f"Extracted {len(result)} WAVs: {audio}")
    return 0


def normalize_ja(text: str) -> str:
    """Remove spacing and punctuation; retain kana, kanji, and actual words."""
    value = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in value if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    prev = list(range(len(hypothesis) + 1))
    for index, char in enumerate(reference, 1):
        curr = [index]
        for col, other in enumerate(hypothesis, 1):
            curr.append(min(curr[-1] + 1, prev[col] + 1,
                            prev[col - 1] + (char != other)))
        prev = curr
    return prev[-1]


def score_rows(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    details: list[dict] = []
    aggregates = {model: {"ref_chars": 0, "edit_errors": 0, "speech_windows": 0,
                          "empty_speech_windows": 0, "no_speech_windows": 0,
                          "false_speech_windows": 0, "boundary_warnings": 0}
                  for model in ("haas", "qwen", "whisper")}
    for row in rows:
        if row.get("verification_status", "").strip().casefold() != "verified":
            continue
        truth = row.get("speech_present", "").strip().casefold()
        if truth not in ("yes", "no"):
            raise ValueError(f"{row['id']}: verified requires speech_present=yes or no")
        reference = normalize_ja(row.get("reference_ja", ""))
        if (truth == "yes" and not reference) or (truth == "no" and reference):
            raise ValueError(f"{row['id']}: reference_ja must be filled for speech, blank for no speech")
        for model, stats in aggregates.items():
            hypothesis = normalize_ja(row.get(f"{model}_ja", ""))
            boundary = bool(row.get(f"{model}_boundary", "").strip())
            stats["boundary_warnings"] += boundary
            if truth == "yes":
                stats["speech_windows"] += 1
                stats["empty_speech_windows"] += not hypothesis
                edits = edit_distance(reference, hypothesis)
                stats["ref_chars"] += len(reference)
                stats["edit_errors"] += edits
            else:
                stats["no_speech_windows"] += 1
                stats["false_speech_windows"] += bool(hypothesis)
                edits = 0
            details.append({"id": row["id"], "model": model, "speech_present": truth,
                            "reference_chars": len(reference), "hypothesis_chars": len(hypothesis),
                            "edit_errors": edits, "boundary_warning": boundary,
                            "window_cer": round(edits / len(reference), 4) if reference else ""})
    summary = []
    for model, stats in aggregates.items():
        summary.append({"model": model, "verified_windows": stats["speech_windows"] + stats["no_speech_windows"],
                        **stats, "cer": round(stats["edit_errors"] / stats["ref_chars"], 4)
                        if stats["ref_chars"] else None,
                        "missing_speech_window_rate": round(
                            stats["empty_speech_windows"] / stats["speech_windows"], 4)
                        if stats["speech_windows"] else None,
                        "false_speech_window_rate": round(
                            stats["false_speech_windows"] / stats["no_speech_windows"], 4)
                        if stats["no_speech_windows"] else None})
    return summary, details


def score(args: argparse.Namespace) -> int:
    summary, details = score_rows(read_csv(args.sheet))
    if not summary[0]["verified_windows"]:
        print("No verified references yet. Listen to the audio and correct the sheet first.")
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "scores.csv", summary, tuple(summary[0]))
    write_csv(args.outdir / "window_errors.csv", details, tuple(details[0]))
    (args.outdir / "scores.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in summary:
        print(f"{line['model']}: CER={line['cer']} verified={line['verified_windows']} "
              f"empty_speech={line['empty_speech_windows']} "
              f"boundary_warnings={line['boundary_warnings']}")
    print("WARNING: whole-run cue alignment is approximate; boundary warnings can inflate CER.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create human correction CSV and optional WAV excerpts")
    prep.add_argument("clip", type=Path)
    prep.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prep.add_argument("--runs-root", type=Path, default=ROOT / "benchmarks" / "runs")
    for model in ("haas", "qwen", "whisper"):
        prep.add_argument(f"--{model}", type=Path, help="explicit benchmark JSONL source")
    prep.add_argument("--outdir", type=Path, default=ROOT / "benchmarks" / "reference" / "working")
    prep.add_argument("--extract-audio", action="store_true", help="extract 30 WAVs using ffmpeg")
    prep.add_argument("--overwrite", action="store_true", help="replace previous review CSV; loses corrections")
    evaluate = sub.add_parser("score", help="Calculate CER only from manually verified reference rows")
    evaluate.add_argument("--sheet", type=Path, default=ROOT / "benchmarks" / "reference" / "working" / "hologta_30_review.csv")
    evaluate.add_argument("--outdir", type=Path, default=ROOT / "benchmarks" / "reference" / "working" / "scores")
    args = parser.parse_args(argv)
    try:
        return prepare(args) if args.command == "prepare" else score(args)
    except (FileNotFoundError, FileExistsError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
