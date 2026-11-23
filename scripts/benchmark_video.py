"""Run reproducible local Japanese-to-English ASR/MT benchmarks.

This tool runs the existing CLI on the same downloaded local clip. It deliberately
does not calculate an "accuracy score" without a human-checked reference.
Nothing here alters production VAD, inference, or subtitle behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = r"\d+:\d{2}(?::\d{2})?\.\d+"
CUE_RE = re.compile(
    rf"^\[(?P<start>{STAMP})–(?P<end>{STAMP})\]\s+"
    r"(?P<ja>.*?)\s+\(vad (?P<vad>-?\d+(?:\.\d+)?)s"
    r" · wait (?P<wait>-?\d+(?:\.\d+)?)s"
    r" · asr (?P<asr>-?\d+(?:\.\d+)?)s"
    r" · mt (?P<mt>-?\d+(?:\.\d+)?)s\)"
)
EN_RE = re.compile(r"^\s*→\s+(.+?)\s*$")


def seconds(stamp: str) -> float:
    parts = [float(x) for x in stamp.split(":")]
    value = 0.0
    for part in parts:
        value = 60.0 * value + part
    return value


def parse_log(text: str) -> list[dict]:
    """Extract completed cues from the default timestamped TerminalOutput."""
    cues: list[dict] = []
    for line in text.splitlines():
        match = CUE_RE.match(line)
        if match:
            row = match.groupdict()
            cues.append({
                "start_s": seconds(row["start"]),
                "end_s": seconds(row["end"]),
                "ja": row["ja"].strip(),
                "en": "",
                **{key + "_s": float(row[key]) for key in ("vad", "wait", "asr", "mt")},
            })
            continue
        translated = EN_RE.match(line)
        if translated and cues and not cues[-1]["en"]:
            cues[-1]["en"] = translated.group(1)
    return cues


def summarize(cues: list[dict], wall_s: float, status: int) -> dict:
    """Operational metrics only. None of these establishes translation quality."""
    durations = [c["end_s"] - c["start_s"] for c in cues]
    result = {
        "exit_code": status,
        "cue_count": len(cues),
        "translated_count": sum(bool(c["en"]) for c in cues),
        "japanese_chars": sum(len(c["ja"]) for c in cues),
        "short_cues_under_1s": sum(duration < 1.0 for duration in durations),
        "long_cues_ge_14s": sum(duration >= 14.0 for duration in durations),
        "long_cues_ge_10s": sum(duration >= 10.0 for duration in durations),
        "english_cues_over_160_chars": sum(len(c["en"]) > 160 for c in cues),
        "median_cue_duration_s": round(statistics.median(durations), 2) if durations else None,
        "wall_s": round(wall_s, 2),
        "mean_asr_s": None,
        "mean_mt_s": None,
        "p95_asr_s": None,
        "p95_mt_s": None,
    }
    if cues:
        for stage in ("asr", "mt"):
            measures = sorted(c[stage + "_s"] for c in cues)
            result["mean_" + stage + "_s"] = round(statistics.mean(measures), 3)
            result["p95_" + stage + "_s"] = round(
                measures[max(0, -(-len(measures) * 95 // 100) - 1)], 3
            )
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential, offline ASR/MT comparisons on one local video or audio clip."
    )
    parser.add_argument("clip", type=Path, help="local media file, e.g. hologta-stress-test.webm")
    parser.add_argument("--asr", nargs="+", default=["haas"],
                        help="ASR backend(s), e.g. haas qwen3-asr whisper")
    parser.add_argument("--mt", nargs="+", default=["gemma4"],
                        help="translation backend(s), e.g. gemma4 plamo")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--context-lines", type=int, choices=range(21),
                        help="optional context experiment; omit for unmodified baseline")
    parser.add_argument("--vad-max-utterance-s", type=float,
                        help="optional VOD soft segment cap; e.g. 10 (does not edit config.toml)")
    parser.add_argument("--vad-threshold", type=float,
                        help="optional VAD threshold, e.g. 0.28 for matched ASR evaluation")
    parser.add_argument("--outdir", type=Path, default=ROOT / "benchmarks" / "runs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without loading models")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="continue with other models after one failed run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    clip = args.clip.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not clip.is_file():
        print(f"Clip not found: {clip}", file=sys.stderr)
        return 2
    if not config.is_file():
        print(f"Config not found: {config}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = args.outdir.expanduser().resolve() / stamp
    combos = [(asr, mt) for asr in args.asr for mt in args.mt]
    plan = []
    for number, (asr, mt) in enumerate(combos, 1):
        name = f"{number:02d}-" + re.sub(r"[^a-zA-Z0-9_-]", "_", f"{asr}-{mt}")
        command = [
            sys.executable, "-m", "auto_translate", str(clip),
            "--config", str(config), "--mode", "vod", "--fast",
            "--no-web", "--safe-playback", "--no-diagnostics",
            "--asr-backend", asr, "--translate-backend", mt,
        ]
        if args.context_lines is not None:
            command.extend(["--context-lines", str(args.context_lines)])
        if args.vad_max_utterance_s is not None:
            command.extend(["--vad-max-utterance-s", str(args.vad_max_utterance_s)])
        if args.vad_threshold is not None:
            command.extend(["--vad-threshold", str(args.vad_threshold)])
        plan.append((name, asr, mt, command))
    if args.dry_run:
        for name, _, _, command in plan:
            print(f"{name}: {subprocess.list2cmdline(command)}")
        return 0

    dest.mkdir(parents=True, exist_ok=False)
    summaries = []
    (dest / "manifest.json").write_text(
        json.dumps({
            "clip": str(clip), "clip_bytes": clip.stat().st_size,
            "config": str(config), "context_lines": args.context_lines,
            "vad_max_utterance_s": args.vad_max_utterance_s,
            "vad_threshold": args.vad_threshold,
            "combinations": [{"asr": a, "mt": m} for _, a, m, _ in plan],
            "note": "Operational metrics are NOT accuracy measurements.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Output folder: {dest}", flush=True)
    for name, asr, mt, command in plan:
        print(f"Running {name} (sequentially)...", flush=True)
        started = time.monotonic()
        log_path = dest / f"{name}.log"
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                status = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
        except KeyboardInterrupt:
            print(f"Interrupted. Check {log_path} for partial results.", file=sys.stderr)
            return 130
        elapsed = time.monotonic() - started
        cues = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
        with (dest / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for cue in cues:
                output.write(json.dumps(cue, ensure_ascii=False) + "\n")
        # Prioritize the cues most likely to require manual review.
        with (dest / f"{name}-review.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(
                output, fieldnames=["start_s", "end_s", "duration_s", "review_reason", "ja", "en"]
            )
            writer.writeheader()
            for cue in cues:
                duration = cue["end_s"] - cue["start_s"]
                reasons = []
                if duration >= 14:
                    reasons.append("long_asr_segment")
                if len(cue["en"]) > 160:
                    reasons.append("long_english_translation")
                if reasons:
                    writer.writerow({
                        "start_s": cue["start_s"],
                        "end_s": cue["end_s"],
                        "duration_s": round(duration, 2),
                        "review_reason": ";".join(reasons),
                        "ja": cue["ja"], "en": cue["en"],
                    })
        row = {"run": name, "asr": asr, "translator": mt,
               **summarize(cues, elapsed, status)}
        summaries.append(row)
        print(f"{name}: exit={status}, cues={len(cues)}, wall={elapsed:.1f}s", flush=True)
        if status != 0 and not args.continue_on_error:
            print(f"Run failed. Inspect {log_path}; stopping further tests.", file=sys.stderr)
            break
    (dest / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summaries:
        with (dest / "summary.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"Results: {dest / 'summary.csv'}", flush=True)
    print("Review Japanese audio against a reference; cue counts and speed do not establish accuracy.")
    return 0 if len(summaries) == len(plan) and all(x["exit_code"] == 0 for x in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
