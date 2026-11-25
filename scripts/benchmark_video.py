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
import threading
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


def read_transcripts(path: Path, column: str = "haas_ja") -> list[dict]:
    """Read reviewed fixed Japanese transcripts for translator-only comparisons."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "start_s", "end_s", column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Transcript CSV is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, 2):
            japanese = (row.get(column) or "").strip()
            if not japanese:
                continue
            try:
                start_s, end_s = float(row["start_s"]), float(row["end_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timestamps in transcript CSV line {line}") from exc
            if start_s < 0 or end_s < start_s:
                raise ValueError(f"Invalid timestamp range in transcript CSV line {line}")
            rows.append({
                "id": row["id"],
                "start_s": start_s,
                "end_s": end_s,
                "focus": row.get("focus", ""),
                "ja": japanese,
            })
    if not rows:
        raise ValueError(f"No nonempty transcripts found in {path} column {column!r}")
    return rows


class GpuMemorySampler:
    """Sample total GPU memory while one translation backend is loaded."""

    def __init__(self, interval_s: float = 0.15) -> None:
        self.interval_s = interval_s
        self.baseline_mib = self._read_memory()
        self.peak_mib = self.baseline_mib
        self._stop = threading.Event()
        self._thread = None
        if self.baseline_mib is not None:
            self._thread = threading.Thread(target=self._sample, name="benchmark-vram", daemon=True)
            self._thread.start()

    @staticmethod
    def _read_memory() -> int | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return int(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            value = self._read_memory()
            if value is not None:
                self.peak_mib = max(self.peak_mib or value, value)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)


def run_transcript_benchmark(args: argparse.Namespace) -> int:
    """Compare translators on the same saved Japanese transcripts, without ASR."""
    from auto_translate.config import get_backend_config, load_config
    from auto_translate.glossary import Glossary
    from auto_translate.model_discovery import resolve_selected_model_paths
    from auto_translate.registry import (
        get_default_registry,
        resolve_local_plamo_model_path,
    )
    from auto_translate.stream_context import StreamContext
    from auto_translate.subtitle_clean import clean_english_output, english_output_issue

    transcript_path = args.transcripts.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not transcript_path.is_file():
        print(f"Transcript CSV not found: {transcript_path}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    try:
        rows = read_transcripts(transcript_path, args.transcript_column)
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"Cannot load transcript benchmark inputs: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Transcript-only benchmark: {len(rows)} inputs from {transcript_path}")
        print(f"Translators, loaded sequentially: {', '.join(args.mt)}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = args.outdir.expanduser().resolve() / stamp
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {
        "benchmark_kind": "fixed_transcript_mt_only",
        "transcripts": str(transcript_path),
        "transcript_column": args.transcript_column,
        "input_count": len(rows),
        "config": str(config_path),
        "stream_title": args.stream_title,
        "context_lines": args.context_lines,
        "context_english": False,
        "models_loaded_sequentially": True,
        "asr_started": False,
        "gpu_memory_sample_interval_s": 0.15,
        "quality_note": "No human-verified English references are included; operational metrics are not accuracy scores.",
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    glossary_path = Path(config.translate.glossary)
    if not glossary_path.is_absolute():
        glossary_path = ROOT / glossary_path
    glossary = Glossary.load(glossary_path).for_stream(args.stream_title)
    stream = StreamContext(title=args.stream_title)
    registry = get_default_registry()
    summaries = []
    for number, backend in enumerate(args.mt, 1):
        name = f"{number:02d}-" + re.sub(r"[^a-zA-Z0-9_-]", "_", backend)
        profile = get_backend_config(config, "translate", backend, selected_backend=backend)
        if args.context_lines is not None:
            profile["context_lines"] = args.context_lines
        # Compare Japanese-only conversational history so model output from a
        # preceding cue cannot make the next input differ between backends.
        profile["context_english"] = False
        profile["parallel_slots"] = 1
        profile["gpu_layers"] = -1
        if backend in ("plamo", "plamo8b"):
            resolved = resolve_local_plamo_model_path(backend, profile.get("model_path", ""))
            if resolved:
                profile["model_path"] = resolved
        profile = resolve_selected_model_paths("translate", backend, profile, root=ROOT)

        sampler = GpuMemorySampler()
        translator = None
        started = time.perf_counter()
        load_s = wall_s = 0.0
        cues, outputs = [], []
        issues: dict[str, int] = {}
        expected_names = preserved_names = 0
        try:
            translator = registry.create_backend(
                "translate", backend, profile, glossary=glossary, stream=stream,
                gpu_layers=-1,
            )
            load_s = time.perf_counter() - started
            history = []
            previous_end = None
            translation_started = time.perf_counter()
            for row in rows:
                if (previous_end is not None
                        and row["start_s"] - previous_end > profile["context_max_gap_s"]):
                    history.clear()
                set_position = getattr(translator, "set_position", None)
                if callable(set_position):
                    set_position(row["start_s"])
                source_context = " ".join(japanese for japanese, _ in history)
                names = glossary.name_terms(row["ja"], context=source_context)
                cue_started = time.perf_counter()
                output = translator.translate(row["ja"], history)
                latency = time.perf_counter() - cue_started
                output = clean_english_output(
                    output,
                    remove_fillers=profile["remove_fillers"],
                    collapse_repetitions=profile["collapse_repetitions"],
                    allow_japanese=not profile["suppress_japanese_in_english"],
                    source_ja=row["ja"],
                )
                issue = english_output_issue(
                    output, allow_japanese=not profile["suppress_japanese_in_english"]
                )
                if issue:
                    issues[issue] = issues.get(issue, 0) + 1
                expected_names += len(names)
                preserved_names += sum(target.casefold() in output.casefold()
                                       for _, target in names)
                cue = {
                    **row, "en": output, "vad_s": 0.0, "wait_s": 0.0,
                    "asr_s": 0.0, "mt_s": latency,
                }
                cues.append(cue)
                outputs.append({
                    **row, "en": output, "mt_s": round(latency, 3),
                    "expected_names": [target for _, target in names],
                    "output_issue": issue,
                })
                # Empty English history keeps the Japanese context identical for
                # the next cue in both model runs.
                history.append((row["ja"], ""))
                previous_end = row["end_s"]
            wall_s = time.perf_counter() - translation_started
            status = 0
        except KeyboardInterrupt:
            status = 130
            print(f"Interrupted while running {backend}; closing its server.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            status = 1
            load_s = time.perf_counter() - started
            issues = {"benchmark_error": 1}
            print(f"{backend} benchmark failed: {exc}", file=sys.stderr)
        finally:
            try:
                if translator is not None:
                    translator.close()
            finally:
                time.sleep(0.3)
                sampler.close()

        with (dest / f"{name}.jsonl").open("w", encoding="utf-8") as output_file:
            for cue in outputs:
                output_file.write(json.dumps(cue, ensure_ascii=False) + "\n")
        summary = {
            "run": name,
            "translator": backend,
            "model_path": profile.get("model_path", ""),
            "seed": profile.get("seed"),
            "exit_code": status,
            "load_s": round(load_s, 3),
            "baseline_vram_mib": sampler.baseline_mib,
            "peak_vram_mib": sampler.peak_mib,
            "peak_vram_delta_mib": (
                max(0, sampler.peak_mib - sampler.baseline_mib)
                if sampler.peak_mib is not None and sampler.baseline_mib is not None else None
            ),
            "glossary_name_mentions": expected_names,
            "glossary_names_preserved_exact": preserved_names,
            "glossary_name_preservation_rate": (
                round(preserved_names / expected_names, 4) if expected_names else None
            ),
            "output_issue_counts": issues,
            **summarize(cues, wall_s, status),
        }
        summary["mean_asr_s"] = None
        summary["p95_asr_s"] = None
        summaries.append(summary)
        print(f"{backend}: exit={status}, cues={len(cues)}, wall={wall_s:.1f}s, "
              f"peak VRAM={summary['peak_vram_mib']} MiB", flush=True)
        if status == 130:
            break
        if status != 0 and not args.continue_on_error:
            break

    (dest / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summaries:
        with (dest / "summary.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"Results: {dest / 'summary.csv'}")
    print("Human translation accuracy and hallucinations require reviewed reference translations.")
    return 0 if len(summaries) == len(args.mt) and all(x["exit_code"] == 0 for x in summaries) else 1


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential ASR/MT comparisons on local media or fixed Japanese transcripts."
    )
    parser.add_argument("clip", type=Path, nargs="?",
                        help="local media file, e.g. hologta-stress-test.webm")
    parser.add_argument("--transcripts", type=Path,
                        help="CSV with fixed Japanese transcripts; skips ASR and compares MT only")
    parser.add_argument("--transcript-column", default="haas_ja",
                        help="Japanese transcript column when using --transcripts")
    parser.add_argument("--stream-title", default="",
                        help="optional stream context for glossary and translation prompts")
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
    if args.transcripts:
        if args.clip:
            print("Use either a media clip or --transcripts, not both.", file=sys.stderr)
            return 2
        return run_transcript_benchmark(args)
    if args.clip is None:
        print("Provide a local media clip or --transcripts CSV.", file=sys.stderr)
        return 2
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
            "--no-web", "--no-diagnostics",
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
