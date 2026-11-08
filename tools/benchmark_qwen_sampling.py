"""Repeatable Qwen sampling comparison; text-only, no ASR or GPU test fixture.

Usage:
  uv run --no-sync python tools/benchmark_qwen_sampling.py
  uv run --no-sync python tools/benchmark_qwen_sampling.py --samples my_ja_en.jsonl --rounds 2

One managed Qwen3.5 server is loaded for the entire experiment. Every
temperature sees the exact same Japanese cues, optional Japanese-only
context, title, glossary and faithful translation prompt. Quality rechecks
are disabled for the benchmark so temperature is the only varied parameter.
Text similarity is NOT ground-truth translation accuracy or ASR accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from auto_translate.config import load_config
from auto_translate.glossary import Glossary
from auto_translate.translate.qwen import QwenTranslator, sampling_parameters


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "tests" / "data" / "qwen_sampling_examples.jsonl"
DEFAULT_TEMPERATURES = (0.1, 0.35, 0.5, 0.7)


@dataclass(frozen=True)
class Example:
    ja: str
    reference_en: str
    title: str = ""
    history_ja: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    must_exclude: tuple[str, ...] = ()


def load_examples(path: Path) -> list[Example]:
    """Require explicit bilingual references, never infer from ASR output."""
    examples: list[Example] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("each line must contain one JSON object")
            ja, en = payload["ja"], payload["reference_en"]
            if not isinstance(ja, str) or not ja.strip():
                raise ValueError("ja must be a nonempty string")
            if not isinstance(en, str) or not en.strip():
                raise ValueError("reference_en must be a nonempty string")
            title = payload.get("title", "")
            if not isinstance(title, str):
                raise ValueError("title must be a string")
            fields: dict[str, tuple[str, ...]] = {}
            for key in ("history_ja", "must_include", "must_exclude"):
                values = payload.get(key, [])
                if (not isinstance(values, list)
                        or not all(isinstance(value, str) and value.strip()
                                   for value in values)):
                    raise ValueError(f"{key} must be a list of nonempty strings")
                fields[key] = tuple(values)
            examples.append(Example(ja.strip(), en.strip(), title=title, **fields))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid benchmark example: {exc}") from exc
    if not examples:
        raise ValueError(f"No usable examples in {path}")
    return examples


def assess(example: Example, translated: str) -> dict:
    """Provide descriptive, inspectable metrics; never claim semantic WER."""
    expected = " ".join(example.reference_en.casefold().split())
    actual = " ".join(translated.casefold().split())
    similarity = round(SequenceMatcher(None, expected, actual).ratio(), 4)
    includes = [needle for needle in example.must_include
                if needle.casefold() not in actual]
    excludes = [needle for needle in example.must_exclude
                if needle.casefold() in actual]
    return {
        "reference_char_similarity": similarity,
        "constraints_checked": len(example.must_include) + len(example.must_exclude),
        "constraints_failed": len(includes) + len(excludes),
        "missing_terms": " | ".join(includes),
        "unexpected_terms": " | ".join(excludes),
    }


def run_comparison(
    translator: QwenTranslator,
    examples: Sequence[Example],
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    rounds: int = 1,
) -> list[dict]:
    """Run each input independently to avoid cross-temperature context leaks."""
    if not examples or not temperatures or rounds < 1:
        raise ValueError("Need examples, temperatures and rounds >= 1")
    previous_temperature = translator.cfg.temperature
    previous_recheck = translator.cfg.quality_recheck_enabled
    result: list[dict] = []
    try:
        translator.cfg.quality_recheck_enabled = False
        for temperature in temperatures:
            translator.cfg.temperature = float(temperature)
            parameters = sampling_parameters(translator.cfg)
            for trial in range(1, rounds + 1):
                for index, example in enumerate(examples, 1):
                    translator.reset_scene_context()
                    translator.vod_title = example.title[:200]
                    # No previous English guesses enter Qwen's prompt.
                    history = [(ja, "") for ja in example.history_ja]
                    started = time.perf_counter()
                    error = ""
                    try:
                        translation = translator.translate(example.ja, history)
                    except Exception as exc:
                        translation = "[translation failed]"
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - started
                    metrics = assess(example, translation)
                    row = {
                        "temperature": float(temperature),
                        "seed": parameters.get("seed", "random"),
                        "top_p": parameters["top_p"], "top_k": parameters["top_k"],
                        "min_p": parameters["min_p"],
                        "repeat_penalty": parameters["repeat_penalty"],
                        "presence_penalty": parameters["presence_penalty"],
                        "style": translator.cfg.subtitle_style,
                        "trial": trial, "example": index,
                        "title": example.title,
                        "history_ja": " / ".join(example.history_ja),
                        "ja": example.ja,
                        "reference_en": example.reference_en,
                        "translated_en": translation, "latency_s": round(elapsed, 4),
                        "failed": int(translation == "[translation failed]"),
                        "error": error,
                        **metrics,
                    }
                    result.append(row)
                    print(
                        f"temp={temperature:g} trial={trial} "
                        f"{index}/{len(examples)} {elapsed:.2f}s | "
                        f"JA: {example.ja[:40]} | EN: {translation[:85]}",
                        flush=True,
                    )
    finally:
        translator.cfg.temperature = previous_temperature
        translator.cfg.quality_recheck_enabled = previous_recheck
        translator.reset_scene_context()
    return result


def summarize(rows: Sequence[dict]) -> list[dict]:
    """Aggregate by temperature without treating string similarity as truth."""
    groups: dict[float, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["temperature"], []).append(row)
    summaries = []
    for temperature, group in sorted(groups.items()):
        total_checks = sum(row["constraints_checked"] for row in group)
        failed_checks = sum(row["constraints_failed"] for row in group)
        latencies = [row["latency_s"] for row in group]
        summaries.append({
            "temperature": temperature,
            "lines": len(group),
            "mean_latency_s": round(statistics.mean(latencies), 3),
            "median_latency_s": round(statistics.median(latencies), 3),
            "mean_reference_char_similarity": round(statistics.mean(
                row["reference_char_similarity"] for row in group
            ), 3),
            "term_checks_passed": total_checks - failed_checks,
            "term_checks_total": total_checks,
            "translation_failures": sum(row["failed"] for row in group),
        })
    return summaries


def write_results(path: Path, rows: list[dict], summary: list[dict]) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = path.with_name(path.stem + "-summary.csv")
    for target, records in ((path, rows), (summary_path, summary)):
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--samples", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Set the same llama.cpp seed on every request")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "logs" / ("qwen-sampling-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
    ))
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.seed < 0:
        parser.error("--rounds must be >= 1 and --seed must be >= 0")
    if not args.temperatures or any(not 0.0 <= t <= 2.0 for t in args.temperatures):
        parser.error("All temperatures must be within [0, 2]")
    examples = load_examples(args.samples)
    cfg = load_config(args.config).translate
    cfg.subtitle_style = "faithful"
    cfg.quality_recheck_enabled = False
    cfg.sampling_seed = args.seed
    glossary_path = Path(cfg.glossary)
    if not glossary_path.is_absolute():
        glossary_path = ROOT / glossary_path
    glossary = Glossary.load(glossary_path)
    # Validate before the expensive Qwen server initialization.
    sampling_parameters(cfg)
    print(
        f"Text-only Qwen sampling benchmark: {len(examples)} input lines; "
        f"temps={args.temperatures}; rounds={args.rounds}; seed={args.seed}; "
        f"other sampling settings fixed. No ASR or MPV.",
        flush=True,
    )
    translator = QwenTranslator(cfg, glossary, vod_title="")
    try:
        rows = run_comparison(translator, examples, args.temperatures, args.rounds)
    finally:
        translator.close()
    summary = summarize(rows)
    detailed, summarized = write_results(args.output, rows, summary)
    print("\nSummary (reference similarity is NOT semantic accuracy):")
    for group in summary:
        print(
            f"temp={group['temperature']:g}: "
            f"mean={group['mean_latency_s']:.3f}s, "
            f"reference similarity={group['mean_reference_char_similarity']:.3f}, "
            f"term checks={group['term_checks_passed']}/{group['term_checks_total']}, "
            f"failures={group['translation_failures']}/{group['lines']}"
        )
    print(f"\nDetailed translations: {detailed}\nSummary: {summarized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
