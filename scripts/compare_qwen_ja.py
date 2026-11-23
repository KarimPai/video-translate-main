"""Compare current Qwen3-ASR and neosophie Qwen3-ASR-JA on identical holoGTA PCM.

ASR only: no VAD, translator, speaker labels, or production config changes.
Both use the repository's existing llama.cpp Qwen3ASR backend, Q8_0 model
and projector pairs, and the same decode/recovery defaults. Their transcripts
are UNVERIFIED hypotheses, not reference Japanese.

Run:
  uv run python scripts/compare_qwen_ja.py hologta-stress-test.flac
For all 30 reference windows, add --all-windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from compare_haas_qwen_windows import decode_window, run_case, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_WINDOWS = ROOT / "benchmarks" / "reference" / "hologta_30.csv"
JA_MODEL = "Qwen3-ASR-1.7B-JA-Q8_0.gguf"
JA_MMPROJ = "mmproj-Qwen3-ASR-1.7B-JA-Q8_0.gguf"
JA_REPO = "minimalcorp/Qwen3-ASR-1.7B-JA-GGUF"
# Proper nouns, overlaps, numerical negation, and three UNVERIFIED low-activity cuts.
DEFAULT_WINDOWS = (
    "HG02", "HG06", "HG07", "HG10", "HG11", "HG12",
    "HG13", "HG21", "HG26", "HG27", "HG28", "HG29",
)
CSV_FIELDS = (
    "window", "start_s", "end_s", "focus", "current_ja", "qwen_ja_ja",
    "current_elapsed_s", "qwen_ja_elapsed_s", "current_language",
    "qwen_ja_language", "current_rejection", "qwen_ja_rejection",
    "agreement", "needs_review",
)


def load_reference_windows(path: Path = REFERENCE_WINDOWS) -> dict[str, dict]:
    """Fixed time boundaries, not automatic VAD-cue midpoint alignment."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    windows: dict[str, dict] = {}
    for row in rows:
        name = row.get("id", "").strip()
        start, end = float(row["start_s"]), float(row["end_s"])
        if not name or name in windows or not (0 <= start < end <= 1201):
            raise ValueError(f"Invalid or duplicate reference window: {name}")
        windows[name] = {
            "start_s": start, "end_s": end, "focus": row.get("focus", ""),
        }
    if not windows:
        raise ValueError(f"No reference windows in {path}")
    return windows


def normalize_for_agreement(text: str) -> str:
    """NFKC and punctuation/space folding; preserve digits and Latin names."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def agreement(left: str, right: str) -> float | None:
    a, b = normalize_for_agreement(left), normalize_for_agreement(right)
    if not a or not b:
        return None
    return round(SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)


def build_review(rows: list[dict], windows: dict[str, dict]) -> list[dict]:
    by_key = {(r["window"], r["model"]): r for r in rows}
    output = []
    for label, spec in windows.items():
        base = by_key.get((label, "current"))
        ja = by_key.get((label, "qwen-ja"))
        if base is None and ja is None:
            continue
        original = base["text"] if base else ""
        tuned = ja["text"] if ja else ""
        sim = agreement(original, tuned) if base and ja else None
        flag = (
            "model_error" if any(r is not None and r["language"] == "error"
                                for r in (base, ja))
            else "missing_hypothesis" if base is None or ja is None
            else "one_model_empty" if bool(original) != bool(tuned)
            else "both_empty_unverified" if not original and not tuned
            else "high_disagreement" if sim is not None and sim < 0.65
            else "review_names_or_meaning" if any(
                term in spec["focus"].casefold()
                for term in ("proper-name", "money", "number", "crosstalk", "overlap")
            ) else ""
        )
        output.append({
            "window": label, "start_s": spec["start_s"], "end_s": spec["end_s"],
            "focus": spec["focus"], "current_ja": original, "qwen_ja_ja": tuned,
            "current_elapsed_s": base["elapsed_s"] if base else "",
            "qwen_ja_elapsed_s": ja["elapsed_s"] if ja else "",
            "current_language": base["language"] if base else "",
            "qwen_ja_language": ja["language"] if ja else "",
            "current_rejection": base["rejection"] if base else "",
            "qwen_ja_rejection": ja["rejection"] if ja else "",
            "agreement": sim if sim is not None else "",
            "needs_review": flag,
        })
    return output


def report_markdown(review: list[dict], clip_name: str) -> str:
    lines = [
        "# Qwen3-ASR 1.7B vs Qwen3-ASR 1.7B-JA — holoGTA",
        "",
        f"Source: {clip_name}",
        "",
        "**Not an accuracy ranking.** Both hypotheses are unverified. "
        "Agreement is not CER; more characters do not imply better accuracy.",
        "The audio windows, Q8_0 precision, llama.cpp runtime, language "
        "auto-detection and ASR decoder/recovery settings are matched. "
        "The models are loaded sequentially to avoid simultaneous VRAM usage.",
        "Speech presence in low-activity windows is UNVERIFIED.",
        "",
        "| Window | Clip time (s) | Current time (s) | JA time (s) | "
        "Agreement | Review |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in review:
        def elapsed(v: float | str) -> str:
            return f"{v:.3f}" if isinstance(v, (float, int)) else "-"
        lines.append(
            f"| {r['window']} | {r['start_s']:.2f}–{r['end_s']:.2f} | "
            f"{elapsed(r['current_elapsed_s'])} | "
            f"{elapsed(r['qwen_ja_elapsed_s'])} | "
            f"{r['agreement'] if r['agreement'] != '' else '-'} | "
            f"{r['needs_review'] or '-'} |"
        )
    lines += [
        "", "## Interpretation", "",
        "- Listen to the audio before selecting which Japanese transcript is correct.",
        "- Focus on proper names, short replies, crosstalk, payment/negation, and hallucinations.",
        "- Neither this report nor model-model agreement provides a verified CER.",
        "- See review.csv for full Japanese transcripts and their exact timestamps.",
        "- No local config.toml, production VAD, or model selection is modified.",
        "",
    ]
    return "\n".join(lines)


def write_results(dest: Path, rows: list[dict], windows: dict[str, dict],
                  media: Path, model_paths: dict[str, dict[str, str]]) -> None:
    review = build_review(rows, windows)
    with (dest / "hypotheses.jsonl").open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (dest / "review.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(review)
    (dest / "report.md").write_text(
        report_markdown(review, media.name), encoding="utf-8",
    )
    (dest / "manifest.json").write_text(json.dumps({
        "clip": str(media), "clip_bytes": media.stat().st_size,
        "windows": list(windows), "model_paths": model_paths,
        "note": "ASR-only; matching PCM/decoder, but no human Japanese references",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="The existing 20-minute holoGTA audio")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--ja-model-dir", type=Path, default=ROOT / "models" / "qwen3-asr-ja")
    parser.add_argument("--reference-windows", type=Path, default=REFERENCE_WINDOWS)
    parser.add_argument("--window", nargs="+", help="Selected HGxx windows")
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--outdir", type=Path, default=ROOT / "benchmarks" / "runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    media, config_file = args.media.resolve(), args.config.resolve()
    all_windows = load_reference_windows(args.reference_windows)
    if args.window and args.all_windows:
        raise ValueError("Use --window or --all-windows, not both")
    chosen = (
        list(all_windows) if args.all_windows
        else args.window if args.window else list(DEFAULT_WINDOWS)
    )
    unknown = set(chosen) - set(all_windows)
    if unknown:
        raise ValueError(f"Unknown windows: {sorted(unknown)}")
    chosen = list(dict.fromkeys(chosen))
    windows = {key: all_windows[key] for key in chosen}
    ja_dir = args.ja_model_dir.resolve()
    ja_model, ja_mmproj = ja_dir / JA_MODEL, ja_dir / JA_MMPROJ

    if not media.is_file() or not config_file.is_file():
        raise FileNotFoundError("Audio clip or config.toml is missing")
    print("Qwen standard vs Japanese fine-tune; same llama.cpp CUDA runtime.")
    print("Windows:", ", ".join(windows))
    print("JA model:", ja_model)
    print("JA mmproj:", ja_mmproj)
    if args.dry_run:
        return 0
    if not ja_model.is_file() or not ja_mmproj.is_file():
        print(
            "Download BOTH llama.cpp-format JA GGUF files first from "
            f"https://huggingface.co/{JA_REPO} (not the speech.cpp one-file GGUF).",
            file=sys.stderr,
        )
        return 2
    if not shutil.which(args.ffmpeg):
        print(f"ffmpeg not available: {args.ffmpeg}", file=sys.stderr)
        return 2

    from auto_translate.config import get_backend_config, load_config
    from auto_translate.registry import get_default_registry

    config = load_config(config_file)
    standard_cfg = get_backend_config(config, "asr", "qwen3-asr")
    ja_cfg = dict(standard_cfg)
    ja_cfg.update(model_path=str(ja_model), mmproj_path=str(ja_mmproj))
    model_configs = {"current": standard_cfg, "qwen-ja": ja_cfg}
    model_paths = {
        name: {"model_path": str(cfg.get("model_path", "")),
               "mmproj_path": str(cfg.get("mmproj_path", ""))}
        for name, cfg in model_configs.items()
    }

    decoded = {
        name: decode_window(media, spec["start_s"], spec["end_s"], args.ffmpeg)
        for name, spec in windows.items()
    }
    dest = args.outdir.resolve() / (
        "qwen-ja-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    dest.mkdir(parents=True, exist_ok=False)
    print("Output:", dest, flush=True)
    rows: list[dict] = []
    try:
        registry = get_default_registry()
        for label, model_config in model_configs.items():
            model = None
            print("Loading:", label, flush=True)
            try:
                model = registry.create_asr("qwen3-asr", model_config)
                for window, audio in decoded.items():
                    spec = windows[window]
                    record = run_case(
                        model, "qwen3-asr", window, spec["start_s"], audio,
                        variant="full", part=0, left=0, right=len(audio),
                    )
                    record["model"] = label
                    rows.append(record)
                    print(
                        f"{label} {window}: {record['elapsed_s']:.3f}s, "
                        f"{record['ja_chars']} Japanese characters, "
                        f"rejection={record['rejection'] or 'none'}",
                        flush=True,
                    )
            finally:
                if model is not None:
                    model.close()
    finally:
        write_results(dest, rows, windows, media, model_paths)
        print("JSONL:", dest / "hypotheses.jsonl")
        print("CSV:", dest / "review.csv")
        print("Report:", dest / "report.md")
    return 1 if any(r["language"] == "error" for r in rows) or len(rows) != 2 * len(windows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
