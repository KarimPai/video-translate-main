"""Compare paired Japanese reference/candidate subtitles without loading models.

CSV columns: reference_ja, output_ja, reference_en, output_en.
Each row must represent the same time-aligned utterance. English comparison is
NOT a semantic translation score: inspect English lines manually.
Usage:
  python scripts/benchmark_quality.py tests/fixtures/quality.csv
"""
import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from auto_translate.translation_guard import suspicious_translation


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def evaluate(rows: list[dict[str, str]]) -> dict:
    reference_chars = 0
    char_errors = 0
    missed = spurious = flags = 0
    for row in rows:
        ref = normalize(row.get("reference_ja") or "")
        out = normalize(row.get("output_ja") or "")
        if ref:
            reference_chars += len(ref)
            char_errors += distance(ref, out)
            missed += not bool(out)
        else:
            spurious += bool(out)
            char_errors += len(out)  # insertions on verified non-speech
        if out and row.get("output_en"):
            flags += bool(suspicious_translation(out, row["output_en"]))
    return {
        "aligned_utterances": len(rows),
        "japanese_character_error_rate": (round(char_errors / reference_chars, 4)
                                          if reference_chars else None),
        "missing_japanese_cues": missed,
        "spurious_japanese_cues": spurious,
        "english_guard_flags_for_review": flags,
        "english_semantic_accuracy": "requires human assessment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"reference_ja", "output_ja", "reference_en", "output_en"}
    if not rows or not required.issubset(rows[0]):
        parser.error("nonempty CSV with columns: " + ", ".join(sorted(required)))
    print(json.dumps(evaluate(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
