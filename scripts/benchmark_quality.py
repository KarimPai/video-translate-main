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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from auto_translate.quality_report import distance, evaluate, generate_report, normalize  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path, help="JSON/HTML folder (default: input folder/reports)")
    parser.add_argument("--jiwer", action="store_true", help="also score using optional jiwer if installed")
    parser.add_argument("--hardware-label", help="hardware/configuration label recorded in this report")
    args = parser.parse_args()
    with args.csv_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"reference_ja", "output_ja", "reference_en", "output_en"}
    if not rows or not required.issubset(rows[0]):
        parser.error("nonempty CSV with columns: " + ", ".join(sorted(required)))
    result = generate_report(args.csv_file, args.output, hardware_label=args.hardware_label,
                             use_jiwer=args.jiwer)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"HTML report: {result['html_path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
