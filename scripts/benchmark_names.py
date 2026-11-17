"""Measure deterministic name decisions against reviewed Japanese text fixtures.

This measures authored source-text decisions and English repairs, not acoustic
ASR accuracy, model translation accuracy, or a representative real-world rate.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from auto_translate.name_hints import preserve_spoken_names
from auto_translate.proper_names import COMMON_NOUN, PERSON, name_glossary, resolve_names

_EXPANSIONS = {
    "Matsuri": "Natsuiro Matsuri", "Miko": "Sakura Miko", "Mikochi": "Sakura Miko",
    "Botan": "Shishiro Botan", "Shishiron": "Shishiro Botan",
    "Watame": "Tsunomaki Watame", "Watamee": "Tsunomaki Watame",
    "Pekora": "Usada Pekora", "Pekoora": "Usada Pekora", "Peko-chan": "Usada Pekora",
    "Sui-chan": "Hoshimachi Suisei", "Lap": "Laplus",
    "Fuwawa": "Fuwawa Abyssgard", "Mococo": "Mococo Abyssgard",
    "Moona": "Moona Hoshinova", "Kaela": "Kaela Kovalskia", "Iofi": "Airani Iofifteen",
}


def evaluate(cases: list[dict]) -> dict:
    exact = repairs = correct_repairs = names = literal = noun_regressions = 0
    nickname_expansions = 0
    failures = []
    for case in cases:
        mentions = resolve_names(case["ja"], previous_ja=case.get("previous_ja", ""),
                                 video_title=case.get("video_title", ""),
                                 confirmed_participants=case.get("confirmed_participants", ()))
        expected = [(row["source"], row["classification"], row.get("target"))
                    for row in case["expected"]]
        actual = [(row.source, row.classification, row.target) for row in mentions]
        exact += actual == expected
        if actual != expected:
            failures.append({"case": case["id"], "expected": expected, "actual": actual})
        for source, label, target in expected:
            names += label == PERSON
            literal += label == COMMON_NOUN
            if label == COMMON_NOUN and any(row.source == source and row.classification == PERSON
                                           for row in mentions):
                # Mixed uses share a surface; compare counts to avoid counting
                # the expected person occurrence as a literal regression.
                expected_people = sum(s == source and kind == PERSON for s, kind, _ in expected)
                actual_people = sum(row.source == source and row.classification == PERSON for row in mentions)
                noun_regressions += actual_people > expected_people
        if "expected_english" in case:
            repairs += 1
            english = preserve_spoken_names(case["ja"], case["raw_english"],
                                            name_glossary(mentions), mentions=mentions)
            correct_repairs += english == case["expected_english"]
            if english != case["expected_english"]:
                failures.append({"case": case["id"], "expected_english": case["expected_english"],
                                 "actual_english": english})
            # Count explicit forbidden full-name spans, even when the short
            # target is a substring of that span (Matsuri in Natsuiro Matsuri).
            for _, label, target in expected:
                if label != PERSON or not target:
                    continue
                short = re.sub(r"-(?:chan|san|sama|senpai|kun)$", "", target)
                expanded = _EXPANSIONS.get(short)
                if (expanded and expanded not in case["expected_english"]
                        and re.search(r"(?<![\w-])" + re.escape(expanded) + r"(?![\w-])", english, re.I)):
                    nickname_expansions += 1
    return {"evidence": "authored Japanese source-text fixtures; no recorded audio or model inference",
            "cases": len(cases), "exact_occurrence_cases": exact,
            "exact_occurrence_case_rate": exact / len(cases) if cases else None,
            "expected_person_occurrences": names, "expected_literal_occurrences": literal,
            "literal_noun_to_name_regressions": noun_regressions,
            "english_repair_cases": repairs, "exact_english_repairs": correct_repairs,
            "invalid_nickname_expansions": nickname_expansions,
            "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path,
                        default=ROOT / "tests" / "fixtures" / "proper_name_cases.json")
    args = parser.parse_args()
    result = evaluate(json.loads(args.fixture.read_text(encoding="utf-8"))["cases"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
