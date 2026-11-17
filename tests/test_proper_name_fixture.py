"""Reviewed source-text decisions and deterministic repairs; no ASR score."""
import json
from pathlib import Path
import unittest

from auto_translate.name_hints import preserve_spoken_names
from auto_translate.proper_names import name_glossary, resolve_names


CASES = json.loads((Path(__file__).parent / "fixtures" / "proper_name_cases.json")
                   .read_text(encoding="utf-8"))["cases"]


class ReviewedProperNameFixtureTests(unittest.TestCase):
    def test_reviewed_occurrences_and_literal_nouns(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                mentions = resolve_names(
                    case["ja"], previous_ja=case.get("previous_ja", ""),
                    video_title=case.get("video_title", ""),
                    confirmed_participants=case.get("confirmed_participants", ()),
                )
                expected = [(row["source"], row["classification"], row.get("target"))
                            for row in case["expected"]]
                self.assertEqual([(m.source, m.classification, m.target) for m in mentions], expected)
                for mention in mentions:
                    self.assertEqual(case["ja"][mention.start:mention.end], mention.source)

    def test_reviewed_english_repairs_preserve_nouns_and_nicknames(self):
        for case in CASES:
            if "expected_english" not in case:
                continue
            with self.subTest(case=case["id"]):
                mentions = resolve_names(
                    case["ja"], previous_ja=case.get("previous_ja", ""),
                    video_title=case.get("video_title", ""),
                    confirmed_participants=case.get("confirmed_participants", ()),
                )
                self.assertEqual(preserve_spoken_names(
                    case["ja"], case["raw_english"], name_glossary(mentions), mentions=mentions,
                ), case["expected_english"])


if __name__ == "__main__":
    unittest.main()
