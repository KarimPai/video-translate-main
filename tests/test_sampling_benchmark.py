"""Sampling benchmark runs without an LLM, GPU, ASR or media downloads."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.benchmark_qwen_sampling import (
    DEFAULT_DATA, DEFAULT_TEMPERATURES, Example, assess, load_examples,
    run_comparison, summarize, write_results,
)


class _FakeTranslator:
    def __init__(self):
        self.cfg = SimpleNamespace(
            temperature=0.35, quality_recheck_enabled=True,
        )
        self.vod_title = ""
        self.calls = []
        self.resets = 0

    def reset_scene_context(self):
        self.resets += 1

    def translate(self, ja, history):
        self.calls.append(
            (self.cfg.temperature, self.cfg.quality_recheck_enabled,
             ja, self.vod_title, list(history))
        )
        if ja == "raise!":
            raise RuntimeError("simulated model failure")
        return "Watame, come over here."


class SamplingBenchmarkTests(unittest.TestCase):
    def test_curated_fixture_has_audio_independent_japanese_examples(self):
        examples = load_examples(DEFAULT_DATA)
        self.assertGreaterEqual(len(examples), 30)
        self.assertGreaterEqual(len({example.title for example in examples}), 3)
        self.assertTrue(all(example.ja and example.reference_en for example in examples))
        self.assertEqual(DEFAULT_TEMPERATURES, (0.1, 0.35, 0.5, 0.7))

    def test_loader_uses_only_explicit_references(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "examples.jsonl"
            path.write_text(
                json.dumps({
                    "ja": "わため大丈夫?", "reference_en": "Are you okay, Watame?",
                    "title": "Hololive GTA", "history_ja": ["車が来た"],
                    "must_include": ["Watame"],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            examples = load_examples(path)
            self.assertEqual(examples[0].history_ja, ("車が来た",))
            self.assertEqual(examples[0].must_include, ("Watame",))
            path.write_text('{"ja": "わため大丈夫?"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference_en"):
                load_examples(path)

    def test_same_japanese_context_each_temperature_without_recheck(self):
        fake = _FakeTranslator()
        examples = [
            Example("こんにちは", "Hello.", title="Greeting",
                    history_ja=("おはよう",), must_include=("Watame",)),
            Example("わため大丈夫?", "Watame, are you okay?", title="Hololive"),
        ]
        rows = run_comparison(fake, examples, (0.1, 0.35, 0.5, 0.7))
        self.assertEqual(len(rows), 8)
        self.assertEqual(fake.resets, 9)  # each sample + final cleanup
        self.assertEqual(
            [record[0] for record in fake.calls],
            [0.1] * 2 + [0.35] * 2 + [0.5] * 2 + [0.7] * 2,
        )
        self.assertTrue(all(not record[1] for record in fake.calls))
        self.assertEqual(fake.calls[0][4], [("おはよう", "")])
        self.assertEqual(fake.calls[2][4], [("おはよう", "")])
        self.assertEqual(fake.cfg.temperature, 0.35)
        self.assertTrue(fake.cfg.quality_recheck_enabled)

    def test_failed_translation_is_captured_but_other_samples_continue(self):
        fake = _FakeTranslator()
        rows = run_comparison(fake, [
            Example("raise!", "expected"),
            Example("わため", "Watame"),
        ], temperatures=[0.1])
        self.assertEqual([row["failed"] for row in rows], [1, 0])
        self.assertIn("simulated model failure", rows[0]["error"])

    def test_constraint_scoring_does_not_claim_semantic_accuracy(self):
        case = Example(
            "わため大丈夫?", "Watame, are you okay?",
            must_include=("Watame",), must_exclude=("Pekora",),
        )
        good = assess(case, "Are you all right, Watame?")
        self.assertEqual(good["constraints_checked"], 2)
        self.assertEqual(good["constraints_failed"], 0)
        self.assertLess(good["reference_char_similarity"], 1.0)
        bad = assess(case, "Pekora, are you okay?")
        self.assertEqual(bad["constraints_failed"], 2)
        self.assertEqual(bad["missing_terms"], "Watame")
        self.assertEqual(bad["unexpected_terms"], "Pekora")

    def test_summaries_and_csv_are_written(self):
        fake = _FakeTranslator()
        rows = run_comparison(fake, [
            Example("わため", "Watame", must_include=("Watame",)),
        ], temperatures=[0.1, 0.35])
        summary = summarize(rows)
        self.assertEqual([x["temperature"] for x in summary], [0.1, 0.35])
        self.assertEqual([x["term_checks_passed"] for x in summary], [1, 1])
        with tempfile.TemporaryDirectory() as d:
            output, summary_file = write_results(Path(d) / "sampling.csv", rows, summary)
            with output.open(encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            with summary_file.open(encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)


if __name__ == "__main__":
    unittest.main()
