"""Offline report and bounded benchmark checks; no models or GPU required."""
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.quality_report import (
    build_report, evaluate, generate_report, load_rows, recommend_presets,
    rows_from_comparisons, write_report,
)


class QualityReportTests(unittest.TestCase):
    def test_corpus_cer_includes_silence_insertions_and_nfkc(self):
        score = evaluate([
            {"reference_ja": "Ａ 猫", "output_ja": "A犬"},
            {"reference_ja": "", "output_ja": "音"},
            {"reference_ja": "話", "output_ja": ""},
            {"output_ja": "unannotated"},
        ])
        self.assertEqual(score["annotated_utterances"], 3)
        self.assertEqual(score["reference_characters"], 3)
        self.assertEqual(score["character_errors"], 3)
        self.assertEqual(score["japanese_character_error_rate"], 1.)
        self.assertEqual(score["missing_japanese_cues"], 1)
        self.assertEqual(score["spurious_japanese_cues"], 1)

    def test_missing_references_and_measurements_are_unavailable(self):
        report = build_report([{"output_ja": "日本語"}])
        self.assertIsNone(report["quality"]["japanese_character_error_rate"])
        for metric in report["metrics"].values():
            self.assertFalse(metric["available"])
            self.assertTrue(metric["unavailable_reason"])
        self.assertTrue(any("manual Japanese references" in s for s in report["recommendations"]))

    def test_tags_and_human_errors_are_not_semantic_scores(self):
        report = build_report([
            {"reference_ja": "名前", "output_ja": "名", "tags": "names|numbers", "names_error": "true"},
            {"reference_ja": "ない", "output_ja": "ない", "tags": ["Negation"]},
        ])
        self.assertEqual(report["per_tag"]["names"]["character_errors"], 1)
        self.assertEqual(report["error_groups"]["names"]["human_reviewed_errors"]["count"], 1)
        self.assertEqual(report["error_groups"]["negation"]["japanese"]["character_errors"], 0)
        self.assertFalse(report["error_groups"]["negation"]["human_reviewed_errors"]["available"])

    def test_percentiles_weighted_rtf_and_outcome_denominators(self):
        report = build_report([
            {"cue_latency_s": 1, "elapsed_s": 2, "audio_duration_s": 1,
             "fallback": True, "timeout": True, "gpu_free_gb": .4, "gpu_used_gb": 7.6},
            {"cue_latency_s": 3, "elapsed_s": 3, "audio_duration_s": 3,
             "fallback": False, "timeout": False},
            {},  # unknown outcomes must not be counted as successes
        ])
        metrics = report["metrics"]
        self.assertEqual(metrics["cue_latency_s"]["p50"], 2)
        self.assertAlmostEqual(metrics["cue_latency_s"]["p95"], 2.9)
        self.assertEqual(metrics["real_time_factor"]["aggregate"], 1.25)
        self.assertEqual(metrics["timeout_rate"]["observed_cues"], 2)
        self.assertEqual(metrics["timeout_rate"]["rate"], .5)
        self.assertTrue(any("0.55 GB" in s for s in report["recommendations"]))

    def test_nonfinite_negative_boolean_metrics_are_rejected(self):
        rows = [{"cue_latency_s": value} for value in ("nan", "inf", -1, True, None)]
        self.assertFalse(build_report(rows)["metrics"]["cue_latency_s"]["available"])

    def test_timing_checkpoints_and_no_stage_sum_latency(self):
        report = build_report([
            {"timings": {"audio_read": 100, "mt_end": 103}},
            {"asr_elapsed_s": 1, "mt_elapsed_s": 2},
        ])
        self.assertEqual(report["metrics"]["cue_latency_s"]["samples"], 1)
        self.assertEqual(report["metrics"]["cue_latency_s"]["p50"], 3)

    def test_stage_merge_once_per_session_generation_cue_and_final_text_wins(self):
        cue = dict(session_id="one", generation=0, start_s=0, end_s=2)
        rows = rows_from_comparisons([
            dict(cue, stage="reconciliation", chosen="初", fallback_reason=None),
            dict(cue, stage="translation", final_ja="最終", final_en="Final", cue_latency_s=2),
            dict(cue, stage="asr_fallback", haas="旧", fallback_reason="TimeoutError"),
            dict(cue, session_id="two", stage="translation", final_ja="別"),
            dict(cue, generation=1, stage="translation", final_ja="再生"),
        ])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["output_ja"], "最終")
        self.assertEqual(rows[0]["output_en"], "Final")
        metrics = build_report(rows)["metrics"]
        self.assertEqual(metrics["fallback_rate"]["count"], 1)
        self.assertEqual(metrics["timeout_rate"]["count"], 1)

    def test_deadline_or_backlog_is_not_invented_timeout(self):
        report = build_report([{"fallback_reason": "deadline_generation_or_backlog"}])
        self.assertEqual(report["metrics"]["fallback_rate"]["count"], 1)
        self.assertEqual(report["metrics"]["timeout_rate"]["count"], 0)

    def test_optional_jiwer_missing_is_explicit(self):
        with patch.dict(sys.modules, {"jiwer": None}):
            score = evaluate([{"reference_ja": "猫", "output_ja": "猫"}], use_jiwer=True)
        self.assertEqual(score["japanese_character_error_rate"], 0)
        self.assertFalse(score["jiwer"]["available"])

    def test_jsonl_bad_tail_and_annotation_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            cue = dict(session_id="s", generation=1, start_s=1, end_s=2)
            path.write_text(json.dumps(dict(cue, stage="translation", final_ja="猫")) + '\n{"cut":', encoding="utf8")
            annotations = Path(folder) / "annotations.csv"
            annotations.write_text("session_id,generation,start_s,end_s,reference_ja,tags\ns,1,1,2,犬,names\n", encoding="utf8")
            result = generate_report(path, annotations_path=annotations)
            self.assertEqual(result["report"]["invalid_log_lines"], 1)
            self.assertEqual(result["report"]["attached_annotations"], 1)
            self.assertEqual(result["report"]["quality"]["character_errors"], 1)
            self.assertTrue(result["html_path"].is_file())
            self.assertTrue(result["json_path"].is_file())

    def test_benchmark_json_load_and_html_escapes_untrusted_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            path.write_text(json.dumps({"clips": [{"reference_ja": "猫", "output_ja": "犬"}]}), encoding="utf8")
            rows, invalid = load_rows(path)
            self.assertEqual(invalid, 0)
            report = build_report(rows, source="<script>alert(1)</script>")
            result = write_report(report, Path(folder))
            content = result["html_path"].read_text(encoding="utf8")
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)

    def test_preset_comparison_requires_complete_matching_workload_and_scope(self):
        reports = []
        for beam, output, elapsed in ((1, "犬", 1), (3, "猫", 2), (6, "猫", 3)):
            report = build_report([{"reference_ja": "猫", "output_ja": output,
                                    "elapsed_s": elapsed, "audio_duration_s": 4}])
            report.update(settings={"beam": beam}, complete=True, workload=["same-sha"],
                          hardware={"gpu": "measured"}, measurement_scope="ASR only")
            reports.append(report)
        self.assertIn('"beam": 3', recommend_presets(reports)[0])
        reports[1]["complete"] = False
        reports[2]["measurement_scope"] = "full pipeline"
        self.assertIn("Fewer than two", recommend_presets(reports)[0])


class ASRBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("benchmark_dual_test", scripts / "benchmark_dual.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.remove(str(scripts))

    def test_sequential_pair_preserves_both_outputs_without_overlap(self):
        calls = []

        class Model:
            def __init__(self, name):
                self.name = name

            def transcribe(self, audio, timeout):
                calls.append((self.name, timeout))
                return self.name

        result = self.module.benchmark_pair(Model("H"), Model("W"), [0.], schedule="sequential", timeout_s=2)
        self.assertEqual(calls, [("H", 2), ("W", 2)])
        self.assertEqual(result["haas"], "H")
        self.assertEqual(result["whisper_turbo"], "W")

    def test_concurrent_pair_is_exactly_two_single_flight_calls(self):
        barrier = threading.Barrier(2, timeout=2)
        calls = []

        class Model:
            def transcribe(self, audio, timeout):
                calls.append(timeout)
                barrier.wait()
                return "日本語"

        result = self.module.benchmark_pair(Model(), Model(), [0.], schedule="concurrent", timeout_s=2)
        self.assertEqual(calls, [2, 2])
        self.assertEqual(result["haas"], result["whisper_turbo"])
        self.assertFalse(result["timeout"])

    def test_decode_failure_is_not_scored_as_a_success(self):
        class Broken:
            def transcribe(self, audio, timeout):
                raise TimeoutError("hardware timeout")

        with self.assertRaisesRegex(RuntimeError, "TimeoutError"):
            self.module.benchmark_pair(Broken(), Broken(), [0.], schedule="sequential", timeout_s=2)


if __name__ == "__main__":
    unittest.main()
