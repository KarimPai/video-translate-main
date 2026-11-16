"""Name-resolution evidence follows its cue without replacing ASR diagnostics."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.pipeline import run_translate
from auto_translate.review import TranscriptReview
from auto_translate.types import Transcript


class NameReviewDiagnosticsTests(unittest.TestCase):
    def test_unresolved_name_is_reviewable_before_or_after_subtitle(self):
        for first in ("diagnostic", "subtitle"):
            with self.subTest(first=first):
                review = TranscriptReview()
                subtitle = dict(start_s=1, end_s=2, ja="みこが来た", en="Someone arrived.")
                diagnostic = dict(start_s=1, end_s=2, stage="translation",
                                  translation_flags=["proper_name_mismatch"])
                if first == "diagnostic":
                    review.add_diagnostic(diagnostic)
                    key = review.add(subtitle)
                else:
                    key = review.add(subtitle)
                    review.add_diagnostic(diagnostic)
                self.assertTrue(review.cues[key].needs_review)
                self.assertEqual(len(review.filtered(needs_review=True)), 1)

    def test_resolved_name_is_evidence_without_a_review_warning(self):
        review = TranscriptReview()
        key = review.add(dict(start_s=1, end_s=2, ja="巫女が来た", en="Miko arrived."))
        review.add_diagnostic(dict(start_s=1, end_s=2, stage="translation",
                                   name_resolutions=[dict(source="巫女", target="Miko")],
                                   translation_flags=[]))
        self.assertFalse(review.cues[key].needs_review)
        self.assertIn("Miko", str(review.cues[key].diagnostic["name_resolutions"]))

    def test_late_translation_preserves_asr_flags(self):
        review = TranscriptReview()
        key = review.add(dict(start_s=1, end_s=2, ja="みこが来た", en="Miko arrived."))
        review.add_diagnostic(dict(start_s=1, end_s=2, stage="reconciliation",
                                   flags=["proper_name"], chosen_by="haas"))
        review.add_diagnostic(dict(start_s=1, end_s=2, stage="translation",
                                   translation_flags=[], name_resolutions=[]))
        self.assertEqual(review.cues[key].diagnostic["flags"], ["proper_name"])
        self.assertTrue(review.cues[key].needs_review)

    @staticmethod
    def run_pipeline(translator, texts):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for index, ja in enumerate(texts):
            incoming.put(Transcript(1 + index * 2, 2 + index * 2, ja,
                                    diagnostic_session="name-review"))
        incoming.put(None)
        with patch("auto_translate.pipeline.append_comparison") as record:
            run_translate(translator, 3, incoming, outgoing)
        records = [call.args[0] for call in record.call_args_list]
        subtitles = []
        while (item := outgoing.get_nowait()) is not None:
            subtitles.append(item)
        return records, subtitles

    def test_pipeline_keeps_raw_japanese_and_cue_local_metadata(self):
        resolution = dict(source="巫女", target="Miko", start=0, end=2,
                          evidence="recent_explicit_name")
        translator = SimpleNamespace(memory_lines=0)
        seen = []

        def translate(ja, history):
            seen.append(ja)
            translator.last_name_resolutions = [resolution] if len(seen) == 1 else []
            translator.last_translation_flags = ["proper_name_mismatch"] if len(seen) == 1 else []
            return "Someone arrived." if len(seen) == 1 else "Hello."

        translator.translate = translate
        records, subtitles = self.run_pipeline(translator, ["巫女が来た", "こんにちは"])
        self.assertEqual(seen, ["巫女が来た", "こんにちは"])
        self.assertEqual(subtitles[0].ja, "巫女が来た")
        self.assertEqual(records[0]["final_ja"], "巫女が来た")
        self.assertEqual(records[0]["name_resolutions"], [resolution])
        self.assertEqual(records[0]["translation_flags"], ["proper_name_mismatch"])
        self.assertEqual(records[1]["name_resolutions"], [])
        self.assertEqual(records[1]["translation_flags"], [])

    def test_optional_diagnostics_ignore_non_lists_and_invalid_entries(self):
        translator = SimpleNamespace(memory_lines=0, translate=lambda *_: "Hello.",
                                     last_name_resolutions=Mock(),
                                     last_translation_flags="proper_name_mismatch")
        records, _ = self.run_pipeline(translator, ["こんにちは"])
        self.assertNotIn("name_resolutions", records[0])
        self.assertNotIn("translation_flags", records[0])
        translator.last_name_resolutions = [dict(source="みこ", target="Miko"), None]
        translator.last_translation_flags = ["proper_name_mismatch", None, 1]
        records, _ = self.run_pipeline(translator, ["みこが来た"])
        self.assertEqual(records[0]["name_resolutions"], [dict(source="みこ", target="Miko")])
        self.assertEqual(records[0]["translation_flags"], ["proper_name_mismatch"])

    def test_failed_translation_does_not_emit_previous_name_evidence(self):
        translator = SimpleNamespace(memory_lines=0,
                                     last_name_resolutions=[dict(source="みこ", target="Miko")],
                                     last_translation_flags=["proper_name_mismatch"])
        translator.translate = Mock(side_effect=RuntimeError("translation unavailable"))
        records, subtitles = self.run_pipeline(translator, ["こんにちは"])
        self.assertEqual(subtitles, [])
        self.assertNotIn("name_resolutions", records[0])
        self.assertNotIn("translation_flags", records[0])
        self.assertTrue(records[0]["suppressed"])


if __name__ == "__main__":
    unittest.main()
