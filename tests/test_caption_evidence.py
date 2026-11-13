"""Regressions for conservative ASR corroboration from aligned JP captions."""
import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.quality_guard import (
    caption_supported_hypothesis,
    disagreement_unresolved,
)
from auto_translate.captions import CaptionMatch
from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_asr
from auto_translate.types import Utterance


def matched(text, coverage=.95):
    return CaptionMatch(text, overlap_s=1., coverage=coverage, reliable=True)


class CaptionEvidenceTests(unittest.TestCase):
    def test_caption_supports_haas_when_lfm_prefers_wrong_hypothesis(self):
        evidence = caption_supported_hypothesis(
            "今から逮捕する", "今から台本する", matched("今から逮捕する")
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence[0], "haas")

    def test_caption_supports_parakeet_when_haas_omits_negation(self):
        evidence = caption_supported_hypothesis(
            "今から行く", "今から行かない", matched("今から行かない")
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence[0], "parakeet")

    def test_rolling_or_low_coverage_caption_cannot_overrule(self):
        for caption in (
            matched("今から逮捕する", coverage=.5),
            matched("別のところへ向かいましょう"),
            CaptionMatch("今から逮捕する", overlap_s=.1, coverage=.95, reliable=False),
        ):
            self.assertIsNone(caption_supported_hypothesis(
                "今から逮捕する", "今から台本する", caption
            ))

    def make_dual(self, secondary, selected, selection):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        dual.worker = Mock(last_stop="complete")
        dual.worker.transcribe.return_value = secondary
        dual.selector = Mock(last_choice=selection)
        dual.selector.choose.return_value = selected
        return dual

    def utterance(self):
        return Utterance(12., 14., np.ones(32000, dtype=np.float32),
                         vad_confidence=.95)

    def test_lfm_wrong_choice_is_overridden_by_strong_jp_caption(self):
        dual = self.make_dual("今から台本する", "今から台本する", "parakeet")
        with patch("auto_translate.asr.dual.append_comparison") as save:
            result = dual.choose(
                self.utterance(), "今から逮捕する", "ホロGTA", lambda: False,
                None, caption_match=matched("今から逮捕する"),
            )
        self.assertEqual(result, "今から逮捕する")
        self.assertEqual(save.call_args.args[0]["chosen_by"],
                         "haas_caption_evidence")

    def test_lfm_uncertain_is_rescued_only_with_strong_caption(self):
        dual = self.make_dual("ここは危険です", "今から行かない", "uncertain")
        with patch("auto_translate.asr.dual.append_comparison") as save:
            result = dual.choose(
                self.utterance(), "今から行かない", "ホロGTA", lambda: False,
                None, caption_match=matched("今から行かない"),
            )
        self.assertEqual(result, "今から行かない")
        self.assertEqual(save.call_args.args[0]["chosen_by"],
                         "haas_caption_evidence")

    def test_lfm_uncertain_without_evidence_still_abstains(self):
        dual = self.make_dual("ここは危険です", "今から行かない", "uncertain")
        with patch("auto_translate.asr.dual.append_comparison") as save:
            result = dual.choose(
                self.utterance(), "今から行かない", "ホロGTA", lambda: False,
                None, caption_match=matched("関係ない話", coverage=.5),
            )
        self.assertEqual(result, "")
        self.assertEqual(save.call_args.args[0]["chosen_by"],
                         "abstain_disagreement")

    def test_lfm_abstention_blocks_unrelated_full_length_transcripts(self):
        self.assertTrue(disagreement_unresolved(
            "今日は病院の前で待っててください",
            "警察署から連絡が来たんですけど",
            "今日は病院の前で待っててください",
            0.18, [], "uncertain",
        ))

    def test_short_unflagged_disagreement_remains_conservative(self):
        self.assertFalse(disagreement_unresolved(
            "はい", "うん", "はい", 0.12, [], "uncertain",
        ))

    def test_pipeline_passes_matching_captions_to_dual(self):
        primary = Mock()
        primary.transcribe.return_value = "今から逮捕する"
        captions = Mock()
        captions.match.return_value = matched("今から逮捕する")
        dual = Mock()
        dual.select_segment.return_value = "今から逮捕する"
        utt = self.utterance()
        a, b = queue.Queue(), queue.Queue()
        a.put(utt)
        a.put(None)
        run_asr(primary, a, b, captions=captions, dual=dual)
        self.assertEqual(captions.match.call_count, 1)
        self.assertIs(dual.select_segment.call_args.args[-1],
                      captions.match.return_value)
        self.assertEqual(b.get_nowait().ja, "今から逮捕する")


if __name__ == "__main__":
    unittest.main()
