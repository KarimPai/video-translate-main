"""Deterministic unit tests for selective Haas multi-pass ASR."""
import unittest

import numpy as np

from auto_translate.asr_retry import (
    choose_hypothesis, needs_retry, quality_score, select_best,
)
from auto_translate.captions import CaptionMatch
from auto_translate.config import AsrConfig


class FakeHaas:
    def __init__(self, results, *, keywords="ホロライブ", config=None, timeout=False):
        self.cfg = config or AsrConfig()
        self.keywords = keywords
        self.results = iter(results)
        self.calls = []
        self.timeout = timeout

    def transcribe_bounded(self, audio, *, timeout_s, use_keywords):
        self.calls.append((len(audio), timeout_s, use_keywords))
        if self.timeout:
            raise TimeoutError("bounded retry expired")
        return next(self.results)


def match(text, *, reliable=True, coverage=.95, overlap_s=1.2):
    return CaptionMatch(text, overlap_s, coverage, reliable)


class RetryScoringTests(unittest.TestCase):
    def test_clear_sentence_does_not_trigger_retry(self):
        asr = FakeHaas(["garbage"])
        audio = np.zeros(40000, dtype=np.float32)
        actual = choose_hypothesis(
            asr, audio, "車を修理してきます",
            start_s=0, end_s=2.5, live=True,
        )
        self.assertEqual(actual, "車を修理してきます")
        self.assertFalse(asr.calls)

    def test_longest_text_does_not_win_without_evidence(self):
        winner = select_best(
            ["わためだよ", "わためだよ、ええええなんとかなんとか"],
            duration_s=2.0,
        )
        self.assertEqual(winner.attempt, 1)
        self.assertEqual(winner.text, "わためだよ")

    def test_caption_alignment_selects_better_transcript(self):
        cap = match("バーガーキング食べた")
        choice = select_best(
            ["あの防号キング倒れた", "バーガーキング食べた"],
            duration_s=2.5, caption=cap,
        )
        self.assertEqual(choice.attempt, 2)
        self.assertGreater(
            quality_score(choice.text, duration_s=2.5, caption=cap),
            quality_score("あの防号キング倒れた", duration_s=2.5, caption=cap),
        )

    def test_weak_caption_never_controls_selection(self):
        cap = match("バーガーキング食べた", reliable=False)
        choice = select_best(
            ["防号キング", "バーガーキング食べた"],
            duration_s=1.4, caption=cap,
        )
        self.assertEqual(choice.attempt, 1)

    def test_short_response_acceptable_for_short_audio(self):
        self.assertFalse(needs_retry("うん", duration_s=.4))
        self.assertTrue(needs_retry("うん", duration_s=2.7))

    def test_gibberish_with_isolated_latin_is_suspicious(self):
        self.assertTrue(needs_retry("あやnめちゃん", duration_s=1.7))

    def test_live_one_optional_retry_and_short_budget(self):
        cfg = AsrConfig(retry_live_budget_s=1.2)
        asr = FakeHaas(["おかゆさんです"], config=cfg)
        audio = np.zeros(32000, dtype=np.float32)
        result = choose_hypothesis(
            asr, audio, "え", start_s=3, end_s=5,
            speech_start_s=3.2, live=True,
        )
        self.assertEqual(result, "おかゆさんです")
        self.assertEqual(len(asr.calls), 1)
        self.assertTrue(asr.calls[0][2])  # boundary edit keeps keyword boost
        self.assertLessEqual(asr.calls[0][1], 1.2)
        self.assertGreater(asr.calls[0][1], 0)

    def test_vod_uses_third_pass_when_caption_disagrees(self):
        cfg = AsrConfig(retry_budget_s=3.0)
        asr = FakeHaas(["防号キング", "バーガーキング食べた"], config=cfg)
        audio = np.zeros(40000, dtype=np.float32)
        result = choose_hypothesis(
            asr, audio, "あの防号キング倒れた",
            start_s=0, end_s=2.5,
            speech_start_s=.2, caption=match("バーガーキング食べた"),
        )
        self.assertEqual(result, "バーガーキング食べた")
        self.assertEqual(len(asr.calls), 2)
        self.assertTrue(asr.calls[0][2])
        self.assertFalse(asr.calls[1][2])
        self.assertLessEqual(asr.calls[1][1], asr.calls[0][1])

    def test_timeout_keeps_original_without_starting_more_passes(self):
        asr = FakeHaas([], timeout=True)
        audio = np.zeros(32000, dtype=np.float32)
        result = choose_hypothesis(
            asr, audio, "え", start_s=0, end_s=2,
            speech_start_s=.2,
        )
        self.assertEqual(result, "え")
        self.assertEqual(len(asr.calls), 1)

    def test_disabled_or_probe_never_retries(self):
        asr = FakeHaas(["better"], config=AsrConfig(retry_uncertain_enabled=False))
        audio = np.zeros(40000, dtype=np.float32)
        self.assertEqual(
            choose_hypothesis(asr, audio, "え", start_s=0, end_s=2.5), "え"
        )
        self.assertFalse(asr.calls)
        asr.cfg.retry_uncertain_enabled = True
        self.assertEqual(
            choose_hypothesis(asr, audio, "え", start_s=0, end_s=2.5, probe=True), "え"
        )
        self.assertFalse(asr.calls)

    def test_no_duplicate_decode_without_genuine_variant(self):
        asr = FakeHaas([], keywords="")
        audio = np.zeros(7 * 16000, dtype=np.float32)
        result = choose_hypothesis(
            asr, audio, "え", start_s=0, end_s=7,
            speech_start_s=0, live=False,
        )
        self.assertEqual(result, "え")
        self.assertFalse(asr.calls)

    def test_zero_budget_skips_retry(self):
        asr = FakeHaas([], config=AsrConfig(retry_budget_s=0))
        audio = np.zeros(40000, dtype=np.float32)
        self.assertEqual(
            choose_hypothesis(asr, audio, "え", start_s=0, end_s=2.5), "え"
        )
        self.assertFalse(asr.calls)


if __name__ == "__main__":
    unittest.main()
