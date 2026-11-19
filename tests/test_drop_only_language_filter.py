"""Offline tests for Japanese-only output with conservative CPU acoustic filtering.

No model download or ONNX runtime needed to run these mocked regressions.
"""
import queue
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.config import Config, LanguageFilterConfig
from auto_translate.language_filter import LanguageDecision, LanguageFilter
from auto_translate.pipeline import run_asr
from auto_translate.types import SAMPLE_RATE, Utterance


def make_filter(results=None, cfg=None):
    filt = object.__new__(LanguageFilter)
    filt.cfg = cfg or LanguageFilterConfig()
    filt.retry_at = 0.0
    filt._predict = Mock(side_effect=results or [])
    return filt


def audio(seconds=4.0):
    return np.full(round(seconds * SAMPLE_RATE), 0.08, dtype=np.float32)


class DropOnlyLanguageTests(unittest.TestCase):
    def test_optional_cpu_filter_is_disabled_by_default(self):
        self.assertFalse(Config().language_filter.enabled)
        self.assertEqual(Config().language_filter.min_duration_s, 2.5)

    def test_confident_english_is_dropped(self):
        filt = make_filter([("en", .95, .01, .90)])
        v = filt.classify(audio())
        self.assertTrue(v.drop)
        self.assertEqual(v.language, "en")

    def test_confident_indonesian_is_dropped(self):
        filt = make_filter([("id", .93, .015, .87)])
        self.assertTrue(filt.classify(audio()).drop)

    def test_other_confident_foreign_languages_are_dropped(self):
        for language in ("zh", "ko", "fr", "de"):
            with self.subTest(language=language):
                filt = make_filter([(language, .97, .01, .91)])
                self.assertTrue(filt.classify(audio()).drop)

    def test_foreign_borrowed_word_or_uncertain_classification_is_preserved(self):
        for result in (("zh", .82, .18, .52), ("ko", .65, .03, .3)):
            filt = make_filter([result])
            self.assertFalse(filt.classify(audio()).drop)

    def test_japanese_always_passes(self):
        filt = make_filter([("ja", .93, .93, .80)])
        self.assertFalse(filt.classify(audio()).drop)

    def test_english_with_japanese_evidence_passes(self):
        filt = make_filter([("en", .82, .20, .55)])
        self.assertFalse(filt.classify(audio()).drop)

    def test_uncertain_english_passes(self):
        filt = make_filter([("en", .65, .03, .3)])
        self.assertFalse(filt.classify(audio()).drop)

    def test_mixed_japanese_english_long_segment_passes(self):
        filt = make_filter([("en", .95, .01, .9), ("ja", .89, .89, .8)])
        self.assertFalse(filt.classify(audio(12.0)).drop)
        self.assertEqual(filt._predict.call_count, 2)

    def test_medium_length_code_switch_protects_japanese(self):
        filt = make_filter([("id", .96, .01, .9), ("ja", .81, .81, .6)])
        self.assertFalse(filt.classify(audio(6.0)).drop)
        self.assertEqual(filt._predict.call_count, 2)

    def test_entire_long_segment_must_be_foreign(self):
        filt = make_filter([("en", .94, .02, .9), ("id", .90, .03, .8)])
        self.assertTrue(filt.classify(audio(12.0)).drop)

    def test_short_reactions_are_preserved(self):
        filt = make_filter()
        v = filt.classify(audio(.8))
        self.assertFalse(v.drop)
        self.assertEqual(v.reason, "too_short")
        filt._predict.assert_not_called()

    def test_transient_inference_failure_recovers_after_cooldown(self):
        filt = make_filter()
        filt._predict.side_effect = RuntimeError("onnx runtime error")
        self.assertFalse(filt.classify(audio()).drop)
        self.assertFalse(filt.classify(audio()).drop)
        self.assertEqual(filt._predict.call_count, 1)
        filt.retry_at = 0.0
        filt._predict.side_effect = [("id", .99, .001, .95)]
        self.assertTrue(filt.classify(audio()).drop)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            LanguageFilterConfig(drop_probability=1.5)
        with self.assertRaises(ValueError):
            LanguageFilterConfig(max_window_s=1.5)


class JapanesePipelineOnlyTests(unittest.TestCase):
    @staticmethod
    def exercise(verdict):
        filter_mock = Mock()
        filter_mock.classify.return_value = verdict
        primary = Mock(last_timing=None)
        primary.transcribe.return_value = "今日はどうする？"
        input_q, output_q = queue.Queue(), queue.Queue()
        utt = Utterance(5., 9., audio(), vad_confidence=.9)
        input_q.put(utt)
        input_q.put(None)
        run_asr(primary, input_q, output_q, language_filter=filter_mock)
        return primary, list(output_q.queue)

    def test_english_audio_produces_no_japanese_or_english_subtitles(self):
        primary, result = self.exercise(
            LanguageDecision("en", .99, .01, True, "confident_non_japanese")
        )
        primary.transcribe.assert_not_called()
        self.assertEqual(result, [None])

    def test_indonesian_audio_produces_no_subtitles(self):
        primary, result = self.exercise(
            LanguageDecision("id", .97, .01, True, "confident_non_japanese")
        )
        primary.transcribe.assert_not_called()
        self.assertEqual(result, [None])

    def test_uncertain_audio_still_uses_existing_japanese_pipeline(self):
        primary, result = self.exercise(
            LanguageDecision("unknown", .4, .15, False, "japanese_or_uncertain")
        )
        primary.transcribe.assert_called_once()
        self.assertEqual(result[0].ja, "今日はどうする？")
        self.assertIsNone(result[1])


if __name__ == "__main__":
    unittest.main()
