"""Offline tests for Japanese-only output with CPU EN/ID audio filtering.

No model download or ONNX runtime needed to run these mocked regressions.
"""
import queue
import unittest
from unittest.mock import Mock

import numpy as np

from auto_translate.config import Config, LanguageFilterConfig
from auto_translate.language_filter import LanguageDecision, LanguageFilter
from auto_translate.asr.quality_guard import foreign_audio_transcript_problem
from auto_translate.pipeline import run_asr
from auto_translate.types import SAMPLE_RATE, Utterance


def make_filter(results=None, cfg=None):
    filt = object.__new__(LanguageFilter)
    filt.cfg = cfg or LanguageFilterConfig()
    filt.disabled = False
    filt._predict = Mock(side_effect=results or [])
    return filt


def audio(seconds=4.0):
    return np.full(round(seconds * SAMPLE_RATE), 0.08, dtype=np.float32)


class DropOnlyLanguageTests(unittest.TestCase):
    def test_default_enabled_and_cpu_only_config(self):
        self.assertTrue(Config().language_filter.enabled)
        self.assertEqual(Config().language_filter.min_duration_s, 2.5)

    def test_confident_english_is_dropped(self):
        filt = make_filter([("en", .95, .01, .90)])
        v = filt.classify(audio())
        self.assertTrue(v.drop)
        self.assertEqual(v.language, "en")

    def test_classifier_exposes_windows_for_code_switch_diagnostics(self):
        filt = make_filter([
            ("id", .91, .01, .78),
            ("ja", .87, .87, .71),
            ("en", .94, .01, .83),
        ])
        result = filt.classify(audio(10.0))
        self.assertFalse(result.drop)
        self.assertEqual(len(result.windows), 3)
        self.assertEqual([p[0] for p in result.windows], ["id", "ja", "en"])

    def test_confident_indonesian_is_dropped(self):
        filt = make_filter([("id", .93, .015, .87)])
        self.assertTrue(filt.classify(audio()).drop)

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
        filt = make_filter([
            ("en", .95, .01, .9),
            ("ja", .89, .89, .8),
            ("en", .92, .01, .85),
        ])
        self.assertFalse(filt.classify(audio(12.0)).drop)
        self.assertEqual(filt._predict.call_count, 3)

    def test_medium_length_code_switch_protects_japanese(self):
        filt = make_filter([("id", .96, .01, .9), ("ja", .81, .81, .6)])
        self.assertFalse(filt.classify(audio(6.0)).drop)
        self.assertEqual(filt._predict.call_count, 2)

    def test_entire_long_segment_must_be_foreign(self):
        filt = make_filter([
            ("en", .94, .02, .9),
            ("id", .90, .03, .8),
            ("en", .93, .02, .9),
        ])
        self.assertTrue(filt.classify(audio(12.0)).drop)

    def test_short_reactions_are_preserved(self):
        filt = make_filter()
        v = filt.classify(audio(.8))
        self.assertFalse(v.drop)
        self.assertEqual(v.reason, "too_short")
        filt._predict.assert_not_called()

    def test_inference_failure_disables_filter_without_dropping_japanese(self):
        filt = make_filter()
        filt._predict.side_effect = RuntimeError("onnx runtime error")
        self.assertFalse(filt.classify(audio()).drop)
        self.assertTrue(filt.disabled)
        self.assertFalse(filt.classify(audio()).drop)
        self.assertEqual(filt._predict.call_count, 1)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            LanguageFilterConfig(drop_probability=1.5)
        with self.assertRaises(ValueError):
            LanguageFilterConfig(max_window_s=1.5)


class AudioEvidenceGibberishTests(unittest.TestCase):
    @staticmethod
    def foreign_verdict():
        # Two windows support foreign speech, but a third is uncertain.
        # The pre-ASR ALL-windows LID filter deliberately passed this cue.
        return LanguageDecision(
            "en", .87, .03, False, "japanese_or_uncertain",
            windows=(("en", .87, .02, .62),
                     ("id", .84, .01, .60),
                     ("en", .60, .03, .18)),
        )

    def test_katakana_soup_requires_acoustic_foreign_evidence(self):
        noise = "アミラムゴーイングアルロファーライフ"
        self.assertEqual(
            foreign_audio_transcript_problem(noise, self.foreign_verdict()),
            "foreign_audio_katakana_soup",
        )
        self.assertIsNone(foreign_audio_transcript_problem(noise, None))

    def test_latin_fragment_soup_requires_foreign_audio(self):
        noise = "no fsです ok mybewarkからsj"
        self.assertEqual(
            foreign_audio_transcript_problem(noise, self.foreign_verdict()),
            "foreign_audio_latin_fragments",
        )

    def test_valid_japanese_and_short_reactions_preserved(self):
        for text in ("救急隊が病院に向かいます", "大丈夫ですか？",
                     "はい、OKです", "カプセルカプセル"):
            with self.subTest(text=text):
                self.assertIsNone(foreign_audio_transcript_problem(
                    text, self.foreign_verdict()))

    def test_japanese_audio_vote_prevents_foreign_guard(self):
        verdict = LanguageDecision(
            "en", .90, .80, False, "japanese_or_uncertain",
            windows=(("en", .90, .01, .8),
                     ("ja", .80, .80, .65),
                     ("en", .91, .01, .7)),
        )
        self.assertIsNone(foreign_audio_transcript_problem(
            "アミラムゴーイングアルロファーライフ", verdict))

    def test_reliable_aligned_japanese_caption_prevents_suppression(self):
        caption = type("Caption", (), {
            "reliable": True, "coverage": .92, "text": "あの人が来ます"
        })()
        self.assertIsNone(foreign_audio_transcript_problem(
            "アミラムゴーイングアルロファーライフ",
            self.foreign_verdict(), caption,
        ))

    def test_only_one_foreign_window_does_not_suppress_mixed_speech(self):
        verdict = LanguageDecision(
            "en", .91, .03, False, "japanese_or_uncertain",
            windows=(("en", .91, .01, .8),
                     ("unknown", .60, .03, .20)),
        )
        self.assertIsNone(foreign_audio_transcript_problem(
            "アミラムゴーイングアルロファーライフ", verdict))

    def test_pipeline_drops_foreign_phonetics_before_translation(self):
        inp, out = queue.Queue(), queue.Queue()
        primary = Mock()
        primary.transcribe.return_value = "アミラムゴーイングアルロファーライフ"
        lid = Mock()
        lid.classify.return_value = self.foreign_verdict()
        inp.put(Utterance(5., 14., audio(9.), vad_confidence=.92))
        inp.put(None)
        run_asr(primary, inp, out, language_filter=lid)
        primary.transcribe.assert_called_once()
        self.assertEqual(list(out.queue), [None])

    def test_pipeline_preserves_coherent_japanese_when_lid_uncertain(self):
        inp, out = queue.Queue(), queue.Queue()
        primary = Mock()
        primary.transcribe.return_value = "病院に向かいます"
        lid = Mock()
        lid.classify.return_value = self.foreign_verdict()
        inp.put(Utterance(5., 14., audio(9.), vad_confidence=.92))
        inp.put(None)
        run_asr(primary, inp, out, language_filter=lid)
        self.assertEqual(out.get_nowait().ja, "病院に向かいます")


class JapanesePipelineOnlyTests(unittest.TestCase):
    @staticmethod
    def exercise(verdict):
        filter_mock = Mock()
        filter_mock.classify.return_value = verdict
        primary = Mock()
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
