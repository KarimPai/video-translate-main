"""Overhaul regressions: keep the original models while tightening output boundaries."""
import queue
import unittest
from unittest.mock import Mock

from auto_translate.config import AsrConfig, DualAsrConfig, TranslateConfig, Config
from auto_translate.pipeline import run_translate
from auto_translate.subtitle_clean import english_output_issue, sanitize_english_output
from auto_translate.types import Transcript


class ModelContractTests(unittest.TestCase):
    def test_original_models_stay_selected(self):
        cfg = Config()
        self.assertEqual(cfg.asr.haas_model, "haas-ja-v1.0")
        self.assertIn("faster-whisper-large-v3-turbo", cfg.dual_asr.secondary_model)
        self.assertIn("plamo-3-translate", cfg.translate.plamo_hf_model)
        self.assertNotIn("Hy-MT", cfg.translate.plamo_hf_model)
        self.assertTrue(cfg.dual_asr.enabled)

    def test_context_gap_validation(self):
        for invalid in (0, -1, True, "15", float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                TranslateConfig(context_max_gap_s=invalid)


class SubtitleBoundaryTests(unittest.TestCase):
    def test_uncensored_speech_survives(self):
        self.assertEqual(sanitize_english_output("Fuck, that's bad!"),
                         "Fuck, that's bad!")
        self.assertEqual(sanitize_english_output("Kusa!"), "Kusa!")
        self.assertEqual(sanitize_english_output("Matsuri is here."), "Matsuri is here.")

    def test_unpaired_quotes_do_not_propagate(self):
        self.assertEqual(sanitize_english_output('"What happened?'), "What happened?")
        self.assertEqual(sanitize_english_output('"What happened?"'), '"What happened?"')

    def test_japanese_leak_and_template_are_suppressed(self):
        self.assertEqual(sanitize_english_output("まって！"), "")
        self.assertIsNotNone(english_output_issue("Translate Japanese speech to English only"))


class DialogueContinuityTests(unittest.TestCase):
    @staticmethod
    def _call_model(cues, gap=15.0):
        translator = Mock()
        translator.cfg = TranslateConfig(context_max_gap_s=gap)
        translator.memory_lines = 4
        translator.last_name_diagnostics = {}
        translator.translate.side_effect = lambda ja, history: "Okay."
        input_q, output_q = queue.Queue(), queue.Queue()
        for start, end, ja in cues:
            input_q.put(Transcript(start, end, ja))
        input_q.put(None)
        run_translate(translator, 3, input_q, output_q)
        return translator.translate.call_args_list, list(output_q.queue)

    def test_recent_history_kept_across_nearby_cues_and_reset_after_silence(self):
        calls, outputs = self._call_model([
            (0, 1, "何？"), (5, 6, "それ何？"), (25, 26, "次に行く")
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].args[1], [])
        self.assertEqual(len(calls[1].args[1]), 1)
        self.assertEqual(calls[2].args[1], [])
        self.assertEqual(len([x for x in outputs if x is not None]), 3)

    def test_expired_translation_does_not_enter_history(self):
        translator = Mock()
        translator.cfg = TranslateConfig()
        translator.memory_lines = 3
        translator.last_name_diagnostics = {}
        translator.translate.side_effect = ["こんにちは", "Yes."]
        input_q, output_q = queue.Queue(), queue.Queue()
        input_q.put(Transcript(0, 1, "こんにちは"))
        input_q.put(Transcript(2, 3, "はい"))
        input_q.put(None)
        run_translate(translator, 3, input_q, output_q)
        self.assertEqual(len(translator.translate.call_args_list), 2)
        self.assertEqual(translator.translate.call_args_list[1].args[1], [])
        subtitles = [x for x in list(output_q.queue) if x is not None]
        self.assertEqual(len(subtitles), 1)
        self.assertEqual(subtitles[0].en, "Yes.")


if __name__ == "__main__":
    unittest.main()
