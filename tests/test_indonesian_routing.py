"""Offline Indonesian language-ID and direct translation regression tests.

The CPU Whisper model is mocked so the suite never downloads model files.
"""
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.dual import DualASR
from auto_translate.asr.indonesian import IndonesianWhisper, IndonesianResult
from auto_translate.asr.nemotron import LanguageTranscript
from auto_translate.config import Config, DualAsrConfig
from auto_translate.glossary import Glossary
from auto_translate.pipeline import run_asr, run_translate
from auto_translate.types import Subtitle, Utterance


def _audio(seconds=2):
    return np.ones(int(seconds * 16000), dtype=np.float32) * 0.1


def _fake_worker(language, probability, alternative, output):
    cpu = object.__new__(IndonesianWhisper)
    cpu.model = Mock()
    info = SimpleNamespace(
        language=language,
        language_probability=probability,
        all_language_probs=[(language, probability),
                            (alternative[0], alternative[1])],
    )
    cpu.model.transcribe.side_effect = [
        (iter(()), info),
        (iter([SimpleNamespace(text=output)]), info),
    ]
    return cpu


class CpuIndonesianWhisperTests(unittest.TestCase):
    def test_confident_id_speech_translates_to_english_on_cpu(self):
        cpu = _fake_worker("id", .93, ("en", .04),
                           "The game crashed, but I can reconnect.")
        result = cpu.transcribe_if_indonesian(_audio(3.0))
        self.assertIsInstance(result, IndonesianResult)
        self.assertEqual(result.english, "The game crashed, but I can reconnect.")
        self.assertEqual(cpu.model.transcribe.call_count, 2)
        detect_call = cpu.model.transcribe.call_args_list[0]
        translate_call = cpu.model.transcribe.call_args_list[1]
        self.assertEqual(detect_call.kwargs["language"], None)
        self.assertEqual(translate_call.kwargs["language"], "id")
        self.assertEqual(translate_call.kwargs["task"], "translate")
        self.assertFalse(translate_call.kwargs["condition_on_previous_text"])

    def test_english_japanese_and_low_confidence_id_are_not_routed(self):
        for language, probability, alt in (
            ("en", .94, ("id", .04)),
            ("ja", .94, ("id", .04)),
            ("id", .61, ("en", .31)),
            ("id", .75, ("en", .70)),
        ):
            with self.subTest(language=language, probability=probability):
                cpu = _fake_worker(language, probability, alt, "irrelevant")
                self.assertIsNone(cpu.transcribe_if_indonesian(_audio(3)))
                cpu.model.transcribe.assert_called_once()

    def test_short_utterance_never_forces_id_detection(self):
        cpu = object.__new__(IndonesianWhisper)
        cpu.model = Mock()
        self.assertIsNone(cpu.transcribe_if_indonesian(_audio(0.5)))
        cpu.model.transcribe.assert_not_called()

    def test_indonesian_without_valid_english_is_not_misrouted_to_japanese(self):
        cpu = _fake_worker("id", .94, ("en", .03), "[Music]")
        result = cpu.transcribe_if_indonesian(_audio(3))
        # A weak/empty translation must never be invented.
        # The engine returns the accepted Indonesian classification for caller
        # to drop rather than using Japanese ASR as its fallback.
        self.assertIsNotNone(result)
        self.assertFalse(result.english)

    def test_config_validates_thresholds(self):
        self.assertTrue(DualAsrConfig().indonesian_routing)
        for kwargs in (
            {"indonesian_min_confidence": 0.20},
            {"indonesian_min_margin": -0.2},
            {"indonesian_cpu_threads": 0},
            {"indonesian_min_duration_s": 0.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DualAsrConfig(**kwargs)


class IndonesianPipelineTests(unittest.TestCase):
    def setup_pipeline(self, response):
        cfg = Config()
        cfg.dual_asr.enabled = False
        dual = DualASR(cfg, Mock(pressure_level="normal"), Glossary({}), False)
        cfg.dual_asr.enabled = True
        dual.indonesian = Mock()
        dual.indonesian.transcribe_if_indonesian.return_value = response
        dual.worker = Mock()
        return dual

    def test_accepted_id_routes_to_en_without_japanese_decode_or_translation(self):
        dual = self.setup_pipeline(IndonesianResult(
            "I need to reconnect to the server.", .96, 1.3
        ))
        primary = Mock()
        in_q, out_q = queue.Queue(), queue.Queue()
        in_q.put(Utterance(22, 25, _audio(3), vad_confidence=.95))
        in_q.put(None)
        with patch("auto_translate.asr.dual.append_comparison") as log_comparison:
            run_asr(primary, in_q, out_q, dual=dual)
        subtitle = out_q.get_nowait()
        self.assertIsInstance(subtitle, Subtitle)
        self.assertEqual((subtitle.start_s, subtitle.end_s), (22, 25))
        self.assertEqual(subtitle.ja, "")
        self.assertEqual(subtitle.en, "I need to reconnect to the server.")
        self.assertIsNone(out_q.get_nowait())
        primary.transcribe.assert_not_called()
        dual.worker.transcribe.assert_not_called()
        dual.worker.transcribe_with_language.assert_not_called()
        record = log_comparison.call_args.args[0]
        self.assertEqual(record["detected_language"], "id-ID")
        self.assertEqual(record["chosen_by"], "indonesian_whisper")

        translate_in, translate_out = queue.Queue(), queue.Queue()
        translate_in.put(subtitle)
        translate_in.put(None)
        plamo = Mock(memory_lines=8)
        run_translate(plamo, 4, translate_in, translate_out)
        self.assertEqual(translate_out.get_nowait().en, subtitle.en)
        plamo.translate.assert_not_called()

    def test_accepted_indonesian_without_english_drops_japanese_hallucination(self):
        dual = self.setup_pipeline(IndonesianResult("", .92, 1.5,
                                                      "indonesian_no_english"))
        primary = Mock()
        source, target = queue.Queue(), queue.Queue()
        source.put(Utterance(22, 25, _audio(3), vad_confidence=.95))
        source.put(None)
        with patch("auto_translate.asr.dual.append_comparison"):
            run_asr(primary, source, target, dual=dual)
        self.assertIsNone(target.get_nowait())
        primary.transcribe.assert_not_called()

    def test_non_indonesian_follows_existing_asr(self):
        dual = self.setup_pipeline(None)
        dual.worker.transcribe_with_language.return_value = LanguageTranscript(
            "待って", "ja-JP"
        )
        dual.worker.transcribe.return_value = "待って"
        primary = Mock(last_stop="complete")
        primary.transcribe.return_value = "待って"
        source, target = queue.Queue(), queue.Queue()
        source.put(Utterance(22, 25, _audio(3), vad_confidence=.95))
        source.put(None)
        with patch("auto_translate.asr.dual.append_comparison"):
            run_asr(primary, source, target, dual=dual)
        primary.transcribe.assert_called_once()
        self.assertEqual(target.get_nowait().ja, "待って")


if __name__ == "__main__":
    unittest.main()
