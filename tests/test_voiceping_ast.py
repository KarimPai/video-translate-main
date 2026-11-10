"""Mock-backed AST regression tests; no GPU download required."""

import contextlib
import queue
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from auto_translate.config import VoicePingAstConfig
from auto_translate.pipeline import run_ast
from auto_translate.types import AudioProgress, SAMPLE_RATE, Utterance
from auto_translate.voiceping_ast import VoicePingAST


class MockVoicePing:
    def __init__(self, text="There is one sheep."):
        self.text = text
        self.calls = []

    def transcribe(self, *, audio, language):
        self.calls.append((audio, language))
        return [types.SimpleNamespace(text=self.text)]


class VoicePingASTTests(unittest.TestCase):
    def test_target_language_and_original_pcm(self):
        fake = MockVoicePing()
        model = VoicePingAST(VoicePingAstConfig(), model=fake)
        pcm = np.full(SAMPLE_RATE, 0.12, dtype=np.float32)
        # The model path imports torch only for the inference-mode context.
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(
                inference_mode=contextlib.nullcontext)}):
            english = model.translate_audio(pcm)
        self.assertEqual(english, "There is one sheep.")
        self.assertEqual(len(fake.calls), 1)
        received, language = fake.calls[0]
        self.assertEqual(language, "English")
        self.assertEqual(received[1], SAMPLE_RATE)
        np.testing.assert_array_equal(received[0], pcm)

    def test_invalid_pcm_not_passed_to_translation(self):
        fake = MockVoicePing()
        model = VoicePingAST(VoicePingAstConfig(), model=fake)
        self.assertEqual(model.translate_audio(np.array([np.nan])), "")
        self.assertEqual(model.translate_audio(np.empty(0, dtype=np.float32)), "")
        self.assertEqual(fake.calls, [])

    def test_no_quantization_or_non_bf16_mode(self):
        with self.assertRaises(ValueError):
            VoicePingAST(VoicePingAstConfig(dtype="float32"), model=MockVoicePing())

    def test_live_worker_single_pass_and_english_only(self):
        fake = MockVoicePing("The sheep is here.")
        model = VoicePingAST(VoicePingAstConfig(), model=fake)
        inp, out = queue.Queue(), queue.Queue()
        inp.put(Utterance(
            start_s=5.0, end_s=6.0, audio=np.full(SAMPLE_RATE, 0.1, dtype=np.float32),
            generation=2, vad_confidence=0.92,
        ))
        inp.put(None)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(
                inference_mode=contextlib.nullcontext)}):
            run_ast(model, inp, out, lambda: 2)
        sub = out.get_nowait()
        self.assertEqual((sub.start_s, sub.end_s, sub.ja, sub.en),
                         (5.0, 6.0, "", "The sheep is here."))
        self.assertEqual(len(fake.calls), 1)
        self.assertIsNone(out.get_nowait())

    def test_seek_generation_discard_and_noise_probe(self):
        fake = MockVoicePing()
        model = VoicePingAST(VoicePingAstConfig(), model=fake)
        inp, out = queue.Queue(), queue.Queue()
        for gen, probe in ((0, False), (1, True)):
            inp.put(Utterance(
                start_s=0.0, end_s=1.0, audio=np.full(SAMPLE_RATE, 0.1, dtype=np.float32),
                generation=gen, probe=probe,
            ))
        inp.put(AudioProgress(SAMPLE_RATE, generation=1))
        inp.put(None)
        run_ast(model, inp, out, lambda: 1)
        progress = out.get_nowait()
        self.assertEqual(progress.generation, 1)
        self.assertIsNone(out.get_nowait())
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
