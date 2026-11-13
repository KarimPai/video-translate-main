"""CPU-only contracts for Japanese Whisper Turbo secondary ASR."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from auto_translate.asr.whisper_turbo import _audio16, _decode, MODEL
from auto_translate.asr.reconciler import MODEL as SELECTOR_MODEL


class WhisperTurboAdapterTests(unittest.TestCase):
    def test_audio_is_original_float32_mono(self):
        sample = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        self.assertTrue(np.array_equal(_audio16(sample), sample))
        self.assertEqual(_audio16([2, -2]).tolist(), [1, -1])
        for invalid in ([float("nan")], [], [[1, 2]]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _audio16(invalid)

    def test_locked_japanese_transcription_and_no_internal_vad(self):
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" まだ行かない ", no_speech_prob=.1, avg_logprob=-.2)]),
            None,
        )
        self.assertEqual(_decode(model, np.ones(16000, dtype=np.float32), 6),
                         "まだ行かない")
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], "ja")
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertEqual(kwargs["beam_size"], 6)

    def test_low_evidence_hallucination_is_dropped(self):
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text="ご視聴ありがとうございました",
                                  no_speech_prob=.95, avg_logprob=-1.4)]), None
        )
        self.assertEqual(_decode(model, np.ones(16000, dtype=np.float32), 6), "")

    def test_checkpoint_and_selector_preserved(self):
        self.assertEqual(MODEL, "h2oai/faster-whisper-large-v3-turbo")
        self.assertEqual(SELECTOR_MODEL, "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF:Q4_K_M")


if __name__ == "__main__":
    unittest.main()
