"""Offline Anime Whisper adapter tests: no model downloads or GPU required."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from auto_translate.anime_whisper import AnimeWhisperASR, MODEL_ID
from auto_translate.asr_quality import is_standalone_laughter, is_formulaic_japanese_outro
from auto_translate.config import WhisperConfig


class FakeProcessor:
    def __init__(self, text="ロケランと車が来た"):
        self.text = text
        self.sampling_rate = None

    def __call__(self, samples, *, sampling_rate, return_tensors,
                 return_attention_mask):
        self.sampling_rate = sampling_rate
        assert return_tensors == "pt"
        assert return_attention_mask is True
        return {
            "input_features": torch.zeros((1, 80, 3000)),
            "attention_mask": torch.ones((1, 3000), dtype=torch.long),
        }

    def batch_decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return [self.text]


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


class AnimeWhisperTests(unittest.TestCase):
    def make_backend(self, **options):
        cfg = WhisperConfig(device="cpu", compute_type="float32", **options)
        model, processor = FakeModel(), FakeProcessor()
        return AnimeWhisperASR(cfg, model=model, processor=processor), model, processor

    def test_official_model_default_and_anti_repetition(self):
        cfg = WhisperConfig()
        self.assertEqual(cfg.model, MODEL_ID)
        self.assertEqual(cfg.language, "ja")
        self.assertEqual(cfg.task, "transcribe")
        self.assertEqual(cfg.no_repeat_ngram_size, 5)
        self.assertEqual(cfg.repetition_penalty, 1.0)
        self.assertFalse(cfg.retry_uncertain)
        self.assertTrue(cfg.adaptive_max_tokens)
        self.assertTrue(cfg.suppress_standalone_laughter)
        self.assertEqual(cfg.max_new_tokens, 160)

    def test_decode_japanese_and_no_initial_prompt_or_hotwords(self):
        asr, model, processor = self.make_backend()
        samples = np.ones(16000, dtype=np.float32) * .2
        result = asr.decode_audio(samples,
                                  prompt="previous English subtitle",
                                  hotwords="Pekora Hololive")
        self.assertEqual(result.text, "ロケランと車が来た")
        self.assertEqual(processor.sampling_rate, 16000)
        self.assertEqual(len(model.calls), 1)
        kwargs = model.calls[0]
        self.assertEqual(kwargs["language"], "japanese")
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertEqual(kwargs["num_beams"], 5)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 5)
        self.assertEqual(kwargs["repetition_penalty"], 1.0)
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("initial_prompt", kwargs)
        self.assertNotIn("prompt_ids", kwargs)
        self.assertNotIn("hotwords", kwargs)
        self.assertEqual(kwargs["max_new_tokens"], 64)  # 1s audio

    def test_silence_and_empty_audio_never_run_decoder(self):
        asr, model, _ = self.make_backend()
        self.assertEqual(asr.transcribe_audio(np.zeros(16000)), "")
        self.assertEqual(asr.transcribe_audio(np.zeros(0)), "")
        self.assertEqual(model.calls, [])

    def test_nonfinite_and_oversized_audio_rejected(self):
        asr, model, _ = self.make_backend()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            asr.transcribe_audio(np.array([float("nan")]))
        with self.assertRaisesRegex(ValueError, "30 seconds"):
            asr.transcribe_audio(np.ones(480001, dtype=np.float32))
        self.assertEqual(model.calls, [])

    def test_temperature_retry_is_optional(self):
        asr, model, _ = self.make_backend()
        output = asr.decode_audio(np.full(16000, .1, dtype=np.float32),
                                  temperature=0.2)
        self.assertEqual(output.temperature, 0.2)
        self.assertTrue(model.calls[0]["do_sample"])
        self.assertEqual(model.calls[0]["temperature"], 0.2)

    def test_invalid_config_rejected_without_download(self):
        cfg = WhisperConfig(model="large-v3", device="cpu")
        with self.assertRaisesRegex(ValueError, "supports only"):
            AnimeWhisperASR(cfg, model=FakeModel(), processor=FakeProcessor())
        cfg = WhisperConfig(device="cpu", task="translate")
        with self.assertRaisesRegex(ValueError, "Japanese transcription"):
            AnimeWhisperASR(cfg, model=FakeModel(), processor=FakeProcessor())

    def test_token_budget_respects_longer_audio_and_config_override(self):
        asr, model, _ = self.make_backend()
        asr.decode_audio(np.full(3 * 16000, .1, dtype=np.float32))
        self.assertEqual(model.calls[0]["max_new_tokens"], 122)
        asr.decode_audio(np.full(8 * 16000, .1, dtype=np.float32))
        self.assertEqual(model.calls[1]["max_new_tokens"], 160)
        asr.cfg.adaptive_max_tokens = False
        asr.decode_audio(np.full(16000, .1, dtype=np.float32))
        self.assertEqual(model.calls[2]["max_new_tokens"], 160)

    def test_only_pure_laughter_is_suppressed(self):
        for text in ("あははは", "はっはっはっ", "ふふ…", "ハハハ!", "ｗｗｗ"):
            self.assertTrue(is_standalone_laughter(text), text)
        for text in ("そうだね、あはは", "うん", "ありがとう", "あはは、やめて",
                     "ええ", "すごい!", "はは、面白い", "何を言ってるんだ"):
            self.assertFalse(is_standalone_laughter(text), text)

    def test_exact_formulaic_japanese_outro_detection(self):
        self.assertTrue(is_formulaic_japanese_outro("ご視聴ありがとうございました。"))
        self.assertFalse(is_formulaic_japanese_outro("ご視聴ありがとうございました、またね"))
        self.assertFalse(is_formulaic_japanese_outro("ありがとう！"))


if __name__ == "__main__":
    unittest.main()
