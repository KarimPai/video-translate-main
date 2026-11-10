"""Offline Kotoba v2.2 + current PLaMo 3 translation regressions.

No model downloads, GPU, server or MPV are required for these tests.
"""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import WhisperConfig, TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.kotoba_whisper import (
    KOTOBA_CT2_ID, KOTOBA_OFFICIAL_ID, KotobaWhisperASR,
)
from auto_translate.translate.plamo import PlamoTranslator


class KotobaPlamoTests(unittest.TestCase):
    def test_defaults_latest_kotoba_with_low_vram_plamo3(self):
        whisper = WhisperConfig()
        plamo = TranslateConfig()
        self.assertEqual(whisper.model, KOTOBA_CT2_ID)
        self.assertEqual(KOTOBA_OFFICIAL_ID, "kotoba-tech/kotoba-whisper-v2.2")
        self.assertEqual(whisper.compute_type, "int8_float16")
        self.assertEqual((whisper.language, whisper.task), ("ja", "transcribe"))
        self.assertFalse(whisper.condition_on_previous_text)
        self.assertIn("plamo-3-translate-2b-GGUF:Q4_K_M", plamo.plamo_hf_model)
        self.assertEqual((plamo.plamo_context, plamo.plamo_gpu_layers), (4096, 16))
        self.assertEqual((plamo.context_lines, plamo.memory_lines), (8, 24))

    def test_rejects_accidentally_loading_whisper_large_v3(self):
        with self.assertRaisesRegex(ValueError, "Kotoba v2.2"):
            KotobaWhisperASR(
                WhisperConfig(model="large-v3"),
                model=object(),
            )

    def test_transcribes_16khz_pcm_without_prior_text_prompts(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                self.audio = audio
                self.kwargs = kwargs
                segment = SimpleNamespace(
                    text="ワタメがロケットランチャーを持ってる",
                    avg_logprob=-0.25,
                    compression_ratio=1.0,
                    no_speech_prob=0.02,
                )
                return iter([segment]), None

        model = Model()
        backend = KotobaWhisperASR(WhisperConfig(), model=model)
        pcm = np.full(16000, .2, dtype=np.float32)
        result = backend.decode_audio(pcm, prompt=None, hotwords=None)
        self.assertEqual(result.text, "ワタメがロケットランチャーを持ってる")
        self.assertEqual(model.kwargs["task"], "transcribe")
        self.assertEqual(model.kwargs["language"], "ja")
        self.assertEqual(model.kwargs["beam_size"], 5)
        self.assertEqual(model.kwargs["condition_on_previous_text"], False)
        self.assertFalse(model.kwargs["vad_filter"])
        self.assertIsNone(model.kwargs["initial_prompt"])
        self.assertIsNone(model.kwargs["hotwords"])
        self.assertEqual(len(model.audio), 16000)
        self.assertEqual(result.avg_logprob, -0.25)

    def test_plamo3_uses_explicit_ja_to_en_with_current_gguf_template(self):
        translator = PlamoTranslator.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig()
        translator.glossary = Glossary({"ワタメ": "Watame"})
        translator.memory_lines = translator.cfg.memory_lines
        translator.vod_title = "Hololive GTA"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        sent = []

        def respond(req, timeout=None):
            sent.append(json.loads(req.data.decode("utf-8")))
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"Watame is here."}}]}'
            )

        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   side_effect=respond):
            translated = translator.translate(
                "ワタメが来た", [],
                fragment_context=("車が来た",),
            )
        self.assertEqual(translated, "Watame is here.")
        body = sent[0]
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["model"], "plamo-3-translate")
        self.assertEqual(body["messages"][-1]["content"], "ワタメが来た")
        options = body["chat_template_kwargs"]
        self.assertEqual(options["source_lang"], "ja")
        self.assertEqual(options["target_lang"], "en")
        self.assertEqual(options["glossary"], [
            {"source": "ワタメ", "target": "Watame"},
        ])
        self.assertIn("車が来た", options["context"])


if __name__ == "__main__":
    unittest.main()
