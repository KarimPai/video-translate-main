"""Granite-only regression tests; no model download or external server needed."""

import io
import json
import queue
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.config import GraniteConfig, load_config
from auto_translate.granite import _clean_result, build_multipart, encode_wav
from auto_translate.pipeline import _discard_probe, run_granite
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import AudioProgress, Subtitle, Utterance


class GraniteWireTests(unittest.TestCase):
    def test_encoder_produces_real_16khz_mono_wav(self):
        data = encode_wav(np.linspace(-0.25, 0.25, 16000, dtype=np.float32))
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 16000)

    def test_encoder_rejects_nonfinite_audio(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encode_wav(np.array([np.nan], dtype=np.float32))

    def test_multipart_contains_model_prompt_and_wav_bytes(self):
        payload, content_type = build_multipart(
            "ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0",
            "translate the speech to English.",
            encode_wav(np.ones(16000, dtype=np.float32) * 0.2),
        )
        self.assertIn(b'name="model"', payload)
        self.assertIn(b'name="prompt"', payload)
        self.assertIn(b'name="file"; filename="speech.wav"', payload)
        self.assertIn(b"translate the speech to English.", payload)
        self.assertIn(b"RIFF", payload)
        self.assertIn("multipart/form-data; boundary=", content_type)

    def test_plain_english_result_and_wrapper(self):
        self.assertEqual(_clean_result({"text": "EN: Get in the car.\n"}), "Get in the car.")
        with self.assertRaises(ValueError):
            _clean_result({"error": "unsupported model"})

    def test_model_is_direct_ast_not_japanese_asr(self):
        cfg = GraniteConfig()
        self.assertEqual(cfg.model_id, "ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0")
        self.assertIn("translate the speech to English", cfg.prompt)
        self.assertTrue(cfg.base_url.endswith(":9797"))

    def test_legacy_asr_and_translation_options_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                "[asr]\nhaas_model = 'legacy'\n"
                "[translate]\nplamo_model = 'legacy'\n"
                "[granite]\nrequest_timeout_s = 12.5\n",
                encoding="utf-8",
            )
            cfg = load_config(path)
        self.assertEqual(cfg.granite.request_timeout_s, 12.5)
        self.assertFalse(hasattr(cfg, "asr"))
        self.assertFalse(hasattr(cfg, "translate"))

    def test_probes_skip_silent_click_noise(self):
        silence = np.zeros(16000, dtype=np.float32)
        clicks = silence.copy()
        clicks[::800] = 0.95
        self.assertTrue(_discard_probe(silence))
        self.assertTrue(_discard_probe(clicks))


class FakeGranite:
    def __init__(self, output="They're coming."):
        self.output = output
        self.calls = 0

    def translate_audio(self, samples):
        self.calls += 1
        return self.output


class GranitePipelineTests(unittest.TestCase):
    @staticmethod
    def _utterance():
        return Utterance(1.0, 2.0, np.ones(16000, dtype=np.float32) * 0.04)

    def test_outputs_english_only_subtitle(self):
        qin, qout = queue.Queue(), queue.Queue()
        qin.put(self._utterance())
        qin.put(None)
        engine = FakeGranite("Get in the car!")
        run_granite(engine, qin, qout)
        sub = qout.get_nowait()
        self.assertIsInstance(sub, Subtitle)
        self.assertEqual(sub.en, "Get in the car!")
        self.assertEqual(sub.ja, "")
        self.assertEqual(engine.calls, 1)
        self.assertIsNone(qout.get_nowait())

    def test_progress_and_seeks_preserved_without_text_stage(self):
        qin, qout = queue.Queue(), queue.Queue()
        qin.put(AudioProgress(16000, generation=3, reset=True))
        qin.put(self._utterance())
        qin.put(None)
        engine = FakeGranite()
        run_granite(engine, qin, qout, get_generation=lambda: 3)
        reset = qout.get_nowait()
        self.assertIsInstance(reset, AudioProgress)
        self.assertTrue(reset.reset)
        self.assertIsNone(qout.get_nowait())
        self.assertEqual(engine.calls, 0)

    def test_vod_cached_english_skips_granite_decode(self):
        cache = SessionSubtitleCache()
        engine = FakeGranite()
        first_in, first_out = queue.Queue(), queue.Queue()
        first_in.put(self._utterance())
        first_in.put(None)
        run_granite(engine, first_in, first_out, session_cache=cache)
        first = first_out.get_nowait()
        cache.add(first)
        reused_in, reused_out = queue.Queue(), queue.Queue()
        reused_in.put(self._utterance())
        reused_in.put(None)
        run_granite(engine, reused_in, reused_out, session_cache=cache)
        second = reused_out.get_nowait()
        self.assertEqual(second.en, first.en)
        self.assertEqual(engine.calls, 1)

    def test_empty_translation_never_shown_as_subtitle(self):
        qin, qout = queue.Queue(), queue.Queue()
        qin.put(self._utterance())
        qin.put(None)
        run_granite(FakeGranite(""), qin, qout)
        self.assertIsNone(qout.get_nowait())


if __name__ == "__main__":
    unittest.main()
