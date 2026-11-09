"""Offline regressions for ReazonSpeech k2 v2 ASR on Windows.

Tests use fake sherpa recognizers and temporary model files, never download
weights or invoke FireRedVAD, PLaMo or MPV.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.asr import create_asr
from auto_translate.asr.reazon import (
    ReazonK2ASR, _FILES, _ensure_models,
)
from auto_translate.config import AsrConfig, load_config


class FakeStream:
    def __init__(self):
        self.accepted = None
        self.result = type("Result", (), {"text": "角巻わためです"})()

    def accept_waveform(self, sample_rate, data):
        self.accepted = (sample_rate, data.copy())


class FakeRecognizer:
    def __init__(self, result_text="角巻わためです"):
        self.calls = 0
        self.streams = []
        self.result_text = result_text

    def create_stream(self):
        s = FakeStream()
        s.result.text = self.result_text
        self.streams.append(s)
        return s

    def decode_stream(self, stream):
        self.calls += 1


class ReazonTests(unittest.TestCase):
    def test_defaults(self):
        cfg = AsrConfig()
        self.assertEqual(cfg.provider, "cpu")
        self.assertEqual(cfg.num_threads, 4)
        self.assertIn("reazonspeech-k2-v2", cfg.model_id)
        self.assertEqual(_FILES[0], "encoder-epoch-99-avg-1.int8.onnx")
        self.assertEqual(_FILES[1], "decoder-epoch-99-avg-1.onnx")
        self.assertEqual(_FILES[2], "joiner-epoch-99-avg-1.int8.onnx")

    def test_japanese_transcript_and_16khz_float(self):
        fake = FakeRecognizer()
        asr = ReazonK2ASR(AsrConfig(), recognizer=fake)
        audio = np.linspace(-2, 2, 16000, dtype=np.float64)
        self.assertEqual(asr.transcribe(audio), "角巻わためです")
        self.assertEqual(fake.calls, 1)
        rate, data = fake.streams[0].accepted
        self.assertEqual(rate, 16000)
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(data.flags.c_contiguous)
        self.assertLessEqual(float(data.max()), 1)
        self.assertGreaterEqual(float(data.min()), -1)

    def test_empty_silent_and_probe_audio_skipped(self):
        fake = FakeRecognizer()
        asr = ReazonK2ASR(AsrConfig(), recognizer=fake)
        self.assertEqual(asr.transcribe(np.zeros(0, dtype=np.float32)), "")
        self.assertEqual(asr.transcribe(np.zeros(16000, dtype=np.float32)), "")
        self.assertEqual(asr.transcribe(np.ones(16000, dtype=np.float32),
                                        probe=True), "")
        self.assertEqual(fake.calls, 0)

    def test_conservative_text_filter(self):
        asr = ReazonK2ASR(AsrConfig(), recognizer=FakeRecognizer("ははははは"))
        self.assertEqual(asr.transcribe(np.ones(16000, dtype=np.float32)), "")
        asr = ReazonK2ASR(
            AsrConfig(blocklist=["危険"]), recognizer=FakeRecognizer("危険ですよ"))
        self.assertEqual(asr.transcribe(np.ones(16000, dtype=np.float32)), "")

    def test_invalid_input_rejected(self):
        asr = ReazonK2ASR(AsrConfig(), recognizer=FakeRecognizer())
        with self.assertRaises(ValueError):
            asr.transcribe(np.array([np.nan], dtype=np.float32))
        with self.assertRaises(ValueError):
            asr.transcribe(np.ones(30 * 16000 + 1, dtype=np.float32))
        with self.assertRaises(ValueError):
            ReazonK2ASR(AsrConfig(provider="cuda"), recognizer=FakeRecognizer())
        with self.assertRaises(ValueError):
            ReazonK2ASR(AsrConfig(num_threads=0), recognizer=FakeRecognizer())

    def test_existing_files_skip_download(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in _FILES:
                (Path(temp) / name).write_bytes(b"test")
            cfg = AsrConfig(model_dir=temp)
            with patch("auto_translate.asr.reazon._MIN_BYTES", (2, 2, 2, 2)):
                with patch("huggingface_hub.snapshot_download") as download:
                    self.assertEqual(_ensure_models(cfg), Path(temp).resolve())
                    download.assert_not_called()

    def test_downloads_missing_files_from_pinned_official_source(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = AsrConfig(model_dir=temp)
            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], cfg.model_id)
                self.assertEqual(kwargs["revision"], cfg.model_revision)
                self.assertEqual(kwargs["allow_patterns"], list(_FILES))
                for name in _FILES:
                    (Path(kwargs["local_dir"]) / name).write_bytes(b"complete")
            with patch("auto_translate.asr.reazon._MIN_BYTES", (2, 2, 2, 2)):
                with patch("huggingface_hub.snapshot_download",
                           side_effect=download) as called:
                    self.assertEqual(_ensure_models(cfg), Path(temp).resolve())
                    called.assert_called_once()

    def test_partial_download_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("auto_translate.asr.reazon._MIN_BYTES", (10, 10, 10, 10)):
                with patch("huggingface_hub.snapshot_download",
                           return_value=temp):
                    with self.assertRaisesRegex(RuntimeError,
                                                "Missing or partial"):
                        _ensure_models(AsrConfig(model_dir=temp))

    def test_old_haas_config_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[asr]\nhaas_model = "haas-ja-v1.0"\n'
                'haas_device = "Vulkan0"\nnum_threads = 2\n',
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertEqual(cfg.asr.num_threads, 2)
            self.assertEqual(cfg.asr.provider, "cpu")
            self.assertEqual(cfg.vad.fallback_s, 0.0)

    def test_factory_uses_only_reazon_backend(self):
        fake = FakeRecognizer()
        with patch("auto_translate.asr.ReazonK2ASR",
                   return_value=ReazonK2ASR(AsrConfig(), recognizer=fake)) as cls:
            instance = create_asr(AsrConfig())
            self.assertIsInstance(instance, ReazonK2ASR)
            cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
