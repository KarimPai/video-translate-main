"""Regression checks for FireRedVAD CPU isolation and long-VOD seek reset.

Uses synthetic model doubles. No GPU, actual ASR model, or download required.
"""
from __future__ import annotations

import queue
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig, WhisperConfig
from auto_translate.types import AudioChunk, AudioProgress, SAMPLE_RATE
from auto_translate.vad import Segmenter, StreamingFireRedVAD
from auto_translate.anime_whisper import AnimeWhisperASR


class FakeConfig:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakeFireRed:
    last_config = None

    @classmethod
    def from_pretrained(cls, path, config):
        cls.last_config = config
        instance = cls()
        instance.reset_count = 0
        return instance

    def reset(self):
        self.reset_count += 1

    def detect_chunk(self, audio):
        # One frame per 160 samples after the 240-sample rolling history.
        count = (len(audio) - 240) // 160
        return [SimpleNamespace(raw_prob=0.05) for _ in range(count)]


class FireRedSeekIsolationTests(unittest.TestCase):
    def test_cpu_device_is_default_in_experimental_branch(self):
        self.assertEqual(VadConfig().device, "cpu")
        self.assertEqual(WhisperConfig().device, "cuda")

    def test_cpu_vad_initializes_without_cuda_runtime_calls(self):
        cuda = SimpleNamespace(
            is_available=lambda: self.fail("CPU VAD queried CUDA availability"),
            get_device_name=lambda _i: self.fail("CPU VAD accessed CUDA device"),
        )
        fake_packages = {
            "torch": SimpleNamespace(cuda=cuda),
            "fireredvad": SimpleNamespace(
                FireRedStreamVad=FakeFireRed,
                FireRedStreamVadConfig=FakeConfig,
            ),
            "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: None),
        }
        with patch.dict(sys.modules, fake_packages):
            vad = StreamingFireRedVAD(0.5, device="cpu")
            self.assertFalse(FakeFireRed.last_config.options["use_gpu"])
            self.assertEqual(vad.device, "CPU")
            vad.reset()
            self.assertEqual(vad._stream.reset_count, 1)
            probabilities = vad(np.zeros(1600, dtype=np.float32))
            self.assertEqual(probabilities.shape, (10,))

    def test_seek_to_five_hours_keeps_cpu_vad_model_instance(self):
        class VADStub:
            def __init__(self, threshold, device="cpu"):
                self.device = device
                self.resets = 0

            def reset(self):
                self.resets += 1

            def __call__(self, audio):
                return np.zeros(len(audio) // 160, dtype=np.float32)

        with patch("auto_translate.vad.StreamingFireRedVAD", VADStub):
            seg = Segmenter(VadConfig())
            original_model = seg.vad
            seek_sample = 5 * 3600 * SAMPLE_RATE
            marker = AudioChunk(
                seek_sample, np.empty(0, dtype=np.float32),
                0.0, reset=True, generation=1,
            )
            self.assertEqual(seg.feed(marker), [])
            self.assertIs(seg.vad, original_model)
            self.assertEqual(seg.vad.resets, 2)
            self.assertEqual(seg.vad.device, "cpu")
            self.assertEqual(seg._generation, 1)
            self.assertEqual(seg._next_frame_sample, seek_sample)
            normal_audio = AudioChunk(
                seek_sample, np.zeros(1600, dtype=np.float32),
                0.1, generation=1,
            )
            self.assertEqual(seg.feed(normal_audio), [])
            self.assertEqual(seg._next_frame_sample, seek_sample + 1600)
            self.assertEqual(
                seg.advance_progress(seg._next_frame_sample, force=True).sample,
                seek_sample + 1600,
            )

    def test_legacy_cuda_mode_is_explicit_opt_in(self):
        # This simply validates configuration, without loading GPU libraries.
        self.assertEqual(VadConfig(device="cuda").device, "cuda")

    def test_asr_shutdown_does_not_call_cuda_empty_cache(self):
        # No new GPU command must be issued after a sticky device-side assert.
        asr = AnimeWhisperASR.__new__(AnimeWhisperASR)
        asr._model = object()
        asr._processor = object()
        asr.device = "cuda"
        asr.close()
        self.assertIsNone(asr._model)
        self.assertIsNone(asr._processor)


if __name__ == "__main__":
    unittest.main()
