"""Whisper AST-only offline contract tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auto_translate.config import WhisperConfig, load_config
from auto_translate.whisper_v3 import WhisperLargeV3


class DirectAstTests(unittest.TestCase):
    def test_stale_config_cannot_reenable_separate_translator(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text(
                '[whisper]\ntask="transcribe"\n'
                '[translate]\nplamo_context=4096\n',
                encoding="utf-8",
            )
            cfg = load_config(path)
        self.assertEqual(cfg.whisper.task, "translate")
        self.assertFalse(hasattr(cfg, "translate"))

    def test_direct_audio_translation_only(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                self.options = kwargs
                return iter([SimpleNamespace(
                    text="Get in the car!", avg_logprob=-0.2,
                    no_speech_prob=0.02, compression_ratio=1.1,
                )]), None

        fake = Model()
        whisper = WhisperLargeV3(WhisperConfig(), model=fake)
        self.assertEqual(
            whisper.translate_audio(np.full(16000, 0.1, np.float32)),
            "Get in the car!",
        )
        self.assertEqual(fake.options["language"], "ja")
        self.assertEqual(fake.options["task"], "translate")
        self.assertFalse(fake.options["condition_on_previous_text"])
        self.assertFalse(fake.options["vad_filter"])

    def test_transcribe_is_not_accepted_as_task(self):
        with self.assertRaisesRegex(ValueError, "task='translate'"):
            WhisperLargeV3(WhisperConfig(task="transcribe"), model=object())


if __name__ == "__main__":
    unittest.main()
