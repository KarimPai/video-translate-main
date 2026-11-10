"""Whisper Large-v3 Japanese audio -> English direct translation regressions."""

import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import WhisperConfig, load_config
from auto_translate.pipeline import run_whisper, _discard_probe
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import AudioProgress, Subtitle, Utterance
from auto_translate.whisper_v3 import WhisperLargeV3


class FakeModel:
    def __init__(self, texts=("Get in the car!",)):
        self.texts = texts
        self.calls = 0
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        self.audio = audio.copy()
        return (SimpleNamespace(text=s) for s in self.texts), SimpleNamespace(language="ja")


class WhisperDirectTests(unittest.TestCase):
    def test_repetitive_decoding_is_bounded(self):
        fake = FakeModel()
        decoder = WhisperLargeV3(WhisperConfig(), model=fake)
        decoder.translate_audio(np.ones(16000, dtype=np.float32))
        self.assertEqual(fake.kwargs["no_repeat_ngram_size"], 3)
        self.assertAlmostEqual(fake.kwargs["repetition_penalty"], 1.05)
        self.assertEqual(fake.kwargs["max_new_tokens"], 160)

    def test_final_low_confidence_is_rejected(self):
        class ScoredModel:
            def transcribe(self, audio, **kwargs):
                parts = [
                    SimpleNamespace(text="Thank you for watching.",
                                    avg_logprob=-1.5,
                                    compression_ratio=1.0,
                                    no_speech_prob=0.8),
                    SimpleNamespace(text="Watch out!",
                                    avg_logprob=-0.2,
                                    compression_ratio=1.1,
                                    no_speech_prob=0.1),
                ]
                return iter(parts), SimpleNamespace(language="ja")

        decoder = WhisperLargeV3(WhisperConfig(), model=ScoredModel())
        self.assertEqual(
            decoder.translate_audio(np.ones(16000, dtype=np.float32)),
            "Watch out!",
        )

    def test_high_compression_and_noise_guard(self):
        decoder = WhisperLargeV3(WhisperConfig(), model=FakeModel())
        self.assertTrue(decoder._unreliable(SimpleNamespace(
            avg_logprob=-0.2, compression_ratio=3.0, no_speech_prob=0.1,
        )))
        self.assertTrue(decoder._unreliable(SimpleNamespace(
            avg_logprob=-0.6, compression_ratio=1.0, no_speech_prob=0.8,
        )))
        self.assertFalse(decoder._unreliable(SimpleNamespace(
            avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.8,
        )))
        self.assertFalse(decoder._unreliable(SimpleNamespace(text="No scores")))

    def test_repeated_outros_suppressed_without_banning_other_thanks(self):
        class SequenceModel:
            def __init__(self):
                self.outputs = iter([
                    "Thank you for watching.", "Thank you for watching until the end.",
                    "Thank you!", "Thanks for watching!", "Let's go.",
                ])
            def translate_audio(self, audio):
                return next(self.outputs)

        source, result = queue.Queue(), queue.Queue()
        for i in range(5):
            source.put(Utterance(i * 30.0, i * 30.0 + 1.0,
                                 np.ones(16000, dtype=np.float32)))
        source.put(None)
        run_whisper(SequenceModel(), source, result)
        outputs = []
        while (item := result.get_nowait()) is not None:
            outputs.append(item.en)
        self.assertEqual(outputs, ["Thank you!", "Let's go."])

    def test_outro_suppression_resets_on_new_seek_generation(self):
        class AlwaysOutro:
            def translate_audio(self, audio):
                return "Thank you for watching."

        source, result = queue.Queue(), queue.Queue()
        source.put(Utterance(1, 2, np.ones(16000, dtype=np.float32),
                             generation=0))
        source.put(AudioProgress(0, reset=True, generation=1))
        source.put(Utterance(1, 2, np.ones(16000, dtype=np.float32),
                             generation=1))
        source.put(None)
        run_whisper(AlwaysOutro(), source, result)
        self.assertTrue(result.get_nowait().reset)
        self.assertIsNone(result.get_nowait())

    def test_defaults_are_large_v3_cuda_translate(self):
        cfg = WhisperConfig()
        self.assertEqual((cfg.model, cfg.device, cfg.compute_type),
                         ("large-v3", "cuda", "int8_float16"))
        self.assertEqual((cfg.language, cfg.task, cfg.beam_size),
                         ("ja", "transcribe", 5))

    def test_source_language_and_task_forced(self):
        fake = FakeModel(("Get in the car!", "Watch out!"))
        decoder = WhisperLargeV3(WhisperConfig(), model=fake)
        result = decoder.translate_audio(np.ones(16000, dtype=np.float32))
        self.assertEqual(result, "Get in the car! Watch out!")
        self.assertEqual(fake.calls, 1)
        self.assertEqual(fake.audio.dtype, np.float32)
        self.assertEqual(fake.kwargs["language"], "ja")
        self.assertEqual(fake.kwargs["task"], "transcribe")
        self.assertEqual(fake.kwargs["beam_size"], 5)
        self.assertEqual(fake.kwargs["temperature"], 0.0)
        self.assertFalse(fake.kwargs["condition_on_previous_text"])
        self.assertFalse(fake.kwargs["vad_filter"])
        self.assertIsNone(fake.kwargs["initial_prompt"])
        self.assertIsNone(fake.kwargs["hotwords"])

    def test_model_loader_uses_cuda_precision(self):
        mock_class = Mock(return_value=FakeModel())
        with patch.dict("sys.modules", {"faster_whisper":
                         SimpleNamespace(WhisperModel=mock_class)}):
            decoder = WhisperLargeV3(WhisperConfig())
        self.assertEqual(mock_class.call_args.args[0], "large-v3")
        self.assertEqual(mock_class.call_args.kwargs["device"], "cuda")
        self.assertEqual(mock_class.call_args.kwargs["compute_type"], "int8_float16")
        decoder.close()

    def test_no_translation_output_if_whisper_returns_empty(self):
        model = WhisperLargeV3(WhisperConfig(), model=FakeModel(()))
        self.assertEqual(model.translate_audio(np.ones(16000, dtype=np.float32)), "")

    def test_reject_empty_and_nonfinite_audio(self):
        fake = FakeModel()
        model = WhisperLargeV3(WhisperConfig(), model=fake)
        with self.assertRaisesRegex(ValueError, "empty"):
            model.translate_audio(np.array([], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.translate_audio(np.array([np.nan], dtype=np.float32))
        self.assertEqual(fake.calls, 0)

    def test_cannot_switch_source_language_or_task(self):
        with self.assertRaisesRegex(ValueError, "language="):
            WhisperLargeV3(WhisperConfig(language="en"), model=FakeModel())
        with self.assertRaisesRegex(ValueError, "task="):
            WhisperLargeV3(WhisperConfig(task="translate"), model=FakeModel())

    def test_legacy_granite_config_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                "[granite]\nmodel_id='obsolete'\n"
                "[translate]\nmodel='obsolete'\n"
                "[whisper]\nbeam_size=4\n",
                encoding="utf-8",
            )
            cfg = load_config(path)
        self.assertEqual(cfg.whisper.beam_size, 4)
        self.assertFalse(hasattr(cfg, "granite"))

    def test_pipeline_produces_english_only_and_caches(self):
        fake = FakeModel()
        decoder = WhisperLargeV3(WhisperConfig(), model=fake)
        cache = SessionSubtitleCache()
        for n in range(2):
            src, dest = queue.Queue(), queue.Queue()
            src.put(Utterance(1, 2, np.ones(16000, dtype=np.float32)))
            src.put(None)
            run_whisper(decoder, src, dest, session_cache=cache)
            item = dest.get_nowait()
            self.assertIsInstance(item, Subtitle)
            self.assertEqual(item.ja, "")
            self.assertEqual(item.en, "Get in the car!")
            self.assertIsNone(dest.get_nowait())
            if n == 0:
                cache.add(item)
        self.assertEqual(fake.calls, 1)

    def test_pipeline_discards_stale_seek_audio(self):
        fake = FakeModel()
        decoder = WhisperLargeV3(WhisperConfig(), model=fake)
        src, dest = queue.Queue(), queue.Queue()
        src.put(AudioProgress(16000, reset=True, generation=3))
        src.put(Utterance(1, 2, np.ones(16000, dtype=np.float32), generation=0))
        src.put(None)
        run_whisper(decoder, src, dest, get_generation=lambda: 3)
        self.assertTrue(dest.get_nowait().reset)
        self.assertIsNone(dest.get_nowait())
        self.assertEqual(fake.calls, 0)

    def test_probe_filter_retains_existing_noise_rejection(self):
        silent = np.zeros(16000, dtype=np.float32)
        click = silent.copy()
        click[::800] = .95
        self.assertTrue(_discard_probe(silent))
        self.assertTrue(_discard_probe(click))


if __name__ == "__main__":
    unittest.main()
