"""CPU speaker turn and subtitle handoff tests (no model downloads)."""
import hashlib
import io
import queue
import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_translate.config import SpeakerConfig
from auto_translate.speaker import (
    SherpaSplitter, Turn, run_speaker, split_turns, _download,
)
from auto_translate.types import AudioProgress, SAMPLE_RATE, Utterance



class ModelDownloadTests(unittest.TestCase):
    def test_primary_standalone_file_download_and_cache_reuse(self):
        model = b"M" * 4096
        expected = hashlib.sha256(model).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segmentation.onnx"
            with patch("auto_translate.speaker.urllib.request.urlopen",
                       return_value=io.BytesIO(model)) as request:
                _download("https://huggingface.co/example/model.onnx", path,
                          fallback_url="https://github.com/example/backup.onnx",
                          expected_sha256=expected)
                _download("https://huggingface.co/example/model.onnx", path,
                          expected_sha256=expected)
            self.assertEqual(path.read_bytes(), model)
            self.assertEqual(request.call_count, 1)

    def test_github_fallback_after_two_huggingface_gateway_errors(self):
        model = b"F" * 4096
        expected = hashlib.sha256(model).hexdigest()
        attempts = []

        def urlopen(request, timeout):
            attempts.append(request.full_url)
            if "huggingface.co/" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 504,
                                             "Gateway Time-out", {}, None)
            return io.BytesIO(model)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.onnx"
            with patch("auto_translate.speaker.urllib.request.urlopen",
                       side_effect=urlopen), patch("auto_translate.speaker.time.sleep"):
                _download("https://huggingface.co/emb.onnx", path,
                          fallback_url="https://github.com/emb.onnx",
                          expected_sha256=expected)
            self.assertEqual(path.read_bytes(), model)
            self.assertEqual(len(attempts), 3)
            self.assertIn("github.com/", attempts[-1])

    def test_bad_cache_and_corrupt_download_cannot_be_used(self):
        valid = b"Z" * 4096
        expected = hashlib.sha256(valid).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.onnx"
            path.write_bytes(b"broken cache" * 300)
            with patch("auto_translate.speaker.urllib.request.urlopen",
                       return_value=io.BytesIO(valid)):
                _download("https://huggingface.co/emb.onnx", path,
                          expected_sha256=expected)
            self.assertEqual(path.read_bytes(), valid)
            path.unlink()
            # Return a NEW stream per retry: io.BytesIO is closed after
            # each attempt, just like a real HTTP response.
            with patch("auto_translate.speaker.urllib.request.urlopen",
                       side_effect=lambda *args, **kwargs: io.BytesIO(b"invalid" * 500)), patch(
                       "auto_translate.speaker.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    _download("https://huggingface.co/emb.onnx", path,
                              expected_sha256=expected)
            self.assertFalse(path.exists())

class SpeakerTurnTests(unittest.TestCase):
    def test_two_speakers_are_separate_nonoverlapping_windows(self):
        result = split_turns(4.0, [
            Turn(0, 1.9, 0), Turn(2.1, 4, 1),
        ], 0.55)
        self.assertEqual(result, [Turn(0, 2, 0), Turn(2, 4, 1)])
        self.assertEqual(sum(x.end - x.start for x in result), 4)

    def test_overlap_is_not_falsely_isolated(self):
        self.assertEqual(
            split_turns(4.0, [Turn(0, 2.5, 0), Turn(1.7, 4, 1)], 0.55),
            [Turn(0, 4, None)],
        )

    def test_separates_clear_changes_around_mixed_overlap(self):
        spans = split_turns(6.0, [
            Turn(0.0, 1.40, 0),
            Turn(1.65, 3.10, 1),
            Turn(2.70, 4.00, 2),
            Turn(4.30, 6.00, 3),
        ], .55)
        self.assertEqual(len(spans), 3)
        self.assertEqual([span.speaker for span in spans], [0, None, 3])
        self.assertAlmostEqual(spans[0].end, 1.525)
        self.assertAlmostEqual(spans[1].start, 1.525)
        self.assertAlmostEqual(spans[1].end, 4.15)
        self.assertAlmostEqual(spans[2].start, 4.15)
        self.assertAlmostEqual(sum(x.end - x.start for x in spans), 6.0)

    def test_tiny_speaker_fragment_is_not_cut(self):
        self.assertEqual(
            split_turns(4.0, [Turn(0, .2, 0), Turn(.25, 4, 1)], .55),
            [Turn(0, 4, None)],
        )

    def test_one_voice_and_unknown_preserve_original(self):
        self.assertEqual(
            split_turns(3.0, [Turn(.2, 2.8, 0)], .55),
            [Turn(0, 3, 0)],
        )
        self.assertEqual(split_turns(3.0, [], .55), [Turn(0, 3, None)])

    def test_boundaries_can_snap_to_real_quiet_gap(self):
        audio = np.full(4 * SAMPLE_RATE, .5, dtype=np.float32)
        audio[round(1.93 * SAMPLE_RATE):round(1.98 * SAMPLE_RATE)] = 0
        cut = split_turns(
            4.0, [Turn(0, 1.85, 0), Turn(2.17, 4.0, 1)],
            .55, audio,
        )[0].end
        self.assertGreaterEqual(cut, 1.93)
        self.assertLessEqual(cut, 1.98)


class SpeakerPipelineTests(unittest.TestCase):
    def test_fake_model_split_preserves_pcm_timestamps(self):
        class FakeDiarizer:
            def process(self, samples):
                self.samples = samples
                return self

            def sort_by_start_time(self):
                return [SimpleNamespace(start=0, end=1.9, speaker=0),
                        SimpleNamespace(start=2.1, end=4, speaker=1)]

        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = FakeDiarizer()
        backend.failures = 0
        pcm = np.concatenate((
            np.full(2 * SAMPLE_RATE, .3, dtype=np.float32),
            np.full(2 * SAMPLE_RATE, -.3, dtype=np.float32),
        ))
        original = Utterance(10.0, 14.0, pcm, generation=2,
                             speech_start_s=10.1)
        result = backend.split(original)
        self.assertEqual(len(result), 2)
        self.assertEqual([(u.start_s, u.end_s) for u in result],
                         [(10.0, 12.0), (12.0, 14.0)])
        self.assertEqual(sum(len(u.audio) for u in result), len(pcm))
        self.assertTrue(np.array_equal(
            np.concatenate([u.audio for u in result]), pcm,
        ))
        self.assertEqual([u.generation for u in result], [2, 2])
        self.assertEqual([u.speech_start_s for u in result], [10.1, 12.0])
        self.assertIsNot(result[0].timings, result[1].timings)
        self.assertFalse(hasattr(result[0], "speaker_id"))

    def test_short_windows_do_not_run_diarization(self):
        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = SimpleNamespace(
            process=lambda samples: (_ for _ in ()).throw(
                AssertionError("No model call for <1.65s utterances")
            )
        )
        backend.failures = 0
        src = Utterance(0, 1.4,
                        np.zeros(round(1.4 * SAMPLE_RATE), dtype=np.float32))
        self.assertEqual(backend.split(src), [src])

    def test_1_8_second_clean_turns_use_padding_and_conserve_pcm(self):
        class FakeDiarizer:
            def process(self, samples):
                self.samples = samples
                return self
            def sort_by_start_time(self):
                return [
                    SimpleNamespace(start=0.60, end=1.35, speaker=0),
                    SimpleNamespace(start=1.65, end=2.40, speaker=1),
                ]

        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = FakeDiarizer()
        backend.failures = 0
        pcm = np.concatenate([
            np.full(round(.9 * SAMPLE_RATE), .2, dtype=np.float32),
            np.full(round(.9 * SAMPLE_RATE), -.2, dtype=np.float32),
        ])
        source = Utterance(30.0, 31.8, pcm)
        result = backend.split(source)
        self.assertEqual(len(backend.diarizer.samples), 3 * SAMPLE_RATE)
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(len(part.audio) for part in result), len(pcm))
        self.assertTrue(np.array_equal(np.concatenate([part.audio for part in result]), pcm))

    def test_two_to_three_second_window_gets_symmetric_padding(self):
        class FakeDiarizer:
            def process(self, samples):
                self.samples = samples
                return self

            def sort_by_start_time(self):
                return [SimpleNamespace(start=.3, end=1.45, speaker=0),
                        SimpleNamespace(start=1.65, end=2.7, speaker=1)]

        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = FakeDiarizer()
        backend.failures = 0
        pcm = np.concatenate((
            np.full(round(1.2 * SAMPLE_RATE), .3, dtype=np.float32),
            np.full(round(1.2 * SAMPLE_RATE), -.3, dtype=np.float32),
        ))
        src = Utterance(40, 42.4, pcm, generation=1)
        result = backend.split(src)
        self.assertEqual(len(backend.diarizer.samples), 3 * SAMPLE_RATE)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.array_equal(
            np.concatenate([part.audio for part in result]), pcm,
        ))
        self.assertAlmostEqual(result[0].start_s, 40)
        self.assertAlmostEqual(result[-1].end_s, 42.4)

    def test_overlapping_model_turns_preserve_original_audio(self):
        class FakeDiarizer:
            def process(self, samples):
                return self

            def sort_by_start_time(self):
                return [SimpleNamespace(start=0, end=2.7, speaker=0),
                        SimpleNamespace(start=1.5, end=4, speaker=1)]

        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = FakeDiarizer()
        backend.failures = 0
        src = Utterance(0, 4, np.ones(4 * SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(backend.split(src), [src])

    def test_legacy_identity_config_is_ignored(self):
        from auto_translate.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[speaker]\nsimilarity_threshold = 0.68\nmax_speakers = 4\n"
                "enabled = true\nmin_turn_s = 0.65\n",
                encoding="utf-8",
            )
            cfg = load_config(path)
        self.assertTrue(cfg.speaker.enabled)
        self.assertEqual(cfg.speaker.min_turn_s, 0.65)
        self.assertFalse(hasattr(cfg.speaker, "similarity_threshold"))
        self.assertFalse(hasattr(cfg.speaker, "max_speakers"))

    def test_disabled_stage_passes_through_original(self):
        source, out = queue.Queue(), queue.Queue()
        original = Utterance(1, 2, np.zeros(SAMPLE_RATE, dtype=np.float32))
        source.put(original)
        source.put(AudioProgress(2 * SAMPLE_RATE, reset=True))
        source.put(None)
        run_speaker(SpeakerConfig(enabled=False), source, out)
        self.assertIs(out.get_nowait(), original)
        self.assertTrue(out.get_nowait().reset)
        self.assertIsNone(out.get_nowait())

    def test_failure_returns_original_audio(self):
        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = SimpleNamespace(process=lambda audio: 1 / 0)
        backend.failures = 0
        src = Utterance(0, 4, np.zeros(4 * SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(backend.split(src), [src])
        self.assertEqual(backend.failures, 1)
