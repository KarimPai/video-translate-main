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
    SherpaSplitter, Turn, VoiceProfiles, run_speaker, split_turns, _download,
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
            with patch("auto_translate.speaker.urllib.request.urlopen",
                       return_value=io.BytesIO(b"invalid" * 500)), patch(
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

    def test_true_overlap_is_not_falsely_isolated(self):
        result = split_turns(4.0, [
            Turn(0, 2.5, 0), Turn(1.7, 4, 1),
        ], 0.55)
        self.assertEqual(result, [Turn(0, 4, None)])

    def test_tiny_speaker_fragment_is_not_cut(self):
        result = split_turns(4.0, [
            Turn(0, 0.2, 0), Turn(0.25, 4, 1),
        ], 0.55)
        self.assertEqual(result, [Turn(0, 4, None)])

    def test_single_voice_retains_whole_original_cue(self):
        self.assertEqual(split_turns(2.0, [Turn(0.2, 1.8, 0)], 0.55),
                         [Turn(0, 2, 0)])
        self.assertEqual(split_turns(2.0, [], 0.55),
                         [Turn(0, 2, None)])

    def test_anonymous_voice_embeddings_reuse_identity(self):
        profiles = VoiceProfiles(0.68, 4)
        self.assertIsNone(profiles.identify(np.array([1, 0])))
        self.assertEqual(profiles.identify(np.array([0.96, 0.04])), 0)
        self.assertEqual(profiles.identify(np.array([1, 0.04])), 0)
        self.assertIsNone(profiles.identify(np.array([0, 1])))
        self.assertEqual(profiles.identify(np.array([0, 0.99])), 1)
        self.assertIsNone(profiles.identify(np.array([float("nan"), 0])))

    def test_same_speaker_variations_reuse_anonymous_label(self):
        profiles = VoiceProfiles(0.68, 4)
        self.assertIsNone(profiles.identify(np.array([1, 0, 0])))
        self.assertEqual(profiles.identify(np.array([.97, .24, 0])), 0)
        for variant in ([.90, .42, 0], [.86, .50, 0], [1, .03, 0],
                        [.82, .56, 0], [.92, .39, 0]):
            self.assertEqual(profiles.identify(np.array(variant)), 0)
        self.assertEqual(len(profiles.vectors), 1)
        self.assertLessEqual(len(profiles._examples[0]), 6)

    def test_no_unbounded_letter_creation_for_new_or_noisy_voices(self):
        profiles = VoiceProfiles(0.68, 2)
        self.assertIsNone(profiles.identify(np.array([1., 0., 0.])))
        self.assertEqual(profiles.identify(np.array([.99, .02, 0.])), 0)
        self.assertIsNone(profiles.identify(np.array([0., 1., 0.])))
        self.assertEqual(profiles.identify(np.array([0., .99, 0.])), 1)
        for variant in ([0., 0., 1.], [.4, 0., .9], [-1., 0., 0.],
                        [-.2, -.2, .95]):
            self.assertIsNone(profiles.identify(np.array(variant)))
        self.assertEqual(len(profiles.vectors), 2)

    def test_short_unfamiliar_speech_never_registers_new_label(self):
        profiles = VoiceProfiles(0.68, 4)
        for _ in range(8):
            self.assertIsNone(profiles.identify(np.array([1., 0., 0.]),
                                                duration_s=0.75))
        self.assertEqual(len(profiles.vectors), 0)
        self.assertEqual(len(profiles._pending), 0)
        self.assertIsNone(profiles.identify(np.array([1., 0., 0.])))
        self.assertEqual(profiles.identify(np.array([1., 0., 0.])), 0)
        self.assertEqual(profiles.identify(np.array([1., 0., 0.]),
                                           duration_s=0.8), 0)

    def test_ties_between_known_speakers_are_unknown(self):
        profiles = VoiceProfiles(0.65, 4)
        for variant in ([1., 0.], [.99, .05], [0., 1.], [.05, .99]):
            profiles.identify(np.array(variant))
        self.assertEqual(len(profiles.vectors), 2)
        self.assertIsNone(profiles.identify(np.array([.707, .707])))

    def test_identity_memory_survives_seek_replay(self):
        profiles = VoiceProfiles(0.68, 4)
        self.assertIsNone(profiles.identify(np.array([1., 0.])))
        self.assertEqual(profiles.identify(np.array([1., .02])), 0)
        self.assertEqual(profiles.identify(np.array([1., .01])), 0)
        self.assertEqual(len(profiles.vectors), 1)


class SpeakerPipelineTests(unittest.TestCase):
    def test_fake_model_split_preserves_pcm_timestamps_and_distinct_ids(self):
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
        backend.profiles = VoiceProfiles(0.68, 4)
        # Seed two already-confirmed voices for independent-turn testing.
        for voice in (np.array([1., 0.]), np.array([0., 1.])):
            self.assertIsNone(backend.profiles.identify(voice))
            self.assertIsNotNone(backend.profiles.identify(voice))
        backend.failures = 0
        backend._embedding = lambda samples: np.array(
            [1.0, 0.0] if float(np.mean(samples)) > 0 else [0.0, 1.0],
            dtype=np.float32,
        )
        pcm = np.concatenate((
            np.ones(2 * SAMPLE_RATE, dtype=np.float32) * .3,
            np.ones(2 * SAMPLE_RATE, dtype=np.float32) * -.3,
        ))
        utterance = Utterance(10.0, 14.0, pcm, generation=2, speech_start_s=10.1)
        split = backend.split(utterance)
        self.assertEqual(len(split), 2)
        self.assertEqual([(u.start_s, u.end_s, u.speaker_id) for u in split],
                         [(10.0, 12.0, 0), (12.0, 14.0, 1)])
        self.assertEqual(sum(len(u.audio) for u in split), len(pcm))
        self.assertTrue(np.array_equal(np.concatenate([u.audio for u in split]), pcm))
        self.assertEqual([u.generation for u in split], [2, 2])
        self.assertEqual([u.speech_start_s for u in split], [10.1, 12.0])
        self.assertIsNot(split[0].timings, split[1].timings)

    def test_disabled_stage_preserves_original_utterance_and_seek_marker(self):
        source, result = queue.Queue(), queue.Queue()
        utterance = Utterance(1, 2, np.zeros(SAMPLE_RATE, dtype=np.float32))
        source.put(utterance)
        source.put(AudioProgress(2 * SAMPLE_RATE, reset=True))
        source.put(None)
        run_speaker(SpeakerConfig(enabled=False), source, result)
        self.assertIs(result.get_nowait(), utterance)
        self.assertTrue(result.get_nowait().reset)
        self.assertIsNone(result.get_nowait())

    def test_failure_returns_original_audio(self):
        backend = object.__new__(SherpaSplitter)
        backend.cfg = SpeakerConfig()
        backend.diarizer = SimpleNamespace(process=lambda audio: 1 / 0)
        backend.failures = 0
        pcm = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        utt = Utterance(0, 4, pcm)
        self.assertEqual(backend.split(utt), [utt])
        self.assertEqual(backend.failures, 1)
