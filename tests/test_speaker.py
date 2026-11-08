"""CPU speaker turn and subtitle handoff tests (no model downloads)."""
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.config import SpeakerConfig
from auto_translate.speaker import (
    SherpaSplitter, Turn, VoiceProfiles, run_speaker, split_turns,
)
from auto_translate.types import AudioProgress, SAMPLE_RATE, Utterance


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
        profiles = VoiceProfiles(0.72, 4)
        self.assertEqual(profiles.identify(np.array([1, 0])), 0)
        self.assertEqual(profiles.identify(np.array([0.96, 0.04])), 0)
        self.assertEqual(profiles.identify(np.array([0, 1])), 1)
        self.assertIsNone(profiles.identify(np.array([float("nan"), 0])))


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
        backend.profiles = VoiceProfiles(0.72, 4)
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
