"""Conservative engine-noise filtering without eliminating real acknowledgements."""
import queue
import unittest

import numpy as np

from auto_translate.captions import CaptionCue, CaptionIndex
from auto_translate.config import AsrConfig
from auto_translate.pipeline import run_asr
from auto_translate.text_filter import (
    caption_confirms_short_acknowledgement,
    is_nonspeech_probe,
    is_short_acknowledgement,
    is_stationary_mechanical_audio,
    strip_nonverbal_labels,
)
from auto_translate.types import Transcript, Utterance


def stationary_engine(seconds=1.5):
    """Stable multi-harmonic 110 Hz motor signal (not a single pure tone)."""
    t = np.arange(int(16000 * seconds), dtype=np.float32) / 16000
    return sum(np.sin(2 * np.pi * (110 * n) * t) / n
               for n in range(1, 7)).astype(np.float32) * 0.15


def voiced_dynamic(seconds=1.5):
    """Synthetic time-varying voiced acoustic control, not a speech recording."""
    t = np.arange(int(16000 * seconds), dtype=np.float32) / 16000
    phase = 2 * np.pi * np.cumsum(125 + 10 * np.sin(2 * np.pi * 4 * t)) / 16000
    amp = .25 + .5 * np.sin(np.pi * t / seconds) ** 2
    return (amp * (np.sin(phase) + .5 * np.sin(2 * phase)
                   + .45 * np.sin(6 * phase))).astype(np.float32) * .15


class TextASR:
    def __init__(self, text, *, guard=True):
        self.text = text
        self.calls = 0
        self.cfg = AsrConfig(engine_noise_ack_guard=guard)

    def transcribe(self, audio, probe=False):
        self.calls += 1
        return self.text


def transcribe(audio, text, *, probe=False, captions=None, guard=True):
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(Utterance(1.0, 1.0 + len(audio) / 16000, audio, probe=probe))
    incoming.put(None)
    asr = TextASR(text, guard=guard)
    run_asr(asr, incoming, outgoing, captions=captions)
    produced = []
    while (item := outgoing.get_nowait()) is not None:
        produced.append(item)
    return produced, asr.calls


class EngineNoiseGuardTests(unittest.TestCase):
    def test_engine_harmonics_are_detected_not_just_a_sine_wave(self):
        engine = stationary_engine()
        self.assertTrue(is_stationary_mechanical_audio(engine))
        self.assertTrue(is_nonspeech_probe(engine))

    def test_varying_voiced_signal_not_classified_as_engine(self):
        voiced = voiced_dynamic()
        self.assertFalse(is_stationary_mechanical_audio(voiced))
        self.assertFalse(is_nonspeech_probe(voiced))

    def test_engine_sound_umm_never_reaches_translation(self):
        produced, calls = transcribe(stationary_engine(), "うん。")
        self.assertEqual(calls, 1)
        self.assertEqual(produced, [])

    def test_ordinary_short_reply_still_allowed(self):
        produced, calls = transcribe(voiced_dynamic(), "うん。")
        self.assertEqual(calls, 1)
        self.assertEqual(len(produced), 1)
        self.assertIsInstance(produced[0], Transcript)
        self.assertEqual(produced[0].ja, "うん。")

    def test_utterance_with_real_words_not_suppressed(self):
        produced, _ = transcribe(stationary_engine(), "うん、車を見に行く")
        self.assertEqual([x.ja for x in produced], ["うん、車を見に行く"])

    def test_vad_negative_probe_only_short_ack_is_dropped(self):
        audio = np.random.default_rng(13).normal(0, .12, 16000).astype(np.float32)
        produced, calls = transcribe(audio, "うん。", probe=True)
        self.assertEqual(calls, 1)
        self.assertEqual(produced, [])

    def test_reliable_same_caption_protects_real_short_reply(self):
        engine = stationary_engine()
        idx = CaptionIndex([CaptionCue(1.0, 2.5, "うん。")])
        produced, calls = transcribe(engine, "うん。", captions=idx)
        self.assertEqual(calls, 1)
        self.assertEqual([x.ja for x in produced], ["うん。"])

    def test_unrelated_caption_cannot_validate_engine_umm(self):
        engine = stationary_engine()
        idx = CaptionIndex([CaptionCue(1.0, 2.5, "車を修理します")])
        produced, _ = transcribe(engine, "うん。", captions=idx)
        # A strong reliable, conflicting caption can replace ASR based on
        # ordinary caption reconciliation, so this test targets the guard
        # itself rather than assuming the reconciliation policy.
        cue = idx.match(1, 2.5)
        self.assertFalse(caption_confirms_short_acknowledgement("うん。", cue))

    def test_opt_out_available_for_diagnostics(self):
        produced, _ = transcribe(stationary_engine(), "うん。", guard=False)
        self.assertEqual([x.ja for x in produced], ["うん。"])

    def test_engine_caption_labels_not_treated_as_speech(self):
        self.assertEqual(strip_nonverbal_labels("[エンジン音]"), "")
        self.assertEqual(strip_nonverbal_labels("[engine noise]"), "")
        self.assertEqual(strip_nonverbal_labels("エンジン音"), "")
        self.assertEqual(strip_nonverbal_labels("[エンジン音] 今行きます"), "今行きます")

    def test_short_ack_scope(self):
        self.assertTrue(is_short_acknowledgement("うん。"))
        self.assertFalse(is_short_acknowledgement("うん、そうだね"))
        self.assertFalse(is_short_acknowledgement("ありがとう"))


if __name__ == "__main__":
    unittest.main()
