"""Offline Whisper silence-hallucination tests (no model downloads)."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auto_translate.config import VadConfig, WhisperConfig
from auto_translate.two_stage import run_whisper_asr
from auto_translate.types import AudioChunk, Utterance
from auto_translate.vad import FRAME, Segmenter
from auto_translate.whisper_v3 import WhisperDecode, WhisperLargeV3


def run_asr(model, *utterances):
    source, target = queue.Queue(), queue.Queue()
    for item in utterances:
        source.put(item)
    source.put(None)
    run_whisper_asr(model, source, target)
    results = []
    while (item := target.get_nowait()) is not None:
        results.append(item)
    return results


def utterance():
    return Utterance(
        1.0, 2.0, np.full(16000, 0.03, dtype=np.float32),
        vad_confidence=0.55,
    )


class SilenceGuardTests(unittest.TestCase):
    def test_previous_text_cannot_be_reenabled_by_old_config(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                self.options = kwargs
                return iter([SimpleNamespace(
                    text="はい", avg_logprob=-0.2,
                    compression_ratio=1.1, no_speech_prob=0.03,
                )]), None

        fake = Model()
        asr = WhisperLargeV3(
            WhisperConfig(condition_on_previous_text=True), model=fake,
        )
        self.assertEqual(asr.decode_audio(utterance().audio).text, "はい")
        self.assertFalse(fake.options["condition_on_previous_text"])
        # FireRedVAD is the single pre-ASR VAD; do not double-filter softly
        # spoken Japanese using a second unrelated VAD.
        self.assertFalse(fake.options["vad_filter"])

    def test_rejected_window_keeps_whisper_silence_evidence(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                return iter([SimpleNamespace(
                    text="ありがとうございました",
                    avg_logprob=-1.5, compression_ratio=1.1,
                    no_speech_prob=0.96,
                )]), None

        decode = WhisperLargeV3(
            WhisperConfig(), model=Model(),
        ).decode_audio(utterance().audio)
        self.assertEqual(decode.text, "")
        self.assertEqual(decode.discarded_segments, 1)
        self.assertAlmostEqual(decode.no_speech_prob, 0.96)

    def test_high_confidence_silence_rejection_never_retries(self):
        class FakeWhisper:
            cfg = WhisperConfig()
            def __init__(self):
                self.calls = 0
            def decode_audio(self, audio, **kwargs):
                self.calls += 1
                return WhisperDecode(
                    "", discarded_segments=1,
                    no_speech_prob=0.96,
                )

        fake = FakeWhisper()
        self.assertEqual(run_asr(fake, utterance()), [])
        self.assertEqual(fake.calls, 1)

    def test_uncertain_quiet_speech_can_still_retry(self):
        class FakeWhisper:
            cfg = WhisperConfig()
            def __init__(self):
                self.calls = 0
            def decode_audio(self, audio, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return WhisperDecode(
                        "", discarded_segments=1,
                        no_speech_prob=0.25,
                    )
                return WhisperDecode(
                    "はい", avg_logprob=-0.2, no_speech_prob=0.04,
                )

        fake = FakeWhisper()
        results = run_asr(fake, utterance())
        self.assertEqual([item.ja for item in results], ["はい"])
        self.assertEqual(fake.calls, 2)

    def test_silence_gap_is_not_sent_to_whisper_and_keeps_timestamps(self):
        # Near-threshold but sustained speech is preserved on both sides
        # of a long gap; no extra fallback ASR probes come from the gap.
        probabilities = np.full(450, 0.01, dtype=np.float32)
        probabilities[70:115] = 0.54  # 450ms faint speech
        probabilities[285:340] = 0.55  # 550ms faint speech
        pcm = np.full(450 * FRAME, 0.03, dtype=np.float32)

        class FakeVAD:
            def reset(self):
                pass
            def __call__(self, audio):
                self.samples = len(audio)
                return probabilities

        with patch("auto_translate.vad.StreamingFireRedVAD",
                   return_value=FakeVAD()):
            segmenter = Segmenter(VadConfig())
            items = segmenter.feed(AudioChunk(0, pcm, 0.0))
            items.extend(segmenter.flush())

        self.assertEqual(len(items), 2)
        self.assertTrue(all(not item.probe for item in items))
        self.assertTrue(all(item.vad_confidence >= 0.5 for item in items))
        self.assertGreater(items[1].start_s - items[0].end_s, 1.0)
        self.assertLess(items[0].start_s, items[0].end_s)
        self.assertLess(items[1].start_s, items[1].end_s)


if __name__ == "__main__":
    unittest.main()
