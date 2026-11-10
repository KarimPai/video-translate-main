"""Regressions for stock-thanks hallucinations without censoring real thanks."""
import queue
import unittest

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.pipeline import _is_stock_thanks, _stable_weak_speech, run_whisper
from auto_translate.types import AudioProgress, Subtitle, Utterance
from auto_translate.whisper_v3 import WhisperDecode


def utt(start, *, generation=0, vad=0.90, handoff=False):
    return Utterance(start, start + 1.0, np.full(16000, 0.10, dtype=np.float32),
                     generation=generation, vad_confidence=vad,
                     speaker_handoff=handoff)


class FakeDecoder:
    def __init__(self, outputs, cfg=None):
        self.cfg = cfg or WhisperConfig()
        self.outputs = iter(outputs)
        self.calls = 0

    def decode_audio(self, audio, **kwargs):
        self.calls += 1
        return next(self.outputs)


def output_for(decoder, *items):
    source, sink = queue.Queue(), queue.Queue()
    for item in items:
        source.put(item)
    source.put(None)
    run_whisper(decoder, source, sink)
    result = []
    while (item := sink.get_nowait()) is not None:
        if isinstance(item, Subtitle):
            result.append(item.en)
    return result


def decoded(text, *, lp=-0.2, ns=0.05):
    return WhisperDecode(text, avg_logprob=lp, no_speech_prob=ns)


class StockThanksTests(unittest.TestCase):
    def test_only_standalone_stock_phrases_are_candidates(self):
        for x in ("Thank you.", "THANK YOU VERY MUCH!", "Thanks a lot!"):
            self.assertTrue(_is_stock_thanks(x))
        for x in ("Thanks for helping", "Thank you, Watame", "I said thank you",
                  "Thank you very much for the rose"):
            self.assertFalse(_is_stock_thanks(x))

    def test_first_thanks_is_not_censored(self):
        model = FakeDecoder([decoded("Thank you very much.")])
        self.assertEqual(output_for(model, utt(1.0)), ["Thank you very much."])
        self.assertEqual(model.calls, 1)

    def test_repeated_weak_thanks_is_suppressed_after_retry(self):
        model = FakeDecoder([
            decoded("Thank you very much."),
            decoded("Thank you very much.", lp=-0.48, ns=0.23),
            decoded("Thank you very much.", lp=-0.60, ns=0.35),
        ])
        self.assertEqual(output_for(model, utt(1), utt(12)),
                         ["Thank you very much."])
        self.assertEqual(model.calls, 3)

    def test_repeated_confident_thanks_is_preserved(self):
        model = FakeDecoder([
            decoded("Thank you very much."),
            decoded("Thank you very much.", lp=-0.2, ns=0.04),
            decoded("Thank you.", lp=-0.18, ns=0.02),
        ])
        self.assertEqual(output_for(model, utt(1), utt(10)),
                         ["Thank you very much.", "Thank you."])
        self.assertEqual(model.calls, 3)

    def test_weak_rescue_never_promotes_generic_thanks(self):
        def weak(text):
            return WhisperDecode("", discarded_segments=1,
                                 weak_text=text, weak_avg_logprob=-1.35,
                                 weak_no_speech_prob=0.06)
        self.assertEqual(_stable_weak_speech(
            weak("Thank you very much."), weak("Thank you very much!"),
            0.90, WhisperConfig(),
        ), "")
        self.assertEqual(_stable_weak_speech(
            weak("Stop the car"), weak("Stop the car!"),
            0.90, WhisperConfig(),
        ), "Stop the car")

    def test_longer_conversational_thanks_unchanged(self):
        model = FakeDecoder([
            decoded("Thank you very much for the rose."),
            decoded("Thank you very much for the rose."),
        ])
        self.assertEqual(len(output_for(model, utt(1), utt(12))), 2)
        self.assertEqual(model.calls, 2)

    def test_seek_reset_clears_repetition_history(self):
        model = FakeDecoder([
            decoded("Thank you very much."),
            decoded("Thank you very much."),
        ])
        cues = output_for(model, utt(1), AudioProgress(0, reset=True, generation=1),
                          utt(1, generation=1))
        self.assertEqual(len(cues), 2)
        self.assertEqual(model.calls, 2)

    def test_new_speaker_does_not_require_repeat_confirmation(self):
        model = FakeDecoder([
            decoded("Thank you very much."),
            decoded("Thank you very much."),
        ])
        self.assertEqual(len(output_for(model, utt(1), utt(12, handoff=True))), 2)
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
