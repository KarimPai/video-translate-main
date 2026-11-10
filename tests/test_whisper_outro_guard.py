"""Focused regression tests for generic Whisper signoff hallucinations."""
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.pipeline import (
    _is_stock_outro, _japanese_supports_outro, run_whisper,
)
from auto_translate.types import Utterance
from auto_translate.whisper_v3 import WhisperLargeV3


def run_one(english: str, japanese: str, *, verify=True):
    class FakeModel:
        def __init__(self):
            self.tasks = []

        def transcribe(self, audio, **options):
            self.tasks.append(options)
            text = japanese if options["task"] == "transcribe" else english
            segment = SimpleNamespace(
                text=text, avg_logprob=-0.2,
                no_speech_prob=0.03, compression_ratio=1.0,
            )
            return iter([segment]), None

    model = FakeModel()
    decoder = WhisperLargeV3(
        WhisperConfig(verify_stock_outros=verify), model=model,
    )
    source, target = queue.Queue(), queue.Queue()
    source.put(Utterance(
        10.0, 11.0, np.full(16000, 0.05, dtype=np.float32),
        vad_confidence=0.88,
    ))
    source.put(None)
    run_whisper(decoder, source, target)
    result = []
    while (item := target.get_nowait()) is not None:
        result.append(item.en)
    return result, model.tasks


class WhisperOutroGuardTests(unittest.TestCase):
    def test_only_generic_full_sentence_matches(self):
        self.assertTrue(_is_stock_outro("Thank you for watching!"))
        self.assertTrue(_is_stock_outro("Thanks for watching."))
        self.assertFalse(_is_stock_outro("Thank you, Watame!"))
        self.assertFalse(_is_stock_outro("Watch out!"))
        self.assertFalse(_is_stock_outro(
            "Thank you for watching the road!"
        ))

    def test_japanese_requires_watching_and_thanks(self):
        self.assertTrue(_japanese_supports_outro("ご視聴ありがとうございました"))
        self.assertTrue(_japanese_supports_outro("見てくれてありがとう"))
        self.assertFalse(_japanese_supports_outro("ありがとう"))
        self.assertFalse(_japanese_supports_outro("今日はゲームをします"))
        self.assertFalse(_japanese_supports_outro("見るのは難しい"))

    def test_unverified_outro_not_published(self):
        output, calls = run_one(
            "Thank you for watching.", "今日は車で行きましょう",
        )
        self.assertEqual(output, [])
        self.assertEqual([x["task"] for x in calls],
                         ["translate", "transcribe"])
        self.assertFalse(calls[1]["condition_on_previous_text"])
        self.assertIsNone(calls[1]["initial_prompt"])
        self.assertIsNone(calls[1]["hotwords"])

    def test_genuine_japanese_outro_still_published(self):
        output, calls = run_one(
            "Thank you for watching!", "ご視聴ありがとうございました",
        )
        self.assertEqual(output, ["Thank you for watching!"])
        self.assertEqual(len(calls), 2)

    def test_generic_thanks_are_not_filtered(self):
        output, calls = run_one("Thank you!", "ありがとう")
        self.assertEqual(output, ["Thank you!"])
        self.assertEqual(len(calls), 1)

    def test_toggle_disables_extra_japanese_pass(self):
        output, calls = run_one(
            "Thank you for watching.", "今日はゲームをします",
            verify=False,
        )
        self.assertEqual(output, ["Thank you for watching."])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
