"""Focused regression tests for generic Whisper signoff hallucinations."""
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from auto_translate.config import WhisperConfig
from auto_translate.pipeline import (
    _is_stock_outro, _japanese_supports_outro, _phantom_phrase_kind,
    _phantom_timing_implausible, _japanese_supports_phantom, run_whisper,
)
from auto_translate.types import Utterance
from auto_translate.whisper_v3 import WhisperLargeV3


def run_one(english: str, japanese: str, *, verify=True,
            filter_phantoms=True, duration_s=1.0, vad_confidence=0.88,
            no_speech_prob=0.03, avg_logprob=-0.2):
    class FakeModel:
        def __init__(self):
            self.tasks = []

        def transcribe(self, audio, **options):
            self.tasks.append(options)
            text = japanese if options["task"] == "transcribe" else english
            segment = SimpleNamespace(
                text=text, avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob, compression_ratio=1.0,
            )
            return iter([segment]), None

    model = FakeModel()
    decoder = WhisperLargeV3(
        WhisperConfig(verify_stock_outros=verify,
                      filter_phantom_phrases=filter_phantoms), model=model,
    )
    source, target = queue.Queue(), queue.Queue()
    source.put(Utterance(
        10.0, 10.0 + duration_s,
        np.full(round(16000 * duration_s), 0.05, dtype=np.float32),
        vad_confidence=vad_confidence,
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

    def test_full_cue_blocklist_matches_only_suspect_phrases(self):
        self.assertEqual(_phantom_phrase_kind("Subtitles by Amara.org."), "credits")
        self.assertEqual(_phantom_phrase_kind("Subtitled by..."), "credits")
        self.assertEqual(_phantom_phrase_kind("Please subscribe to my channel!"),
                         "subscribe")
        self.assertEqual(_phantom_phrase_kind("Like and subscribe!"), "subscribe")
        self.assertEqual(_phantom_phrase_kind("Thank you for watching!"), "outro")
        self.assertEqual(_phantom_phrase_kind("Subscribe to the train tickets"), "")
        self.assertEqual(_phantom_phrase_kind("We need subtitles for this"), "")
        self.assertEqual(_phantom_phrase_kind("Watame said subscribe to my channel"), "")

    def test_japanese_corroboration_by_phrase_type(self):
        self.assertTrue(_japanese_supports_phantom(
            "credits", "字幕は私が作成しました"))
        self.assertFalse(_japanese_supports_phantom(
            "credits", "字幕が表示されない"))
        self.assertTrue(_japanese_supports_phantom(
            "subscribe", "チャンネル登録よろしくお願いします"))
        self.assertFalse(_japanese_supports_phantom(
            "subscribe", "ゲームに登録できました"))

    def test_credits_without_japanese_evidence_suppressed(self):
        output, calls = run_one("Subtitles by Amara.org", "今日はゲームをします")
        self.assertEqual(output, [])
        self.assertEqual(len(calls), 2)

    def test_subscribe_without_japanese_evidence_suppressed(self):
        output, calls = run_one(
            "Please subscribe to my channel!", "逃げろ逃げろ")
        self.assertEqual(output, [])
        self.assertEqual(len(calls), 2)

    def test_genuine_credit_and_subscribe_not_suppressed(self):
        credit, credit_calls = run_one(
            "Subtitles by Watame", "字幕は私が制作しました")
        subscribe, sub_calls = run_one(
            "Please subscribe to my channel!", "チャンネル登録よろしくお願いします")
        self.assertEqual(credit, ["Subtitles by Watame"])
        self.assertEqual(subscribe, ["Please subscribe to my channel!"])
        self.assertEqual(len(credit_calls), 2)
        self.assertEqual(len(sub_calls), 2)

    def test_cadence_flags_only_extreme_cases(self):
        self.assertTrue(_phantom_timing_implausible(
            "Thank you for watching!", 0.20))
        self.assertTrue(_phantom_timing_implausible(
            "Please subscribe to my channel!", 0.45))
        self.assertFalse(_phantom_timing_implausible(
            "Thank you for watching!", 1.00))
        output, _ = run_one(
            "Thank you for watching!", "ご視聴ありがとうございました",
            duration_s=0.20, vad_confidence=0.60)
        self.assertEqual(output, [])
        output, _ = run_one(
            "Thank you for watching!", "ご視聴ありがとうございました",
            duration_s=0.20, vad_confidence=0.95)
        self.assertEqual(output, ["Thank you for watching!"])

    def test_weak_audio_drops_phantom_before_second_decode(self):
        output, calls = run_one(
            "Please subscribe to my channel!", "チャンネル登録よろしくお願いします",
            vad_confidence=0.55, no_speech_prob=0.90)
        self.assertEqual(output, [])
        self.assertEqual(len(calls), 1)

    def test_other_phantom_toggle_does_not_disable_outro_guard(self):
        output, calls = run_one(
            "Please subscribe to my channel!", "今日は車で行きます",
            filter_phantoms=False)
        self.assertEqual(output, ["Please subscribe to my channel!"])
        self.assertEqual(len(calls), 1)
        output, calls = run_one(
            "Thank you for watching!", "今日は車で行きます",
            filter_phantoms=False)
        self.assertEqual(output, [])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
