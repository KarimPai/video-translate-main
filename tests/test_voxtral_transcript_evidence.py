"""Offline transcript gates; these tests measure rules, not model accuracy."""
import unittest
import queue
from types import SimpleNamespace

import numpy as np

from auto_translate.asr.quality_guard import (
    potential_japanese_transcript,
    supports_japanese_transcript,
)
from auto_translate.language_filter import LanguageDecision
from auto_translate.pipeline import run_asr
from auto_translate.types import Utterance


class VoxtralTranscriptEvidenceTests(unittest.TestCase):
    japanese = LanguageDecision("ja", .94, .94, False, "japanese_or_uncertain")
    english = LanguageDecision("en", .99, .001, True, "confident_non_japanese")
    uncertain = LanguageDecision("unknown", 0., 0., False, "too_short")

    def test_japanese_code_switching_keeps_romanized_names_and_loanwords(self):
        for text in ("Botanちゃんが来た！", "PCのbuttonを押して", "OK、すぐ行く"):
            with self.subTest(text=text):
                self.assertTrue(supports_japanese_transcript(text))

    def test_reviewed_latin_names_survive_adapter_for_pipeline_evidence(self):
        for text in ("Botan!", "Botan-chan", "Shishiro Botan", "Mikochi", "Lap-sama"):
            with self.subTest(text=text):
                self.assertTrue(potential_japanese_transcript(text))
                self.assertFalse(supports_japanese_transcript(text))
                self.assertTrue(supports_japanese_transcript(text, language_decision=self.japanese))

    def test_named_metadata_can_support_only_the_actual_latin_name(self):
        for title in ("Botan livestream", "Shishiro Botan【Minecraft】", "獅白ぼたん"):
            with self.subTest(title=title):
                self.assertTrue(supports_japanese_transcript("Botan-chan!", video_context=title))
        self.assertTrue(supports_japanese_transcript(
            "Mikochi", confirmed_participants=("Mikochi",)))
        self.assertFalse(supports_japanese_transcript("Botan", video_context="Hololive gaming"))
        self.assertFalse(supports_japanese_transcript("Miko", video_context="Mikoto plays"))
        self.assertFalse(supports_japanese_transcript("Lap", video_context="Laplace music"))

    def test_romanized_reactions_need_actual_japanese_acoustic_evidence(self):
        for text in ("hai", "kusa", "sugoi", "Ｄａｉｊｏｕｂｕ!"):
            with self.subTest(text=text):
                self.assertTrue(potential_japanese_transcript(text))
                self.assertTrue(supports_japanese_transcript(text, language_decision=self.japanese))
                self.assertFalse(supports_japanese_transcript(text, language_decision=self.uncertain))
                self.assertFalse(supports_japanese_transcript(text, video_context="Hololive kusa"))

    def test_confident_foreign_audio_cannot_be_overruled_by_metadata(self):
        for text in ("Botan", "hai", "こんにちは", "你好"):
            with self.subTest(text=text):
                self.assertFalse(supports_japanese_transcript(
                    text, language_decision=self.english, video_context="Botan"))

    def test_names_inside_foreign_sentences_do_not_authorize_the_sentence(self):
        for text in ("Hello Botan", "I am Botan", "Botan is here", "Thank you for watching",
                     "How are you doing today?", "안녕하세요", "12345", "", None):
            with self.subTest(text=text):
                self.assertFalse(potential_japanese_transcript(text))
                self.assertFalse(supports_japanese_transcript(
                    text, language_decision=self.japanese, video_context="Botan"))

    def test_pipeline_applies_original_audio_evidence_and_preserves_raw_text(self):
        for text, verdict, title, expected in (
                ("Botan!", self.uncertain, "Botan livestream", True),
                ("kusa", self.japanese, "", True),
                ("kusa", self.uncertain, "Hololive", False),
                ("Botan", self.english, "Botan livestream", False),
                ("Hello Botan", self.japanese, "Botan livestream", False)):
            with self.subTest(text=text, language=verdict.language):
                source, output = queue.Queue(), queue.Queue()
                source.put(Utterance(5., 8., np.ones(48000, np.float32), speech_start_s=5.1))
                source.put(None)
                worker = SimpleNamespace(transcribe=lambda *_: text,
                                         last_raw_transcript="  " + text + "\n")
                run_asr(worker, source, output, vod_title=title,
                        language_filter=SimpleNamespace(classify=lambda _: verdict))
                rows = [row for row in output.queue if row is not None]
                self.assertEqual(len(rows), int(expected))
                if rows:
                    self.assertEqual(rows[0].ja, text)
                    self.assertEqual(rows[0].name_context["voxtral_raw"], "  " + text + "\n")
                    self.assertEqual(rows[0].speech_start_s, 5.1)


if __name__ == "__main__":
    unittest.main()
