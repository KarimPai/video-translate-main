"""Offline neutral subtitle and integration tests; no model download."""
import queue
import unittest
import numpy as np
from auto_translate.config import WhisperConfig
from auto_translate.neutral_subtitles import needs_reference_evidence, neutralize_with_japanese
from auto_translate.pipeline import run_whisper
from auto_translate.types import Utterance
from auto_translate.whisper_v3 import WhisperDecode


class NeutralWhisperTests(unittest.TestCase):
    def test_only_selected_phrases_trigger_verification(self):
        for cue in ("Their first customer!", "Your first customer.",
                    "Her customer arrived.", "He left.", "I don't know.",
                    "But the police killed him!", "Those guys are here."):
            self.assertTrue(needs_reference_evidence(cue))
        for cue in ("Look out!", "The first customer arrived.",
                    "My friend is strong.", "We're ready!"):
            self.assertFalse(needs_reference_evidence(cue))

    def test_no_owner_prefers_noun_phrase(self):
        cases = [
            ("Their first customer!", "初めてのお客さん", "First customer!"),
            ("Your first customer.", "最初のお客さん", "First customer."),
            ("Their first customer arrived.", "最初のお客さんが来た", "The first customer arrived."),
            ("It's their first customer.", "最初のお客さんだ", "It's the first customer."),
            ("Her customer arrived.", "お客さんが来た", "The customer arrived."),
        ]
        for english, japanese, expected in cases:
            with self.subTest(english=english):
                self.assertEqual(neutralize_with_japanese(english, japanese), expected)

    def test_explicit_ownership_is_preserved(self):
        for english, japanese in [
            ("Your first customer.", "お前の最初のお客さん"),
            ("My first customer!", "俺の初めてのお客さん"),
            ("Their first customer.", "あの人の最初のお客さん"),
            ("Her first customer.", "彼女の初めてのお客さん"),
            ("Their first customer.", "ポルカさんの最初のお客さん"),
        ]:
            with self.subTest(japanese=japanese):
                self.assertEqual(neutralize_with_japanese(english, japanese), english)

    def test_gendered_short_clauses_only(self):
        self.assertEqual(neutralize_with_japanese("He left.", "もう行った"), "Someone left.")
        self.assertEqual(neutralize_with_japanese("She is here!", "来てるよ"), "Someone is here!")
        self.assertEqual(neutralize_with_japanese("She left.", "彼女は出た"), "She left.")
        self.assertEqual(neutralize_with_japanese("He told me.", "話をした"), "Someone told me.")

    def test_gender_neutrality_also_covers_full_dialogue(self):
        self.assertEqual(
            neutralize_with_japanese("But the police killed him!", "でも警察に殺された"),
            "But the police killed someone!",
        )
        self.assertEqual(
            neutralize_with_japanese("Those guys are here.", "あの人たちがいる"),
            "Those people are here.",
        )
        self.assertEqual(
            neutralize_with_japanese("This guy is amazing.", "この人すごい"),
            "This person is amazing.",
        )
        self.assertEqual(
            neutralize_with_japanese(
                "Those guys aren't your father.", "あの人たちは父親じゃない"
            ),
            "Those people aren't your father.",
        )
        self.assertEqual(
            neutralize_with_japanese("Those guys left.", "男たちが去った"),
            "Those guys left.",
        )
        self.assertEqual(
            neutralize_with_japanese("And he arrived.", "それで来ました"),
            "And someone arrived.",
        )
        self.assertEqual(
            neutralize_with_japanese("But he was killed.", "彼は殺された"),
            "But he was killed.",
        )
        self.assertEqual(
            neutralize_with_japanese("She met him.", "彼女は彼に会った"),
            "She met him.",
        )
        self.assertEqual(
            neutralize_with_japanese("This guy is here.", ""),
            "This guy is here.",
        )
        # Neither gender nor ownership is inferred for explicit Japanese names.
        self.assertEqual(
            neutralize_with_japanese("He arrived.", "みこさんが来た"),
            "He arrived.",
        )

    def test_omitted_first_person_short_reactions(self):
        self.assertEqual(neutralize_with_japanese("I don't know.", "分からない"), "Not sure.")
        self.assertEqual(neutralize_with_japanese("I found it!", "見つけた"), "Found it.")
        self.assertEqual(neutralize_with_japanese("I finished it.", "終わった"), "Finished.")
        self.assertEqual(neutralize_with_japanese("I got it.", "分かった"), "Got it.")
        self.assertEqual(neutralize_with_japanese("I don't know.", "俺は分からない"), "I don't know.")
        self.assertEqual(neutralize_with_japanese("I think so.", "そう思う"), "I think so.")

    def test_no_evidence_means_unchanged(self):
        for text in ("Their first customer!", "She left.", "I don't know."):
            self.assertEqual(neutralize_with_japanese(text, ""), text)

    @staticmethod
    def run_case(english, japanese, cfg=None, duration_s=1.0):
        class Fake:
            def __init__(self):
                self.cfg = cfg or WhisperConfig()
                self.ja_calls = 0
                self.ast_calls = 0
            def decode_audio(self, audio, *, prompt=None, hotwords=None, temperature=None):
                self.ast_calls += 1
                return WhisperDecode(english, avg_logprob=-.2,
                                     no_speech_prob=.02, compression_ratio=1.1)
            def transcribe_japanese_evidence(self, audio):
                self.ja_calls += 1
                return japanese
        fake = Fake()
        inside, outside = queue.Queue(), queue.Queue()
        n = int(duration_s * 16000)
        inside.put(Utterance(0.0, duration_s,
                             np.full(n, .1, dtype=np.float32), vad_confidence=.95))
        inside.put(None)
        run_whisper(fake, inside, outside)
        found = []
        while (result := outside.get_nowait()) is not None:
            found.append(result)
        return fake, found

    def test_pipeline_full_sentence_gender_neutral_pass(self):
        model, subs = self.run_case("But the police killed him!", "でも警察に殺された")
        self.assertEqual([x.en for x in subs], ["But the police killed someone!"])
        self.assertEqual((model.ast_calls, model.ja_calls), (1, 1))

    def test_pipeline_targeted_audio_pass(self):
        model, subs = self.run_case("Their first customer!", "最初のお客さん")
        self.assertEqual([x.en for x in subs], ["First customer!"])
        self.assertEqual((model.ast_calls, model.ja_calls), (1, 1))

    def test_pipeline_preserves_actual_owner(self):
        model, subs = self.run_case("Your first customer!", "あなたの最初のお客さん")
        self.assertEqual([x.en for x in subs], ["Your first customer!"])
        self.assertEqual(model.ja_calls, 1)

    def test_pipeline_no_extra_decoding_for_normal_text(self):
        model, subs = self.run_case("Get in the car!", "車に乗って")
        self.assertEqual([x.en for x in subs], ["Get in the car!"])
        self.assertEqual(model.ja_calls, 0)

    def test_pipeline_can_disable_neutrality(self):
        model, subs = self.run_case(
            "Their first customer!", "最初のお客さん",
            WhisperConfig(neutral_subtitles=False),
        )
        self.assertEqual([x.en for x in subs], ["Their first customer!"])
        self.assertEqual(model.ja_calls, 0)

    def test_pipeline_empty_japanese_keeps_source_english(self):
        _, subs = self.run_case("Their first customer!", "")
        self.assertEqual([x.en for x in subs], ["Their first customer!"])

    def test_audio_length_guard(self):
        model, subs = self.run_case(
            "Their first customer!", "最初のお客さん",
            WhisperConfig(neutral_verify_max_audio_s=.5),
        )
        self.assertEqual([x.en for x in subs], ["Their first customer!"])
        self.assertEqual(model.ja_calls, 0)


if __name__ == "__main__":
    unittest.main()
