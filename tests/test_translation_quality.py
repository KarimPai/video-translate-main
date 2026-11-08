"""No-model checks for suspicious Japanese / English subtitle handling."""

import unittest

from auto_translate.translate.quality import (
    is_obvious_nonverbal_japanese,
    is_skip_token,
    translation_review_reason,
)


class TranslationQualityTests(unittest.TestCase):
    def test_only_obvious_syllable_loops_are_skipped_before_translation(self):
        for source in ("タタタタあっとたっとと", "んけんけんけん", "ああああああ"):
            with self.subTest(source=source):
                self.assertTrue(is_obvious_nonverbal_japanese(source))
        for source in (
            "わため", "ぺこら", "警察が来た！", "タタタタ、警察が来た！",
            "あっ来た来た来た", "すごいね", "今日は楽しかった",
            "ゴロゴロ", "おはようございます", "ごめんごめんごめん",
            "やばいやばいやばい", "ここここここ",
            "いけいけいけ", "こいこいこい", "よしよしよし",
        ):
            with self.subTest(source=source):
                self.assertFalse(is_obvious_nonverbal_japanese(source))

    def test_suspicious_phonetic_output_triggers_review(self):
        self.assertEqual(
            translation_review_reason("わたしが来た", "Ta-ta-ta-ta, at-tat-to"),
            "romanized_syllable_loop",
        )
        self.assertEqual(
            translation_review_reason("んけんけんけん", "N-ken-ken-ken"),
            "romanized_syllable_loop",
        )
        self.assertEqual(
            translation_review_reason("ドーンパチジャマーシャー", "Boom-pachi-jammer-sha!"),
            "phonetic_transliteration",
        )

    def test_protect_names_repeated_real_words_and_meaningful_dialogue(self):
        for ja, en in (
            ("わためが来た", "Watame is here."),
            ("ペコラの車", "Pekora's car."),
            ("待って、待って、待って", "Wait, wait, wait!"),
            ("行け行け行け", "Go go go!"),
            ("撃って撃って撃って", "Fire fire fire!"),
            ("パジェロ", "Pajero"),
            ("よろしく", "Hello there."),
            ("すごいね", "That's amazing!"),
        ):
            with self.subTest(ja=ja, en=en):
                self.assertIsNone(translation_review_reason(ja, en))

    def test_polarity_depends_on_the_whole_source(self):
        for ja, en, expected in [
            ("すごいね", "That's awful.", "positive_appraisal_reversed"),
            ("勝てるよ", "You can’t win.", "affirmative_ability_reversed"),
            ("行けない", "I can go.", "possible_missing_negation"),
            ("行けない", "I can’t go.", None),
            ("最高だよ", "Couldn't be better!", None),
            ("上手いね", "Not bad!", None),
            ("すごく悪い", "That's terrible.", None),
            ("すごいダメージ", "That's bad damage.", None),
            ("行かないかもしれない", "I might stay home.", None),
        ]:
            with self.subTest(ja=ja, en=en):
                self.assertEqual(translation_review_reason(ja, en), expected)

    def test_only_exact_skip_token_is_accepted(self):
        self.assertTrue(is_skip_token("[SKIP]"))
        self.assertTrue(is_skip_token(" [skip] "))
        self.assertFalse(is_skip_token("SKIP"))
        self.assertFalse(is_skip_token("[SKIP] but more text"))
        self.assertEqual(
            translation_review_reason("おはよう", "[SKIP]"), "unexpected_skip"
        )

    def test_explicit_negative_mistranslation_only_flags_likely_loss(self):
        self.assertEqual(
            translation_review_reason("晴れてないわ", "It's sunny!"),
            "possible_missing_negation",
        )
        self.assertIsNone(
            translation_review_reason("晴れてないわ", "It's not sunny!")
        )
        self.assertIsNone(
            translation_review_reason("寝ないといけない", "I have to sleep.")
        )
        self.assertIsNone(
            translation_review_reason("運転が下手", "The driving is bad.")
        )


if __name__ == "__main__":
    unittest.main()
