"""No-model checks for suspicious Japanese / English subtitle handling."""

import unittest

from auto_translate.translate.quality import (
    is_obvious_nonverbal_japanese,
    is_skip_token,
    translation_review_reason,
    repair_high_confidence_meaning,
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

    def test_rhetorical_gaming_slang_detects_meaning_reversal(self):
        self.assertEqual(
            translation_review_reason(
                "えぐないきたきたきたまあ僕のアングラにお任せあれ",
                "It's not that bad, let's go, just leave it to me.",
            ), "rhetorical_egunai_reversed"
        )
        self.assertIsNone(
            translation_review_reason("えぐない?", "Isn't that brutal?")
        )
        self.assertIsNone(
            translation_review_reason("えぐいな", "It's not that bad.")
        )

    def test_colloquial_shika_nee_means_no_choice_not_cannot(self):
        self.assertEqual(
            translation_review_reason(
                "もつ煮食うしかねえ", "I can't eat that stewed dish."
            ), "exclusive_necessity_reversed",
        )
        for english in (
            "Guess I'll have to eat motsuni stew.",
            "I can't help but eat motsuni.",
            "There's nothing else to do but eat it.",
        ):
            with self.subTest(english=english):
                self.assertIsNone(
                    translation_review_reason("もつ煮食うしかねえ", english)
                )
        self.assertEqual(
            repair_high_confidence_meaning(
                "もつ煮食うしかねえ", "I can't eat that stewed dish."
            ), "Guess I'll have to eat motsuni stew.",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "もつ煮食うしかねえ", "I'll have to eat it."
            ), "I'll have to eat it.",
        )
        # Do not rephrase an unrelated refusal.
        self.assertEqual(
            repair_high_confidence_meaning("もつ煮食えねえ", "I can't eat motsuni."),
            "I can't eat motsuni.",
        )

    def test_japanese_man_unit_and_lost_sight_are_not_reversed(self):
        self.assertEqual(
            translation_review_reason("30万だろ!?", "Thirty thousand?!"),
            "japanese_man_unit_wrong",
        )
        self.assertEqual(
            repair_high_confidence_meaning("30万だろ!?", "Thirty thousand?!"),
            "300,000?!",
        )
        self.assertEqual(
            repair_high_confidence_meaning("30万だろ!?", "30,000?!"),
            "300,000?!",
        )
        self.assertIsNone(
            translation_review_reason("30万だろ!?", "300,000, right?!")
        )
        self.assertEqual(
            repair_high_confidence_meaning("30分だろ!?", "30,000?!"),
            "30,000?!",
        )
        self.assertEqual(
            translation_review_reason("見失いました", "I've lost my way."),
            "lost_sight_as_lost_way",
        )
        self.assertEqual(
            repair_high_confidence_meaning("見失いました", "I've lost my way."),
            "I lost sight of them.",
        )
        self.assertEqual(
            repair_high_confidence_meaning("見失いました", "I lost sight of the car."),
            "I lost sight of the car.",
        )

    def test_one_minute_extension_is_not_an_extra_minute(self):
        ja = "ぶつかっちゃって1分に延ばせって言われてます"
        examples = [
            ("I got bumped into, and they're telling me to keep it going for one more minute.",
             "I got bumped into, and they're telling me to extend it to one minute."),
            ("We crashed; they asked to extend it for another minute.",
             "We crashed; they asked to extend it to one minute."),
        ]
        for english, corrected in examples:
            with self.subTest(english=english):
                self.assertEqual(
                    translation_review_reason(ja, english),
                    "duration_extension_one_minute",
                )
                self.assertEqual(
                    repair_high_confidence_meaning(ja, english),
                    corrected,
                )
        self.assertIsNone(
            translation_review_reason(ja, "They told us to extend it to one minute.")
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "1分延ばせって言われた", "Add another minute.",
            ),
            "Add another minute.",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "1分に延ばせって言われた", "They told me to extend it to one minute.",
            ),
            "They told me to extend it to one minute.",
        )

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
