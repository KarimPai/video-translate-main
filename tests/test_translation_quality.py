"""No-model CAT subtitle filtering and high-confidence repairs."""
import unittest

from auto_translate.translate.quality import (
    is_obvious_nonverbal_japanese,
    is_skip_token,
    repair_high_confidence_meaning,
)


class TranslationQualityTests(unittest.TestCase):
    def test_only_obvious_syllable_loops_are_skipped(self):
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

    def test_three_hundred_thousand_not_thirty_thousand(self):
        self.assertEqual(
            repair_high_confidence_meaning("30万だろ!?", "Thirty thousand?!"),
            "300,000?!",
        )
        self.assertEqual(
            repair_high_confidence_meaning("30万だろ!?", "30,000?!"),
            "300,000?!",
        )
        self.assertEqual(
            repair_high_confidence_meaning("30分だろ!?", "30,000?!"),
            "30,000?!",
        )

    def test_lost_sight_not_lost_way(self):
        self.assertEqual(
            repair_high_confidence_meaning("見失いました", "I've lost my way."),
            "I lost sight of them.",
        )
        self.assertEqual(
            repair_high_confidence_meaning("見失いました", "I lost sight of the car."),
            "I lost sight of the car.",
        )

    def test_one_minute_extension_not_additional_minute(self):
        source = "ぶつかっちゃって1分に延ばせって言われてます"
        examples = [
            ("I got bumped into, and they're telling me to keep it going for one more minute.",
             "I got bumped into, and they're telling me to extend it to one minute."),
            ("We crashed; they asked to extend it for another minute.",
             "We crashed; they asked to extend it to one minute."),
        ]
        for english, corrected in examples:
            with self.subTest(english=english):
                self.assertEqual(
                    repair_high_confidence_meaning(source, english), corrected,
                )
        self.assertEqual(
            repair_high_confidence_meaning(
                "1分延ばせって言われた", "Add another minute.",
            ),
            "Add another minute.",
        )

    def test_shikanee_is_necessity_not_inability(self):
        self.assertEqual(
            repair_high_confidence_meaning(
                "もつ煮食うしかねえ", "I can't eat that stewed dish.",
            ),
            "Guess I'll have to eat motsuni stew.",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "もつ煮食うしかねえ", "I'll have to eat it.",
            ),
            "I'll have to eat it.",
        )
        self.assertEqual(
            repair_high_confidence_meaning("もつ煮食えねえ", "I can't eat motsuni."),
            "I can't eat motsuni.",
        )

    def test_repeated_rejection_preserves_meaning(self):
        self.assertEqual(
            repair_high_confidence_meaning("95はのんのんのんよ", "Level 95 is fine"),
            "Level 95? No, no, no.",
        )

    def test_CAT_thirty_second_duration_is_not_fifteen(self):
        self.assertEqual(
            repair_high_confidence_meaning(
                "30秒待つと言ったな",
                "You said you’d wait 15 seconds, didn’t you?",
            ),
            "You said you’d wait 30 seconds, didn’t you?",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "30秒待つと言ったな",
                "You said you’d wait 30 seconds, didn’t you?",
            ),
            "You said you’d wait 30 seconds, didn’t you?",
        )
        self.assertEqual(
            repair_high_confidence_meaning("15秒待て", "Wait 15 seconds."),
            "Wait 15 seconds.",
        )
        self.assertEqual(
            repair_high_confidence_meaning("30秒後に15秒待て", "Wait 15 seconds."),
            "Wait 15 seconds.",
        )

    def test_CAT_hearing_statement_is_not_dont_listen(self):
        self.assertEqual(
            repair_high_confidence_meaning("聞こえるな!", "Don't listen!"),
            "I can hear it!",
        )
        self.assertEqual(
            repair_high_confidence_meaning("聞こえるな!", "I can hear you!"),
            "I can hear you!",
        )
        self.assertEqual(
            repair_high_confidence_meaning("聞くな!", "Don't listen!"),
            "Don't listen!",
        )

    def test_CAT_greeting_only_repairs_attested_professional_hallucination(self):
        self.assertEqual(
            repair_high_confidence_meaning(
                "よろしくお願いします!",
                "Please find me 100% reliable and professional.",
            ),
            "Nice to meet you!",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "よろしくお願いします!",
                "Thanks in advance!",
            ),
            "Thanks in advance!",
        )
        self.assertEqual(
            repair_high_confidence_meaning(
                "よろしくお願いしますが質問です",
                "Please find me reliable and professional.",
            ),
            "Please find me reliable and professional.",
        )

    def test_only_exact_skip_token_is_accepted(self):
        self.assertTrue(is_skip_token("[SKIP]"))
        self.assertTrue(is_skip_token(" [skip] "))
        self.assertFalse(is_skip_token("SKIP"))
        self.assertFalse(is_skip_token("[SKIP] but more text"))


if __name__ == "__main__":
    unittest.main()
