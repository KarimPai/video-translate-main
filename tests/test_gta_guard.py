"""Evidence-bound GTA terminology and polarity checks (no model required)."""
import unittest

from auto_translate.translate.gta_guard import (
    gta_prompt_terms,
    polarity_conflict,
    repair_gta_translation,
    short_polarity_fallback,
)


class GtaTranslationGuardTests(unittest.TestCase):
    def test_vest_meaning_requires_safety_context(self):
        self.assertIn(
            ("ベスト", "protective vest / body armor (equipment)"),
            gta_prompt_terms("ベスト3枚あってもいいな", "Hololive GTA"),
        )
        self.assertEqual(
            gta_prompt_terms("ベスト3枚あってもいいな", "Japanese fashion"),
            [],
        )
        self.assertEqual(
            repair_gta_translation(
                "ベスト3枚あってもいいな",
                "It would be nice to have three of the best.",
                "Hololive GTA",
            ),
            "It would be nice to have three vests.",
        )
        self.assertEqual(
            repair_gta_translation(
                "ベスト3人だ", "They are the best three.", "Hololive GTA",
            ),
            "They are the best three.",
        )
        self.assertIn(
            ("ベスト", "protective vest / body armor (equipment)"),
            gta_prompt_terms("ベスト買うにお金足りない", "Hololive GTA"),
        )
        self.assertEqual(
            repair_gta_translation(
                "ベスト買うにお金足りない",
                "I don't have enough money to buy the best one.",
                "Hololive GTA",
            ),
            "I don't have enough money to buy a vest.",
        )


    def test_rocket_ammo_repair_does_not_replace_unrelated_ball(self):
        self.assertEqual(
            repair_gta_translation(
                "ロケランの弾一千万",
                "Rocket Launcher: 10 Million Balls.",
                "Hololive GTA",
            ),
            "Rocket launcher ammunition costs 10 million.",
        )
        self.assertEqual(
            repair_gta_translation(
                "野球の玉一千万", "Ten million balls.", "Hololive GTA",
            ),
            "Ten million balls.",
        )

    def test_sticker_context_can_repair_clipped_side_question(self):
        self.assertEqual(
            repair_gta_translation(
                "この横貼りたくね?", "This horizontal strip?",
                "Hololive GTA", ["車両飾る", "ステッカー貼ろう"],
            ),
            "Wouldn't it look good stuck on the side?",
        )
        self.assertEqual(
            repair_gta_translation(
                "この横貼りたくね?", "This horizontal strip?",
                "Reading poetry", [],
            ),
            "This horizontal strip?",
        )

    def test_short_polarity_does_not_override_context_dependent_lines(self):
        self.assertTrue(polarity_conflict("すごい", "That's awful."))
        self.assertTrue(polarity_conflict("だめ", "Great!"))
        self.assertFalse(polarity_conflict("すごい", "Amazing!"))
        self.assertFalse(polarity_conflict("だめ", "That's not okay."))
        self.assertFalse(polarity_conflict("やばい", "That's bad."))
        self.assertFalse(polarity_conflict("すごくない?", "That's not great?"))
        self.assertEqual(short_polarity_fallback("すごい。"), "Amazing!")
        self.assertEqual(short_polarity_fallback("無理"), "That's impossible.")
        self.assertIsNone(short_polarity_fallback("すごいことになった"))


if __name__ == "__main__":
    unittest.main()
