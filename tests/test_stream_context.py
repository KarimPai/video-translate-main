"""Live Hololive ARK regression cases.
Run: python -m unittest discover -s tests -p test_stream_context.py
"""
import unittest
from auto_translate.stream_context import (
    ark_glossary_hits, is_ark_context, is_incomplete_fragment,
    repair_ark_spelling, repair_peko_nickname, repair_verified_caption,
    repair_gta_carjacking, is_horse_racing_context, repair_horse_racing_spelling,
)
from auto_translate.translate.quality import translation_review_reason, repair_high_confidence_meaning


class TestStreamContext(unittest.TestCase):
    def test_game(self):
        self.assertFalse(is_ark_context("", "ギガのと"))
        self.assertTrue(is_ark_context("ARK配信", "ギガのと"))
        self.assertTrue(is_ark_context("", "ギガのと", ["恐竜がいなくなった"]))
        self.assertEqual(repair_ark_spelling("ギガのと討伐", "ARK"), "ギガノト討伐")
        self.assertEqual(repair_ark_spelling("ギガの土地", "ARK"), "ギガの土地")
        self.assertEqual(repair_ark_spelling("恐竜をテームした", "ARK"), "恐竜をテイムした")
        self.assertEqual(repair_ark_spelling("ギガのと討伐"), "ギガのと討伐")
        self.assertEqual(repair_ark_spelling("気がの都が襲来", "ARK"), "ギガノトが襲来")
        self.assertEqual(repair_ark_spelling("このギガナとどっちが強い", "ARK"), "このギガノトどっちが強い")
        self.assertEqual(repair_ark_spelling("ギガノドだけ討伐", "ARK"), "ギガノトだけ討伐")
        self.assertEqual(repair_ark_spelling("気がの都が襲来"), "気がの都が襲来")
        # Single genuine ARK signal plus a weak, phonetically close spelling
        # is enough; neither one by itself is sufficient.
        self.assertFalse(is_ark_context("", "ギガノドだけ討伐"))
        self.assertTrue(is_ark_context("", "ギガノドだけ討伐", ["ギガノトが来た"]))
        self.assertEqual(repair_ark_spelling("ギガノドだけ討伐", "", ["ギガノトが来た"]),
                         "ギガノトだけ討伐")
        self.assertEqual(repair_ark_spelling("このギガナとどっちが強い", "", ["ギガノトが来た"]),
                         "このギガノトどっちが強い")
        self.assertEqual(repair_ark_spelling("ギガの土地", "", ["ギガノトが来た"]),
                         "ギガの土地")
        self.assertEqual(repair_ark_spelling("おさらちってこと?"), "おさらちってこと?")
        self.assertEqual(repair_ark_spelling("おさらちってこと?", "", ["更地になってる"]),
                         "更地ってこと?")
        self.assertEqual(repair_ark_spelling("あーでもおさらちどういうこと?", "", ["更地ってこと?"]),
                         "あーでも更地どういうこと?")
        self.assertEqual(repair_ark_spelling("おさらちが来た", "", ["更地になってる"]),
                         "おさらちが来た")
        self.assertIn(("ギガノト", "Giganotosaurus"), ark_glossary_hits("ギガノト討伐"))

    def test_gta_carjacking_requires_dialogue_evidence(self):
        # Directly observed errors from the Hololive GTA VOD.
        self.assertEqual(
            repair_gta_carjacking(
                "え、カーチェイスカーチェイスじゃないカジャックされたとこに行くよ",
                "Hololive GTA",
            ),
            "え、カーチェイスカーチェイスじゃないカージャックされたとこに行くよ",
        )
        self.assertEqual(
            repair_gta_carjacking(
                "カイジャックする!", "", [
                    "カーチェイスじゃないカージャックされたとこに行くよ",
                ],
            ),
            "カージャックする!",
        )
        self.assertEqual(
            repair_gta_carjacking("カイジャックする!"),
            "カイジャックする!",
        )
        self.assertEqual(
            repair_gta_carjacking("ハイジャックする!", "Hololive GTA"),
            "ハイジャックする!",
        )
        self.assertEqual(
            repair_gta_carjacking("カージャックする!", "Hololive GTA"),
            "カージャックする!",
        )

    def test_horse_racing_context_and_conservative_homophone_repair(self):
        self.assertTrue(is_horse_racing_context("ホロ1GP", "オッズ"))
        self.assertTrue(is_horse_racing_context("", "パドックでは", ["推奨馬は6番"]))
        self.assertFalse(is_horse_racing_context("", "推奨バーとして"))
        self.assertTrue(is_horse_racing_context("", "ワイドに軸馬", ["馬券買おう"]))
        self.assertTrue(is_horse_racing_context("", "払い戻し行こうか", ["馬券買ったの?"]))
        self.assertFalse(is_horse_racing_context("", "ワイド画面にして"))
        self.assertEqual(
            repair_horse_racing_spelling(
                "これを推奨バーとしてさせていただきたい",
                "", ["今日の推奨馬は6番", "パドックで良かった"],
            ),
            "これを推奨馬としてさせていただきたい",
        )
        self.assertEqual(
            repair_horse_racing_spelling(
                "これを推奨バーとして", "競馬配信"
            ), "これを推奨バーとして",
        )
        self.assertEqual(
            repair_horse_racing_spelling(
                "推奨バーを作ろう", "", ["推奨馬は6番"]
            ), "推奨バーを作ろう",
        )

    def test_nickname(self):
        text = "実はテコちゃんに討伐依頼をしに来た"
        self.assertEqual(repair_peko_nickname(text), text)
        self.assertIn("ぺこちゃん", repair_peko_nickname(text, title="Pekora ARK"))
        self.assertIn("ぺこちゃん", repair_peko_nickname(text, caption="実はぺこちゃんに討伐依頼をしに来た"))
        self.assertEqual(repair_peko_nickname("猫ちゃん", title="Pekora"), "猫ちゃん")

    def test_caption(self):
        a = "実はいろいろあってテコちゃんに討伐依頼をしに来た"
        b = "実はいろいろあってぺこちゃんに討伐依頼をしに来た"
        self.assertEqual(repair_verified_caption(a, b), b)
        self.assertEqual(repair_verified_caption("ここは安全じゃないと思います", "ここは安全じゃあると思います"), "ここは安全じゃないと思います")
        self.assertEqual(repair_verified_caption("レベルは95だったかもしれない", "レベルは85だったかもしれない"), "レベルは95だったかもしれない")
        self.assertEqual(repair_verified_caption("テコちゃん", "ぺこちゃん"), "テコちゃん")

    def test_translation(self):
        self.assertEqual(translation_review_reason("読み込みが軽くなっちまっておる", "The loading has slowed down."), "loading_performance_reversed")
        self.assertEqual(translation_review_reason("95はのんのんのんよ", "95 is totally fine"), "rejection_reversed")
        self.assertEqual(translation_review_reason(
            "95はのんのんのんよ。", "95 is the fourth, the fifth, the sixth, the seventh."
        ), "rejection_reversed")
        self.assertEqual(repair_high_confidence_meaning(
            "読み込みが軽くなっちまっておるあそこすごいぞ!",
            "The loading has slowed down—that place is insane!"
        ), "The load is lighter now—that place is insane!")
        self.assertEqual(repair_high_confidence_meaning(
            "95はのんのんのんよ。", "95 is totally fine."
        ), "Level 95? No, no, no.")
        self.assertEqual(repair_high_confidence_meaning(
            "95はのんのんのんよ。", "95 is the fourth, the fifth, the sixth, the seventh."
        ), "Level 95? No, no, no.")
        self.assertEqual(repair_high_confidence_meaning(
            "95はのんのんのんよ。", "No, no, no. 95 is too low."
        ), "Level 95? No, no, no.")
        self.assertEqual(repair_high_confidence_meaning(
            "95は大丈夫だよ。", "95 is fine."
        ), "95 is fine.")
        self.assertEqual(repair_high_confidence_meaning(
            "読み込みが重くなった", "The loading has slowed down"
        ), "The loading has slowed down")
        self.assertTrue(is_incomplete_fragment("拠点から離せたんだけど"))
        self.assertFalse(is_incomplete_fragment("倒してきます。"))


if __name__ == "__main__":
    unittest.main()
