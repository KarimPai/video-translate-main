"""Live Hololive ARK regression cases.
Run: python -m unittest discover -s tests -p test_stream_context.py
"""
import unittest
from auto_translate.stream_context import (
    ark_glossary_hits, is_ark_context, is_incomplete_fragment,
    repair_ark_spelling, repair_peko_nickname, repair_verified_caption,
    repair_gta_carjacking, is_gta_context, is_horse_racing_context, repair_horse_racing_spelling,
    valorant_glossary_hits, repair_valorant_translation,
    repair_barotrauma_translation, repair_ark_tier_translation,
)
from auto_translate.translate.quality import repair_high_confidence_meaning


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

    def test_ark_ascendant_tier_only_in_supported_game_context(self):
        self.assertIn(
            ("至高のサドル", "Ascendant saddle (ARK quality tier)"),
            ark_glossary_hits("至高のサドルは神"),
        )
        self.assertEqual(
            repair_ark_spelling("思考のサドルは神", "Hololive ARK"),
            "至高のサドルは神",
        )
        self.assertEqual(
            repair_ark_spelling("思考のサドルは神", "Philosophy lecture"),
            "思考のサドルは神",
        )
        self.assertEqual(
            repair_ark_tier_translation(
                "至高のサドルは神", "The ultimate saddle is god-tier.",
                "Hololive ARK",
            ),
            "The Ascendant saddle is god-tier.",
        )
        self.assertEqual(
            repair_ark_tier_translation(
                "思考のサドルは神", "The saddle of thought is god.",
                "Hololive ARK",
            ),
            "The Ascendant saddle is god.",
        )
        self.assertEqual(
            repair_ark_tier_translation(
                "思考のサドル", "The saddle of thought",
                "Philosophy lecture",
            ),
            "The saddle of thought",
        )
        self.assertEqual(
            repair_ark_tier_translation(
                "至高のサドル", "The Ascendant saddle", "ARK",
            ),
            "The Ascendant saddle",
        )
        self.assertEqual(
            repair_ark_tier_translation(
                "至高の演奏だ", "An ultimate performance", "ARK",
            ),
            "An ultimate performance",
        )

    def test_valorant_game_aliases_are_evidence_bound(self):
        self.assertIn(
            ("バローラント", "VALORANT (5v5 tactical shooter game)"),
            valorant_glossary_hits("バローラント3人でやってみる?"),
        )
        self.assertIn(
            ("バロランと", "VALORANT (5v5 tactical shooter game)"),
            valorant_glossary_hits("バロランとってどんなゲーム?", ["バローラントだ"]),
        )
        self.assertIn(
            ("バローランド", "VALORANT (5v5 tactical shooter game)"),
            valorant_glossary_hits("バローランド面白い", ["バロラントやった?"]),
        )
        self.assertEqual(valorant_glossary_hits("バローランド面白い"), [])
        self.assertEqual(valorant_glossary_hits("バロランと"), [])
        self.assertEqual(valorant_glossary_hits("バロトラウマ"), [])
        self.assertEqual(valorant_glossary_hits("バロランと", ["バロトラウマ"]), [])
        self.assertIn(
            ("バロランと", "VALORANT (5v5 tactical shooter game)"),
            valorant_glossary_hits("バロランとだバロランとってどんなゲーム?"),
        )

    def test_valorant_output_repairs_names_not_dialogue(self):
        self.assertEqual(
            repair_valorant_translation(
                "バローラント3人でバローラントやってみるか",
                "Let's try playing Barotrauma with three people.",
            ), "Let's try playing VALORANT with three people.",
        )
        self.assertEqual(
            repair_valorant_translation(
                "え、バローラントって3人でもできるの?",
                "Can you play Baro-Rant with just three people?",
            ), "Can you play VALORANT with just three people?",
        )
        self.assertEqual(
            repair_valorant_translation(
                "バローランドめっちゃおもしろい",
                "Baroland is so fun!", ["ヴァロラントやる?"],
            ), "VALORANT is so fun!",
        )
        self.assertEqual(
            repair_valorant_translation(
                "バローランドめっちゃおもしろい",
                "Baroland is so fun!",
            ), "Baroland is so fun!",
        )
        self.assertEqual(
            repair_valorant_translation(
                "バロトラウマを遊ぼう", "Let's play Barotrauma.",
                ["VALORANTも遊びたい"],
            ), "Let's play Barotrauma.",
        )
        self.assertEqual(
            repair_valorant_translation(
                "VALORANTとバロトラウマを比べよう",
                "Let's compare Valorant and Barotrauma.",
            ), "Let's compare Valorant and Barotrauma.",
        )

    def test_hyphenated_valorant_and_distinct_barotrauma_are_corrected(self):
        source = "バローランドっておもしろいぞ"
        recent = ["バロラントやってる?"]
        for wrong in ("Baro-land", "Baro Rando", "Baro-rant", "Baroland"):
            with self.subTest(wrong=wrong):
                self.assertEqual(
                    repair_valorant_translation(
                        source, f"{wrong} is fun.", recent,
                    ),
                    "VALORANT is fun.",
                )
                # Without independent evidence, do not guess a game title.
                self.assertEqual(
                    repair_valorant_translation(source, f"{wrong} is fun."),
                    f"{wrong} is fun.",
                )
        for wrong in ("Valentine's Trauma", "Bartourma", "Barttrauma"):
            with self.subTest(wrong=wrong):
                self.assertEqual(
                    repair_barotrauma_translation(
                        "バロトラウマは別のゲームだよ",
                        f"{wrong} is a different game.",
                    ),
                    "Barotrauma is a different game.",
                )
        self.assertEqual(
            repair_barotrauma_translation(
                "バローラントは違うゲーム", "Valentine's Trauma is a different game.",
            ),
            "Valentine's Trauma is a different game.",
        )
        self.assertEqual(
            repair_valorant_translation(
                "VALORANTとバロトラウマを比べる",
                "Compare VALORANT to Barotrauma.",
            ),
            "Compare VALORANT to Barotrauma.",
        )

    def test_gta_title_context_is_explicit(self):
        self.assertTrue(is_gta_context("Hololive GTA Day 4"))
        self.assertTrue(is_gta_context("【グラセフ】ホロライブ"))
        self.assertFalse(is_gta_context("ARK racing stream"))

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

    def test_shared_translation_repairs_without_index_review(self):
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
