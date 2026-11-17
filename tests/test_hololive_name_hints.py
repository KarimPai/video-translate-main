"""Offline tests for evidence-limited Hololive name normalization."""
import unittest
from unittest.mock import Mock

from auto_translate.config import TranslateConfig

from auto_translate.asr_correction import correct_known_asr_terms, lap_spelling_from_peer
from auto_translate.glossary import Glossary
from auto_translate.name_hints import whisper_name_prompt, canonicalize_english_names
from auto_translate.translate.plamo import PlamoTranslator, _current_glossary_matches
from auto_translate.translation_guard import suspicious_translation
from pathlib import Path


class HololiveNameHintsTests(unittest.TestCase):
    def test_glossary_covers_spoken_nicknames(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for jp, en in [("はーちゃまって初犯なんですか", "Haachama"),
                       ("はやちゃまは捕まってない", "Haachama"),
                       ("ラプラスラプラス", "Laplus")]:
            self.assertIn(en, [target for _, target in glossary.find(jp)])
        self.assertIn(("ラプラス", "Laplus"),
                      _current_glossary_matches("ラプラスが来た", "ホロGTA", glossary))
        self.assertIn(("ラプラス・ダークネス", "Laplus"),
                      _current_glossary_matches("ラプラス・ダークネス。", "ホロGTA", glossary))
        self.assertIn(("ラプラスダークネス", "Laplus"),
                      _current_glossary_matches("ラプラスダークネス。", "ホロGTA", glossary))

    def test_translation_preserves_spoken_form_and_surrounding_words(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig()
        translator.glossary = Glossary.load(Path("glossary.toml"))
        translator.vod_title = "ホロライブ ラプラス"
        for ja, raw, expected in (
            ("草。", "LOL!", "kusa!"),
            ("草を刈って", "Cut the lol.", "Cut the grass."),
            ("ラプ来た", "Rap arrived.", "Lap arrived."),
            ("みこちすごい", "Sakura Miko is amazing.", "Mikochi is amazing."),
            ("まつり来た", "The festival arrived.", "Matsuri arrived."),
            ("ラプが来た", "Rap is here.", "Lap is here."),
            ("ラプ様どこ？", "Where's Laplus?", "Where's Lap-sama?"),
            ("らぷさん", "rap-san", "Lap-san"),
            ("ラプちゃん", "La+ Darkness", "Lap-chan"),
            ("ラプラスが来た", "Laplace is here.", "Laplus is here."),
            ("ラップが上手い", "Good at rapping.", "Good at rapping."),
            ("まつりが来た", "The festival arrived.", "Matsuri arrived."),
            ("祭りちゃんどこ？", "Where is Festivals?", "Where is Matsuri-chan?"),
            ("マツリさん", "Natsuiro Matsuri", "Matsuri-san"),
            ("まつり", "Festivals!", "Matsuri!"),
            ("まつりちゃんと夏祭りに行く", "Going to the summer festival with Matsuri-chan.",
             "Going to the summer festival with Matsuri-chan."),
            ("お祭りに行く", "Going to the festival.", "Going to the festival."),
            ("夏祭りが来た", "The summer festival is here.", "The summer festival is here."),
            ("みこちゃん", "Sakura Miko", "Miko-chan"),
            ("みこが来た", "The shrine maiden arrived.", "Miko arrived."),
            ("みこち", "sakura miko", "Mikochi"),
            ("さくらみこ", "Sakura Miko", "Sakura Miko"),
            ("巫女さん", "Shrine maiden", "Shrine maiden"),
            ("わためぇ", "Tsunomaki Watame", "Watamee"),
            ("わため", "Tsunomaki Watame", "Watame"),
            ("トワ様", "towa sama", "Towa-sama"),
            ("すいちゃん", "Sui", "Sui-chan"),
            ("ラプとみこちが来た", "Laplus and Sakura Miko arrived.", "Lap and Mikochi arrived."),
            ("ラプはラップが上手い", "Lap is good at rapping.", "Lap is good at rapping."),
            ("ラプとラプラス", "Lap and Laplus", "Lap and Laplus"),
            ("ラプが来ない", "lap isn't coming.", "Lap isn't coming."),
        ):
            with self.subTest(ja=ja):
                translator._chat = Mock(return_value=raw)
                # Ambiguous bare nouns need confirmed Japanese dialogue;
                # a Hololive title alone does not establish person identity.
                prior = ("夏色まつりだよ" if ja in ("まつり来た", "まつりが来た", "まつり")
                         else "みこちだよ" if ja == "みこが来た" else "")
                self.assertEqual(translator.translate(ja, [(prior, "")] if prior else []), expected)
                translator._chat.assert_called_once()

    def test_context_and_word_boundaries(self):
        glossary = Glossary.load(Path("glossary.toml"))
        def hits(ja, prior="", title="ホロライブ ラプラス"):
            return dict(_current_glossary_matches(ja, title, glossary, prior))
        for word in ("ストラップ", "ラプソディー", "サランラップ", "はらぺこちゃん",
                     "申しこみこみ", "みこと", "みこし", "わためいと", "今日は晴れ"):
            with self.subTest(word=word):
                self.assertFalse(glossary.nicknames.keys() & hits(word).keys())
        self.assertFalse(hits("祭り", title="ホロライブ"))
        self.assertFalse(hits("まつり", prior="神社の屋台の話", title="ホロライブ"))
        self.assertEqual(hits("祭り", prior="夏色まつりのことだよ", title="")["祭り"], "Matsuri")
        self.assertEqual(hits("夏色まつり")["夏色まつり"], "Natsuiro Matsuri")
        self.assertNotIn("船長", hits("船長", title="宝鐘マリン ホロライブ"))
        self.assertEqual(hits("船長", "宝鐘マリンだよ")["船長"], "Senchou")
        self.assertEqual(hits("船長が出航する", "宝鐘マリンだよ")["船長"], "captain")
        self.assertNotIn("みこ", hits("みこ", title=""))
        self.assertNotIn("みこ", hits("神社のみこ"))
        self.assertEqual(hits("みこが来た", "みこちだよ", title="")["みこ"], "Miko")
        for jp, en in (("フワワ", "Fuwawa"), ("イオフィ", "Iofi"), ("ルイ姉", "Lui-nee")):
            self.assertEqual(hits(jp)[jp], en)

    def test_only_aligned_lap_spelling_or_corroborated_peer(self):
        heard, peer = "ラップが来た", "ラプが来た"
        self.assertEqual(lap_spelling_from_peer(heard, peer), heard)
        self.assertEqual(lap_spelling_from_peer(heard, peer, recent=["ラプ様だよ"]), peer)
        self.assertEqual(lap_spelling_from_peer("ラップ様どこ", "ラプ様どこ"), "ラプ様どこ")
        self.assertEqual(correct_known_asr_terms(heard, video_context="ホロライブ ラプラス"), heard)
        self.assertEqual(correct_known_asr_terms(heard, caption=peer), peer)
        self.assertEqual(correct_known_asr_terms(peer, caption=heard), heard)
        self.assertEqual(correct_known_asr_terms(heard, caption="ラプが来ない"), heard)
        for raw, caption in (("ラップが上手い", "ラプが上手い"), ("ラップで包む", "ラプで包む"),
                             ("ストラップ", "ストラプ"), ("rap", ""), ("rapping", "")):
            self.assertEqual(correct_known_asr_terms(raw, caption=caption), raw)

    def test_title_gated_name_correction_preserves_caption_conflict(self):
        ja = "確かはやちゃまは捕まってない"
        self.assertEqual(
            correct_known_asr_terms(ja, video_context="ホロGTA"),
            "確かはあちゃまは捕まってない",
        )
        self.assertEqual(correct_known_asr_terms(ja, video_context="料理配信"), ja)
        self.assertEqual(
            correct_known_asr_terms(ja, video_context="ホロGTA", caption=ja), ja,
        )

    def test_whisper_prompt_not_global(self):
        # No static name seed may bias Whisper for unrelated dialogue.
        self.assertIsNone(whisper_name_prompt("ホロGTA 3日目"))
        self.assertIsNone(whisper_name_prompt("ホロGTA 3日目", enabled=True))
        self.assertIsNone(whisper_name_prompt("普通のニュース放送"))
        self.assertIsNone(whisper_name_prompt("ホロGTA", enabled=False))

    def test_english_name_spelling_only_when_evidenced(self):
        self.assertEqual(
            canonicalize_english_names("はあちゃまが捕まった", "Hasa-chama got caught.", "ホロGTA"),
            "Haachama got caught.",
        )
        self.assertEqual(
            canonicalize_english_names("はーちゃまって初犯?", "Ha-chama's first offense?", "ホロGTA"),
            "Haachama's first offense?",
        )
        self.assertEqual(
            canonicalize_english_names("ラプラスラプラス", "Laplace, Laplace!", "ホロGTA"),
            "Laplus, Laplus!",
        )
        for original in (
            "La+ Darknesss.", "La+ Darkness.", "Laplus Darknesss.",
            "Laplace Darknesss.", "La Plus Darknesss.", "Laplus.",
        ):
            with self.subTest(original=original):
                self.assertEqual(
                    canonicalize_english_names("ラプラス・ダークネス。", original, "ホロGTA"),
                    "Laplus.",
                )
        self.assertEqual(
            canonicalize_english_names("ラプラスが来た", "La+ Darknesss arrived!", "ホロGTA"),
            "Laplus arrived!",
        )
        self.assertEqual(
            canonicalize_english_names("ラプラスだ", "Laplace.", "math lecture"), "Laplace.",
        )
        self.assertEqual(
            canonicalize_english_names("今日は晴れ", "La+ Darknesss.", "ホロGTA"),
            "La+ Darknesss.",
        )
        self.assertEqual(
            canonicalize_english_names("今日は晴れ", "Laplace.", "ホロGTA"), "Laplace.",
        )

    def test_guard_protects_matched_names(self):
        self.assertEqual(
            suspicious_translation("はあちゃまが来た", "Someone arrived",
                                   names=[("はあちゃま", "Haachama")]),
            "proper_name_mismatch",
        )
        self.assertIsNone(
            suspicious_translation("ラプラスが来た", "Laplus arrived",
                                   names=[("ラプラス", "Laplus")])
        )


if __name__ == "__main__":
    unittest.main()
