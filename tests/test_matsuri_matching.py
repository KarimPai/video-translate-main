"""Matsuri name recall with literal festival and custom-glossary controls."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from auto_translate.glossary import Glossary
from auto_translate.name_context import NameMention, is_matsuri_target, is_name_target


class MatsuriMatchingTests(unittest.TestCase):
    def setUp(self):
        self.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")

    def hits(self, ja, *, title="ホロライブ", previous=""):
        return self.glossary.find(ja, video_context=title, previous_ja=previous)

    def test_kana_and_honorifics_keep_spoken_form(self):
        for name in ("まつり", "マツリ"):
            for suffix, target in (("", "Matsuri"), ("ちゃん", "Matsuri-chan"),
                                   ("さん", "Matsuri-san"), ("様", "Matsuri-sama"),
                                   ("さま", "Matsuri-sama")):
                with self.subTest(name=name, suffix=suffix):
                    self.assertIn((name + suffix, target), self.hits(name + suffix + "はどうした"))

    def test_full_names_and_honorifics_do_not_need_metadata(self):
        for name in ("夏色まつり", "夏色マツリ", "夏色祭り"):
            for suffix, target in (("", "Natsuiro Matsuri"), ("ちゃん", "Natsuiro Matsuri-chan"),
                                   ("さん", "Natsuiro Matsuri-san"), ("様", "Natsuiro Matsuri-sama")):
                with self.subTest(name=name, suffix=suffix):
                    self.assertIn((name + suffix, target), self.hits(name + suffix + "が来た", title=""))

    def test_natural_person_continuations(self):
        for text in ("まつりはどうした", "マツリによろしく", "まつりと一緒に行く", "まつりもやってる",
                     "ねえ、まつりはどこ？", "まつりがいなくなった", "まつり来た"):
            with self.subTest(text=text):
                self.assertTrue(any(target == "Matsuri" for _, target in self.hits(text)))

    def test_bare_kana_requires_context_and_recent_identity_uses_japanese(self):
        self.assertEqual(self.hits("まつりはどうした", title="Minecraft"), [])
        self.assertIn(("まつり", "Matsuri"), self.hits("まつりはどうした", title="Minecraft",
                                                     previous="夏色まつりだよ\n了解"))
        self.assertIn(("マツリ", "Matsuri"), self.hits("マツリはどこ", title="Minecraft",
                                                     previous="まつりちゃんが来た\nちょっと待って"))
        self.assertFalse(self.hits("まつり", previous="神社の屋台の話"))

    def test_kanji_needs_stronger_name_identity(self):
        self.assertNotIn(("祭り", "Matsuri"), self.hits("祭り"))
        self.assertIn(("祭り", "Matsuri"), self.hits("祭り", title="", previous="夏色まつりのことだよ"))
        self.assertIn(("祭り", "Matsuri"), self.hits("祭りはどうした", title="Natsuiro Matsuri"))
        self.assertIn(("祭", "Matsuri"), self.hits("祭はどこ", title="夏色まつり"))
        self.assertIn(("祭りちゃん", "Matsuri-chan"), self.hits("祭りちゃんどこ？"))

    def test_literal_events_and_compounds_never_become_names(self):
        for text in ("お祭りに行く", "夏祭りが来た", "秋祭りが始まる", "ひなまつり", "文化祭", "学園祭",
                     "血祭りにあげる", "まつりごと", "まつりばやし", "神社のまつり", "まつりは明日開催",
                     "まつりに参加する", "祭りの屋台", "祭り会場", "お祭りちゃん",
                     "まつりの実行委員会", "まつりの準備", "まつりの開催", "まつりの屋台"):
            with self.subTest(text=text):
                self.assertFalse(any(is_matsuri_target(target) for _, target in self.hits(text, title="夏色まつり ホロライブ")))

    def test_person_and_event_in_one_cue_are_distinguished_locally(self):
        for text in ("まつりは神社に行った", "まつりと夏祭りに行く", "まつりちゃんと夏祭りに行く",
                     "夏色まつりは夏祭りに行く", "祭りちゃんとお祭りに行く",
                     "まつりはこのお祭りに来る", "まつりがあの祭りへ行く"):
            with self.subTest(text=text):
                names = [(source, target) for source, target in self.hits(text) if is_matsuri_target(target)]
                self.assertEqual(len(names), 1)
                self.assertNotIn("夏祭り", [source for source, _ in names])

    def test_bare_calendar_and_event_reactions_stay_literal(self):
        for previous in ("", "夏祭りの屋台について話そう"):
            for text in ("まつりは明日だ", "まつりが楽しみ", "マツリは来週です", "祭りは明日から"):
                with self.subTest(previous=previous, text=text):
                    self.assertFalse(self.glossary.matsuri_mentions(
                        text, video_context="ホロGTA", previous_ja=previous))
        for text, target in (("まつりさんは明日だ", "Matsuri-san"),
                              ("夏色まつりが楽しみ", "Natsuiro Matsuri")):
            with self.subTest(text=text):
                self.assertTrue(any(item.target == target for item in self.glossary.matsuri_mentions(
                    text, video_context="ホロGTA")))

    def test_nearby_or_recent_event_topic_vetoes_bare_kanji(self):
        for text, previous in (("今日は神社に行く。祭りが来た", ""),
                               ("祭りはどうした", "神社の屋台を見よう"),
                               ("祭りが来た。夏祭りの話だよ", "")):
            with self.subTest(text=text, previous=previous):
                self.assertNotIn(("祭り", "Matsuri"), self.hits(text, title="夏色まつり", previous=previous))

    def test_same_surface_literal_uses_suppress_global_mapping_but_keep_spans(self):
        text = "まつりが来た。神社のまつりもいた"
        mentions = self.glossary.matsuri_mentions(text, video_context="ホロライブ")
        self.assertEqual(len(mentions), 1)
        self.assertEqual((mentions[0].source, mentions[0].target), ("まつり", "Matsuri"))
        self.assertNotIn(("まつり", "Matsuri"), self.hits(text))

    def test_demonstratives_identify_literal_occurrence_in_mixed_cue(self):
        for demonstrative in ("この", "その", "あの", "どの"):
            with self.subTest(demonstrative=demonstrative):
                text = "まつりは" + demonstrative + "まつりに来る"
                mentions = self.glossary.matsuri_mentions(text, video_context="ホロGTA")
                self.assertEqual([(item.start, item.source, item.target) for item in mentions],
                                 [(0, "まつり", "Matsuri")])
                self.assertNotIn(("まつり", "Matsuri"), self.hits(text, title="ホロGTA"))
        for text, source in (("あのまつりさんが来た", "まつりさん"),
                              ("この夏色まつりが来た", "夏色まつり")):
            with self.subTest(text=text):
                self.assertTrue(any(item.source == source for item in self.glossary.matsuri_mentions(
                    text, video_context="ホロGTA")))

    def test_repeated_names_have_separate_immutable_spans(self):
        text = "まつり、マツリちゃん、まつり"
        mentions = self.glossary.matsuri_mentions(text, video_context="ホロライブ")
        self.assertEqual([text[item.start:item.end] for item in mentions], ["まつり", "マツリちゃん", "まつり"])
        self.assertTrue(all(isinstance(item, NameMention) for item in mentions))
        with self.assertRaises(FrozenInstanceError):
            mentions[0].target = "Other"

    def test_combined_api_keeps_both_name_families(self):
        text = "まつりとみこちゃんが来た"
        mentions = self.glossary.name_mentions(text, video_context="ホロライブ")
        self.assertEqual([(item.source, item.target) for item in mentions],
                         [("まつり", "Matsuri"), ("みこちゃん", "Miko-chan")])
        self.assertTrue(all(is_name_target(item.target) for item in mentions))

    def test_reviewed_no_space_iroha_matsuri_bijou_roster(self):
        text = "okいろは祭りあとビジューカリスマ"
        mentions = self.glossary.matsuri_mentions(text, video_context="ホロGTA")
        self.assertEqual([(item.source, item.target, item.evidence) for item in mentions],
                         [("祭り", "Matsuri", "roster_names")])
        self.assertEqual(text[mentions[0].start:mentions[0].end], "祭り")
        self.assertIn(("祭り", "Matsuri"), self.hits(text, title="ホロGTA"))
        for control in ("okいろは祭りあとゲーム", "ok祭りあとビジューカリスマ",
                        "いろは祭りを開催あとビジュー", "いろはお祭りあとビジュー",
                        "いろは祭りあとビジュー、屋台を出す"):
            with self.subTest(control=control):
                self.assertFalse(self.glossary.matsuri_mentions(control, video_context="ホロGTA"))
        self.assertFalse(self.glossary.matsuri_mentions(text, video_context="街のイベント"))

    def test_reviewed_tactical_self_correction_preserves_both_name_spans(self):
        text = "何もいない祭りあー祭りじゃないまとりサーマル使って"
        mentions = self.glossary.matsuri_mentions(text, video_context="ホロGTA")
        self.assertEqual([(item.source, item.target, item.evidence) for item in mentions],
                         [("祭り", "Matsuri", "spoken_self_correction")] * 2)
        self.assertEqual([text[item.start:item.end] for item in mentions], ["祭り", "祭り"])
        self.assertIn("じゃないまとり", text)
        self.assertIn(("祭り", "Matsuri"), self.hits(text, title="ホロGTA"))
        for control in ("祭りあー祭りじゃないまとり", "お祭りあー祭りじゃないまとりサーマル使って",
                        "祭りあー祭りじゃないまとりサーマル使って、屋台を出そう"):
            with self.subTest(control=control):
                self.assertFalse(self.glossary.matsuri_mentions(control, video_context="ホロGTA"))
        self.assertFalse(self.glossary.matsuri_mentions(text, video_context="街のイベント"))

    def test_empty_removed_and_custom_glossary_do_not_synthesize_names(self):
        self.assertEqual(Glossary({}).name_mentions("まつりと祭りちゃん", video_context="夏色まつり"), ())
        glossary = Glossary({"まつり": "custom"})
        self.assertEqual(glossary.find("まつりはどうした", video_context="ホロライブ"), [("まつり", "custom")])
        self.assertEqual(glossary.name_mentions("祭りが来た", video_context="夏色まつり"), ())
        glossary = Glossary({"祭り": "festival"}, nicknames={"まつり": "Matsuri"})
        self.assertEqual(glossary.find("祭り", video_context="夏色まつり"), [("祭り", "festival")])
        self.assertEqual(glossary.matsuri_mentions("祭り", video_context="夏色まつり"), ())


if __name__ == "__main__":
    unittest.main()
