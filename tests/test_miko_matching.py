"""Name recall and literal-role controls for the shared Miko resolver."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from auto_translate.glossary import Glossary


class MikoMatchingTests(unittest.TestCase):
    def setUp(self):
        self.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")

    def hits(self, ja, *, title="ホロGTA", previous=""):
        return self.glossary.find(ja, previous_ja=previous, video_context=title)

    def test_kana_aliases_and_spoken_honorifics(self):
        for source, target in (("みこ", "Miko"), ("ミコ", "Miko"),
                               ("みこち", "Mikochi"), ("ミコチ", "Mikochi"),
                               ("みこちゃん", "Miko-chan"), ("ミコちゃん", "Miko-chan"),
                               ("みこさん", "Miko-san"), ("ミコさん", "Miko-san"),
                               ("みこ様", "Miko-sama"), ("ミコさま", "Miko-sama")):
            with self.subTest(source=source):
                self.assertIn((source, target), self.hits(source + "はどうした？"))

    def test_common_kana_honorifics_do_not_establish_miko_identity(self):
        for source in ("みこさん", "ミコさん", "みこ様", "ミコさま", "みこちゃん"):
            with self.subTest(source=source):
                self.assertEqual(self.hits(source + "が来た", title="日本文化の見学"), [])
                self.assertIn((source, self.glossary.nicknames[source]),
                              self.hits(source + "が来た", title="ホロGTA"))
                self.assertIn((source, self.glossary.nicknames[source]),
                              self.hits(source + "が来た", title="Minecraft", previous="さくらみこだよ"))
                self.assertNotIn(("巫女", "Miko"), self.hits(
                    "巫女が来た", title="日本文化の見学", previous="神社について。" + source + "が来た"))
                self.assertNotIn(("巫女", "Miko"), self.hits(
                    "巫女が来た", title="ホロGTA", previous=source + "が来た"))
        self.assertIn(("みこち", "Mikochi"), self.hits("みこちが来た", title=""))
        self.assertIn(("ミコチ", "Mikochi"), self.hits("ミコチが来た", title=""))

    def test_full_name_variants_do_not_require_metadata(self):
        for source in ("さくらみこ", "さくらミコ", "サクラみこ", "サクラミコ"):
            with self.subTest(source=source):
                self.assertIn((source, "Sakura Miko"), self.hits(source + "です", title=""))
        self.assertIn(("さくらみこ", "Sakura Miko"), self.hits("神社のさくらみこは元気", title=""))

    def test_full_name_variants_preserve_spoken_honorifics(self):
        for spelling in ("さくらみこ", "さくらミコ", "サクラみこ", "サクラミコ"):
            for honorific, target in (("ちゃん", "Sakura Miko-chan"), ("さん", "Sakura Miko-san"),
                                      ("様", "Sakura Miko-sama"), ("さま", "Sakura Miko-sama")):
                source = spelling + honorific
                with self.subTest(source=source):
                    line = source + "が来た"
                    mentions = self.glossary.miko_mentions(line)
                    self.assertEqual([(item.source, item.target) for item in mentions], [(source, target)])
                    self.assertEqual(self.hits(line, title=""), [(source, target)])

    def test_natural_particle_continuations(self):
        for line in ("みこはどうした", "ミコがいなくなった", "みこによろしく", "みこもやってる", "みこと話す"):
            with self.subTest(line=line):
                self.assertTrue(any(target == "Miko" for _, target in self.hits(line)))

    def test_bare_name_needs_metadata_or_recent_japanese(self):
        self.assertFalse(self.hits("みこが来た", title="Minecraft"))
        self.assertIn(("みこ", "Miko"), self.hits("みこが来た", title="Minecraft", previous="さくらみこだよ\n了解"))
        self.assertIn(("ミコ", "Miko"), self.hits("ミコはどうした", title="Minecraft", previous="みこちがいた"))
        self.assertIn(("みこ", "Miko"), self.hits("みこが来た", title="Minecraft", previous="みこが戻った"))
        self.assertFalse(self.hits("みこが来た", title="Minecraft", previous="神社のみこだよ"))
        self.assertFalse(self.hits("みこが来た", title="Minecraft", previous="みこです"))
        self.assertFalse(self.hits("みこが来た", title="文化見学", previous="神社について。みこが来た"))

    def test_substrings_and_other_words_are_not_names(self):
        for word in ("みこしが来た", "みこと", "申しこみこみ", "ミコト", "みこみ", "御みこし"):
            with self.subTest(word=word):
                self.assertFalse(any("Miko" in target for _, target in self.hits(word)))

    def test_local_literal_roles_do_not_veto_a_separate_name(self):
        self.assertIn(("みこ", "Miko"), self.hits("みこは巫女アイドルです"))
        self.assertIn(("ミコ", "Miko"), self.hits("ミコは神社に行った"))
        self.assertIn(("みこちゃん", "Miko-chan"), self.hits("みこちゃんと神社に行く"))
        self.assertIn(("みこ", "Miko"), self.hits("みこの衣装はかわいい"))
        for line in ("神社のみこ", "神社のミコ", "みこ装束", "エリート巫女アイドル", "巫女さんが来た"):
            with self.subTest(line=line):
                self.assertFalse(any("Miko" in target for _, target in self.hits(line, title="さくらみこ ホロライブ")))

    def test_kanji_homophone_requires_strong_identity_and_person_syntax(self):
        self.assertNotIn(("巫女", "Miko"), self.hits("巫女が来た"))
        self.assertIn(("巫女", "Miko"), self.hits("巫女が来た", title="Sakura Miko Ch."))
        self.assertIn(("巫女", "Miko"), self.hits("巫女はどうした", title="Minecraft", previous="みこちがいるよ"))
        self.assertNotIn(("巫女", "Miko"), self.hits("巫女が来た", title="Minecraft", previous="みこが来た"))
        self.assertNotIn(("巫女", "Miko"), self.hits("巫女です", title="さくらみこ"))

    def test_mixed_same_surface_keeps_spans_without_global_glossary_pair(self):
        line = "みこが来た。神社のみこもいた"
        mentions = self.glossary.miko_mentions(line, video_context="Sakura Miko Ch.")
        self.assertEqual(len(mentions), 1)
        self.assertEqual((mentions[0].source, mentions[0].target), ("みこ", "Miko"))
        self.assertNotIn(("みこ", "Miko"), self.hits(line, title="Sakura Miko Ch."))

    def test_kanji_stays_ambiguous_during_nearby_shrine_discussion(self):
        for line, previous in (
            ("今日は神社に行く。巫女が来た", ""),
            ("巫女が来た。神社に行く", ""),
            ("巫女が来た。神社の巫女さんもいた", ""),
            ("巫女が来た", "今日は神社に行く\n了解"),
            ("巫女はどこ", "みこちがいるよ\n巫女の仕事を見に行く"),
        ):
            with self.subTest(line=line, previous=previous):
                self.assertEqual(self.glossary.miko_mentions(
                    line, previous_ja=previous, video_context="さくらみこ ホロライブ"), ())
                self.assertNotIn(("巫女", "Miko"), self.hits(line, title="さくらみこ ホロライブ", previous=previous))
        # Explicit kana names still work in the same discussion; only the
        # ambiguous kanji interpretation is withheld.
        self.assertIn(("みこ", "Miko"), self.hits("みこは巫女アイドルです", previous="神社の話"))
        self.assertIn(("ミコ", "Miko"), self.hits("神社に行く。ミコが来た"))

    def test_old_shrine_topic_does_not_override_new_name_context_forever(self):
        previous = "昨日は神社に行った。" + "普通のゲームの話です。" * 40 + "さくらみこだよ"
        self.assertIn(("巫女", "Miko"), self.hits("巫女が来た", title="Minecraft", previous=previous))

    def test_mentions_are_immutable_exact_source_spans(self):
        line = "みこちとミコさんが来た"
        mentions = self.glossary.miko_mentions(line, video_context="ホロライブ")
        self.assertEqual([(line[item.start:item.end], item.source, item.target) for item in mentions],
                         [("みこち", "みこち", "Mikochi"), ("ミコさん", "ミコさん", "Miko-san")])
        with self.assertRaises(FrozenInstanceError):
            mentions[0].target = "Other"

    def test_empty_or_custom_glossary_does_not_synthesize_names(self):
        self.assertEqual(Glossary({}).miko_mentions("みこちと巫女が来た", video_context="さくらみこ"), ())
        self.assertEqual(Glossary({"みこ": "custom"}).find("みこが来た", video_context="さくらみこ"), [("みこ", "custom")])
        self.assertFalse(Glossary({"さくらみこ": "Sakura Miko"}).miko_mentions("巫女が来た", video_context="さくらみこ"))

    def test_legacy_context_keyword_remains_supported(self):
        self.assertIn(("みこ", "Miko"), self.glossary.find("みこはどうした", miko_context=True))


if __name__ == "__main__":
    unittest.main()
