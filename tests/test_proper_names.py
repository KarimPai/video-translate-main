"""Deterministic source-text safety checks, not acoustic-accuracy evidence."""
from pathlib import Path
import unittest
from unittest.mock import patch

from auto_translate.glossary import Glossary
from auto_translate.name_hints import preserve_spoken_names
from auto_translate.proper_names import (
    COMMON_NOUN, PERSON, UNKNOWN, name_glossary, resolve_names, reviewed_spoken_name_forms,
)


class ProperNameResolutionTests(unittest.TestCase):
    def test_confirmed_matsuri_spelling_variants(self):
        for spelling in ("まつり", "マツリ", "祭り"):
            with self.subTest(spelling=spelling):
                mentions = resolve_names(spelling + "が来た", previous_ja="夏色まつりだよ")
                self.assertEqual((mentions[0].classification, mentions[0].target), (PERSON, "Matsuri"))

    def test_title_and_participant_alone_cannot_create_a_person(self):
        for title in ("", "ホロライブ 夏色まつり さくらみこ"):
            for text in ("まつりが来た", "みこが来た"):
                with self.subTest(title=title, text=text):
                    self.assertEqual(resolve_names(text, video_title=title)[0].classification, UNKNOWN)
        self.assertEqual(resolve_names("今日は晴れ", confirmed_participants=("Matsuri", "Miko")), [])

    def test_reviewed_participant_plus_spoken_person_syntax(self):
        mention = resolve_names("みこが来た", confirmed_participants=("Sakura Miko",))[0]
        self.assertEqual((mention.classification, mention.target), (PERSON, "Miko"))
        self.assertIn("verified_participant", mention.evidence)

    def test_actual_festivals_remain_literal_with_person_context(self):
        for text in ("お祭りに行く", "夏祭りが来た", "おまつりに行く", "文化祭に行く"):
            with self.subTest(text=text):
                mentions = resolve_names(text, previous_ja="夏色まつりだよ")
                self.assertTrue(mentions)
                self.assertTrue(all(item.classification == COMMON_NOUN for item in mentions))
                self.assertEqual(name_glossary(mentions), [])

    def test_matsuri_and_festival_in_one_sentence(self):
        mentions = resolve_names("まつりが文化祭に来た", previous_ja="夏色まつりの話だよ")
        self.assertEqual([(item.source, item.classification) for item in mentions],
                         [("まつり", PERSON), ("祭", COMMON_NOUN)])
        self.assertEqual(name_glossary(mentions), [("まつり", "Matsuri")])

    def test_repeated_spelling_with_two_meanings_never_gets_global_glossary(self):
        ja = "まつりがおまつりに行く"
        mentions = resolve_names(ja, previous_ja="まつりちゃんがいる")
        self.assertEqual([item.classification for item in mentions], [PERSON, COMMON_NOUN])
        self.assertEqual(name_glossary(mentions), [])
        self.assertEqual([ja[item.start:item.end] for item in mentions], ["まつり", "まつり"])
        mentions = resolve_names("まつりがまつりに来た", previous_ja="まつりちゃんがいる")
        self.assertEqual([item.classification for item in mentions], [PERSON, COMMON_NOUN])
        self.assertEqual(name_glossary(mentions), [])

    def test_miko_and_shrine_maiden_in_one_clause(self):
        mentions = resolve_names("みこが巫女さんと神社にいる", previous_ja="みこちだよ")
        self.assertEqual([(item.source, item.classification) for item in mentions],
                         [("みこ", PERSON), ("巫女さん", COMMON_NOUN)])
        self.assertEqual(name_glossary(mentions), [("みこ", "Miko")])

    def test_shrine_maiden_remains_literal(self):
        for ja in ("巫女さんが神社にいる", "神社のみこ", "神社の巫女"):
            with self.subTest(ja=ja):
                self.assertTrue(all(item.classification == COMMON_NOUN
                                    for item in resolve_names(ja, video_title="ホロライブ さくらみこ")))

    def test_confirmed_kanji_person_is_validated(self):
        mentions = resolve_names("巫女が来た", previous_ja="さくらみこだよ")
        self.assertEqual(name_glossary(mentions), [("巫女", "Miko")])
        self.assertEqual(preserve_spoken_names("巫女が来た", "The shrine maiden arrived.",
                                              name_glossary(mentions), mentions=mentions), "Miko arrived.")

    def test_full_names_are_preserved_only_when_spoken(self):
        self.assertEqual(resolve_names("夏色まつりがいる")[0].target, "Natsuiro Matsuri")
        self.assertEqual(resolve_names("さくらみこがいる")[0].target, "Sakura Miko")
        self.assertEqual(resolve_names("桜巫女がいる")[0].target, "Sakura Miko")
        self.assertEqual(resolve_names("まつりちゃん")[0].target, "Matsuri-chan")

    def test_required_short_forms_and_honorifics(self):
        cases = {"ラプ": "Lap", "ラプ様": "Lap-sama", "ラプちゃん": "Lap-chan",
                 "ラプラス": "Laplus", "みこち": "Mikochi", "みこちゃん": "Miko-chan",
                 "まつりちゃん": "Matsuri-chan", "わため": "Watame"}
        for ja, target in cases.items():
            with self.subTest(ja=ja):
                self.assertEqual(resolve_names(ja)[0].target, target)

    def test_multiple_distinct_names(self):
        mentions = resolve_names("ラプとみこちとまつりちゃんが来た")
        self.assertEqual([item.target for item in mentions], ["Lap", "Mikochi", "Matsuri-chan"])

    def test_verified_en_id_short_forms_do_not_expand(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for source, target in (("ムーナ", "Moona"), ("カエラ", "Kaela"), ("モココ", "Mococo")):
            with self.subTest(source=source):
                self.assertEqual(resolve_names(source)[0].target, target)
                self.assertIn((source, target), glossary.find(source))

    def test_person_glossary_never_matches_inside_a_literal_spelling(self):
        rows = [{"source": "みこ", "target": "Miko", "classification": PERSON},
                {"source": "みこさん", "target": None, "classification": COMMON_NOUN}]
        self.assertEqual(name_glossary(rows), [])

    def test_common_words_do_not_become_names(self):
        for ja in ("ストラップ", "ラプソディー", "サランラップ", "申しこみこみ", "みこと", "みこし", "わためいと", "まつり縫い"):
            with self.subTest(ja=ja):
                self.assertEqual(resolve_names(ja, video_title="ホロライブ"), [])


    def test_prior_literal_noun_is_not_confirmed_person_context(self):
        self.assertEqual(resolve_names("まつりが来た", previous_ja="夏祭りの話だよ")[0].classification, UNKNOWN)




    def test_glossary_uses_resolver_without_title_boolean_bypass(self):
        glossary = Glossary.load(Path("glossary.toml"))
        self.assertNotIn(("みこ", "Miko"), glossary.find("みこが来た", miko_context=True))
        self.assertIn(("みこ", "Miko"), glossary.find("みこが来た", previous_ja="みこちだよ"))
        self.assertIn(("みこちゃん", "Miko-chan"), glossary.find("みこちゃん"))

    def test_mixed_literal_output_is_never_globally_repaired(self):
        ja = "みこが巫女さんと神社にいる"
        mentions = resolve_names(ja, previous_ja="みこちだよ")
        english = "The shrine maiden is with a shrine maiden at the shrine."
        self.assertEqual(preserve_spoken_names(ja, english, name_glossary(mentions), mentions=mentions), english)

    def test_botan_glossary_cannot_bypass_occurrence_evidence(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for source in ("ボタン", "ぼたん"):
            with self.subTest(source=source):
                self.assertNotIn((source, "Botan"), glossary.find(source + "が来た"))
                self.assertIn((source, "Botan"), glossary.find(
                    source + "が来た", previous_ja="ししろんだよ"))
                self.assertNotIn((source, "Botan"), glossary.find(
                    source + "を押して", previous_ja="獅白ぼたんだよ"))
        self.assertIn(("ボタンちゃん", "Botan-chan"), glossary.find("ボタンちゃんが来た"))

    def test_mixed_button_name_spelling_is_not_globally_repaired(self):
        ja = "ボタンがボタンを押した"
        mentions = resolve_names(ja, previous_ja="ししろんだよ")
        self.assertEqual([m.classification for m in mentions], [PERSON, COMMON_NOUN])
        self.assertEqual(name_glossary(mentions), [])
        original = "The button pressed the button."
        self.assertEqual(preserve_spoken_names(ja, original, [], mentions=mentions), original)

    def test_reviewed_latin_forms_do_not_double_honorifics(self):
        forms = reviewed_spoken_name_forms()
        for target in ("Botan", "Botan-chan", "Matsuri-san", "Sui-chan", "Moona"):
            self.assertIn(target, forms)
        self.assertNotIn("Sui-chan-chan", forms)
        self.assertNotIn("Sakura Mikochi", forms)

    def test_plural_literal_repairs_do_not_break_verb_agreement(self):
        ja = "まつりちゃんはここにいない"
        mentions = resolve_names(ja)
        for english in ("Festivals aren't here.", "The festivals have left.", "Festivals' hats are here."):
            with self.subTest(english=english):
                self.assertEqual(preserve_spoken_names(ja, english, name_glossary(mentions), mentions=mentions), english)

    def test_bounded_group_endings_keep_the_spoken_name_span(self):
        for source, target in (("まつりちゃん", "Matsuri-chan"), ("みこちゃん", "Miko-chan")):
            for ending in ("たち", "達", "ら"):
                with self.subTest(source=source, ending=ending):
                    ja = source + ending + "が来た"
                    mentions = resolve_names(ja)
                    self.assertEqual([(m.source, m.classification, m.target) for m in mentions],
                                     [(source, PERSON, target)])
                    self.assertEqual(ja[mentions[0].start:mentions[0].end], source)
        for ja in ("まつりたちばな", "みこちゃんたちばな", "まつりちゃんらしい"):
            with self.subTest(ja=ja):
                self.assertEqual(resolve_names(ja), [])

    def test_bare_group_references_still_need_person_evidence(self):
        for source, prior, target in (("まつり", "夏色まつりだよ", "Matsuri"),
                                      ("みこ", "みこちだよ", "Miko")):
            for ending in ("たち", "達", "ら"):
                with self.subTest(source=source, ending=ending):
                    ja = source + ending + "が来た"
                    self.assertEqual(resolve_names(ja, video_title=prior)[0].classification, UNKNOWN)
                    mention = resolve_names(ja, previous_ja=prior)[0]
                    self.assertEqual((mention.source, mention.classification, mention.target),
                                     (source, PERSON, target))

    def test_literal_honorific_does_not_seed_prior_or_local_person_context(self):
        literal = "神社のみこさんが神楽を舞った"
        self.assertEqual(resolve_names(literal)[0].classification, COMMON_NOUN)
        mention = resolve_names("みこが来た", previous_ja=literal)[0]
        self.assertEqual(mention.classification, UNKNOWN)
        self.assertNotIn("recent_confirmed_name", mention.evidence)
        for ja, expected in ((literal + "。みこが来た。", [COMMON_NOUN, UNKNOWN]),
                             ("みこが来た。" + literal + "。", [UNKNOWN, COMMON_NOUN]),
                             ("おまつりさんがいる。まつりが来た。", [COMMON_NOUN, UNKNOWN])):
            with self.subTest(ja=ja):
                mentions = resolve_names(ja)
                self.assertEqual([m.classification for m in mentions], expected)
                self.assertNotIn("local_explicit_name", mentions[expected.index(UNKNOWN)].evidence)

    def test_real_explicit_evidence_survives_literal_honorific_exclusion(self):
        for prior in ("さくらみこだよ", "みこちだよ", "みこさんが来た"):
            with self.subTest(prior=prior):
                self.assertEqual(resolve_names("みこが来た", previous_ja=prior)[0].classification, PERSON)
                ja = prior + "。神社のみこさんが神楽を舞った。みこが来た。"
                self.assertEqual([m.classification for m in resolve_names(ja)],
                                 [PERSON, COMMON_NOUN, PERSON])
        self.assertEqual(resolve_names("まつりが開催された", previous_ja="まつりちゃんだよ")[0].classification,
                         COMMON_NOUN)

    def test_prior_evidence_evaluation_is_bounded_and_shared(self):
        previous = "以前の話。" * 200 + "みこちだよ"
        with patch("auto_translate.proper_names.resolve_names", wraps=resolve_names) as resolver:
            mentions = resolver("みこが来た。みこはいる。", previous_ja=previous)
        self.assertTrue(all(m.classification == PERSON for m in mentions))
        self.assertEqual(resolver.call_count, 2)
        self.assertLessEqual(len(resolver.call_args_list[1].args[0]), 600)

    def test_punctuation_vocatives_require_context_and_an_actual_imperative(self):
        for ja in ("まつり、こっち来て！", "まつり, こっち来て！", "まつり、待って！"):
            with self.subTest(ja=ja):
                self.assertEqual(resolve_names(ja, previous_ja="夏色まつりだよ")[0].classification, PERSON)
                self.assertEqual(resolve_names(ja)[0].classification, UNKNOWN)
                self.assertEqual(resolve_names(ja, video_title="夏色まつり ホロライブ")[0].classification, UNKNOWN)
        for ja in ("まつり、明日は雨だ。", "まつり、こっち来てる。"):
            with self.subTest(ja=ja):
                self.assertEqual(resolve_names(ja, previous_ja="夏色まつりだよ")[0].classification, UNKNOWN)

    def test_common_noun_repairs_cannot_move_a_name_to_another_sentence(self):
        cases = (
            ("まつりちゃんは来ない。イベントは明日からだ。", "She isn't coming. The festival starts tomorrow."),
            ("まつりちゃんは来ない。催しは明日からだ。", "She isn't coming. The festival starts tomorrow."),
            ("みこちゃんは来ない。神職は明日からだ。", "She isn't coming. The shrine maiden starts tomorrow."),
            ("ボタンちゃんは来ない。スイッチを押して。", "She isn't coming. Press the button."),
        )
        for ja, english in cases:
            with self.subTest(ja=ja):
                mentions = resolve_names(ja)
                for supplied in (mentions, None):
                    self.assertEqual(preserve_spoken_names(ja, english, name_glossary(mentions), mentions=supplied),
                                     english)
                self.assertTrue(all(ja[m.start:m.end] == m.source for m in mentions))

    def test_abbreviation_fragments_cannot_redirect_common_noun_repairs(self):
        for source, noun, literal in (("まつりちゃん", "festival", "催しについて話した"),
                                      ("みこちゃん", "shrine maiden", "神職と話した"),
                                      ("ボタンちゃん", "button", "スイッチを押した")):
            ja = "田中教授は" + literal + "。" + source + "は来ない。"
            mentions = resolve_names(ja)
            for title in ("Prof.", "Assoc.", "Capt."):
                english = title + " Tanaka mentioned the " + noun + "; not coming."
                with self.subTest(source=source, title=title):
                    self.assertEqual(preserve_spoken_names(ja, english, name_glossary(mentions), mentions=mentions),
                                     english)

    def test_aligned_name_sentence_can_be_repaired_without_touching_other_sentences(self):
        for source, target, noun in (("まつりちゃん", "Matsuri-chan", "festival"),
                                     ("みこちゃん", "Miko-chan", "shrine maiden"),
                                     ("ボタンちゃん", "Botan-chan", "button")):
            with self.subTest(source=source):
                ja = source + "は来ない。天気は晴れだ。"
                english = "The " + noun + " isn't coming. The weather is clear."
                mentions = resolve_names(ja)
                self.assertEqual(preserve_spoken_names(ja, english, name_glossary(mentions), mentions=mentions),
                                 target + " isn't coming. The weather is clear.")

    def test_ambiguous_multisentence_noun_repairs_abstain_but_spelling_repairs_remain_safe(self):
        for source, target, noun in (("まつりちゃん", "Matsuri-chan", "festival"),
                                     ("みこちゃん", "Miko-chan", "shrine maiden"),
                                     ("ボタンちゃん", "Botan-chan", "button")):
            ja = source + "は来ない。天気は晴れだ。"
            mentions = resolve_names(ja)
            names = name_glossary(mentions)
            for english in ("The " + noun + " isn't coming in this clear weather.",
                            "The " + noun + " isn't coming. The weather is clear. Wait!",
                            "The " + noun + " isn't coming. The weather is clear"):
                with self.subTest(source=source, english=english):
                    self.assertEqual(preserve_spoken_names(ja, english, names, mentions=mentions), english)
            english = target.removesuffix("-chan") + " isn't coming in this clear weather."
            self.assertEqual(preserve_spoken_names(ja, english, names, mentions=mentions),
                             target + " isn't coming in this clear weather.")


if __name__ == "__main__":
    unittest.main()
