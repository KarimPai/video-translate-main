"""Offline tests for Hy-MT2 prompt construction and context-gated glossary hints."""
from pathlib import Path
import tempfile
import unittest

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext, describe_stream
from auto_translate.translate.hymt import (
    HymtTranslator, build_context, build_prompt, clean_output, garbled_name, missing_names,
)
from test_english_reliability import fake_translator

GLOSSARY = Glossary.load(Path("glossary.toml"))


class PromptTemplateTests(unittest.TestCase):
    def test_plain_template_without_terms_or_context(self):
        self.assertEqual(build_prompt("こんにちは", [], ""),
                         "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\nこんにちは")

    def test_terminology_template(self):
        prompt = build_prompt("ぺこらだよ", [("ぺこら", "Pekora")], "")
        self.assertEqual(prompt, "参考下面的翻译：\nぺこら 翻译成 Pekora\n"
                                 "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\nぺこらだよ")

    def test_background_template_combined_with_terms(self):
        prompt = build_prompt("草", [("草", "lol")], "前の文\nEarlier line.")
        # Context leads so the growing dialogue stays a cacheable prefix.
        self.assertTrue(prompt.startswith("【背景信息】\n前の文\nEarlier line.\n\n参考下面的翻译：\n草 翻译成 lol\n请结合"))
        self.assertTrue(prompt.endswith("不要额外解释。\n\n【待翻译文本】\n草"))
        self.assertEqual(build_prompt("草", [], "前の文"),
                         "【背景信息】\n前の文\n\n请结合背景信息将以下文本翻译为英语，"
                         "注意只需要输出翻译后的结果，不要额外解释。\n\n【待翻译文本】\n草")

    def test_style_template_with_and_without_context(self):
        self.assertEqual(build_prompt("草", [("草", "lol")], "", "简洁直白"),
                         "参考下面的翻译：\n草 翻译成 lol\n"
                         "请将以下文本翻译为 英语。\n注意翻译的风格要严格符合【简洁直白】\n\n草")
        self.assertEqual(build_prompt("草", [], "前の文", "简洁直白"),
                         "【背景信息】\n前の文\n\n请结合背景信息将以下文本翻译为 英语。\n"
                         "注意翻译的风格要严格符合【简洁直白】\n\n【待翻译文本】\n草")

    def test_style_config_validation(self):
        self.assertEqual(TranslateConfig().style, "简洁直白")
        self.assertEqual(TranslateConfig(style="").style, "")
        for bad in ("【x】", "a\nb", 3, "x" * 41):
            with self.subTest(style=bad), self.assertRaises(ValueError):
                TranslateConfig(style=bad)

    def test_context_keeps_recent_japanese_only_within_budget(self):
        history = [(f"文{i}", f"Line {i}.") for i in range(6)]
        self.assertEqual(build_context("Title", history, 3, 300), "Title\n直前の会話：\n文3\n文4\n文5")
        self.assertEqual(build_context("", history, 3, 4), "文5")
        self.assertEqual(build_context("Title", history, 0, 300), "Title")

    def test_previous_english_is_opt_in_and_skips_failed_lines(self):
        history = [("文1", "Line 1."), ("文2", "")]
        self.assertEqual(build_context("", history, 2, 300, english=True), "文1（英訳：Line 1.）\n文2")
        self.assertEqual(build_context("", history, 2, 300), "文1\n文2")

    def test_header_label_counts_against_budget(self):
        history = [("あいうえお", "")]
        self.assertEqual(build_context("Title", history, 1, 18), "Title\n直前の会話：\nあいうえお")
        self.assertEqual(build_context("Title", history, 1, 17), "Title")

    def test_oversized_recent_source_does_not_discard_older_fitting_source(self):
        history = [("すいちゃんが来た", "Suixian arrived."), ("長" * 100, "Long."), ("はい！", "Yes!")]
        self.assertEqual(build_context("", history, 3, 20), "すいちゃんが来た")

    def test_clean_output_joins_lines_and_drops_role_prefix(self):
        self.assertEqual(clean_output("Translation: Hello\nthere."), "Hello there.")
        self.assertEqual(clean_output("「Hello!」"), "Hello!")


class GlossaryHintTests(unittest.TestCase):
    def test_full_names_always_hinted(self):
        self.assertIn(("兎田ぺこら", "Usada Pekora"), GLOSSARY.find("兎田ぺこらです"))

    def test_ambiguous_nickname_needs_talent_in_context(self):
        self.assertNotIn(("ボタン", "Botan"), GLOSSARY.find("ボタンを押して"))
        self.assertIn(("ボタン", "Botan"),
                      GLOSSARY.find("ボタンが来た", context="獅白ぼたん 雑談\nボタンが来た"))

    def test_miko_blocked_in_shrine_context(self):
        context = "さくらみこ 配信"
        self.assertNotIn(("みこ", "Miko"), GLOSSARY.find("神社のみこがいる", context=context))

    def test_senchou_resolves_by_context(self):
        self.assertIn(("船長", "captain"), GLOSSARY.find("船長、出航の準備はできた？"))
        self.assertNotIn("船長", [ja for ja, _ in GLOSSARY.find("船長が来た")])
        self.assertIn(("船長", "Senchou"), GLOSSARY.find("船長が来た", context="宝鐘マリン 配信"))

    def test_proper_names_exclude_ambiguous_words(self):
        names = GLOSSARY.proper_names()
        self.assertIn("兎田ぺこら", names)
        self.assertNotIn("ボタン", names)
        self.assertNotIn("まつり", names)

    def test_asr_name_context_is_focused_on_metadata(self):
        names = GLOSSARY.proper_names("【宝鐘マリン】雑談")
        self.assertIn("宝鐘マリン", names)
        self.assertNotIn("兎田ぺこら", names)
        self.assertNotIn("ボタン", names)
        self.assertEqual(GLOSSARY.proper_names("宝鐘マリンとさくらみこ"), ["宝鐘マリン", "さくらみこ"])
        self.assertEqual(GLOSSARY.proper_names("兎田ぺこら"), ["兎田ぺこら"])
        self.assertEqual(GLOSSARY.proper_names("無関係な配信"), [])

    def test_translator_prompt_uses_title_for_terms_and_context(self):
        obj = fake_translator(glossary=GLOSSARY)
        obj.stream = StreamContext(title="【宝鐘マリン】雑談")
        obj.cfg = TranslateConfig(context_lines=2)
        prompt = obj.prompt_for("船長が来た", [("おはよう", "Good morning."), ("元気？", "How are you?")])
        self.assertIn("船長 翻译成 Senchou", prompt)
        self.assertTrue(prompt.startswith("【背景信息】\n配信タイトル：【宝鐘マリン】雑談\n直前の会話：\nおはよう\n元気？\n\n"))
        self.assertNotIn("Good morning.", prompt)
        self.assertNotIn("How are you?", prompt)
        self.assertTrue(prompt.endswith("船長が来た"))

    def test_stream_profile_chapter_and_cast_lead_the_prompt(self):
        stream = describe_stream({"title": "【コラボ】雑談", "channel": "Marine Ch. 宝鐘マリン",
                                  "tags": ["兎田ぺこら"], "description": "ゲスト：星街すいせい\n宝鐘マリン",
                                  "chapters": [{"start_s": 600, "title": "マイクラ"},
                                               {"start_s": 0, "title": "オープニング"}]}, GLOSSARY)
        self.assertEqual((stream.host, stream.cast), ("宝鐘マリン", ("兎田ぺこら", "星街すいせい")))
        self.assertEqual(stream.chapter_at(None), "")
        self.assertEqual(stream.chapter_at(30), "オープニング")
        self.assertEqual(stream.chapter_at(900), "マイクラ")
        self.assertEqual(stream.lines(900), [
            "配信タイトル：【コラボ】雑談", "チャンネル：Marine Ch. 宝鐘マリン",
            GLOSSARY.host_line("宝鐘マリン"), "共演：兎田ぺこら、星街すいせい", "現在のチャプター：マイクラ"])
        self.assertIn("自称「船長」「マリン」は配信者本人", stream.lines()[2])
        self.assertEqual(stream.recognition_text(),
                         "【コラボ】雑談\nMarine Ch. 宝鐘マリン\n宝鐘マリン\n兎田ぺこら\n星街すいせい")
        obj = fake_translator(glossary=GLOSSARY)
        obj.stream = stream
        obj.set_position(900)
        prompt = obj.prompt_for("船長が来た", [])
        self.assertTrue(prompt.startswith("【背景信息】\n配信タイトル：【コラボ】雑談\n"))
        self.assertIn("現在のチャプター：マイクラ", prompt)
        self.assertIn("船長 翻译成 Senchou", prompt)

    def test_stream_without_metadata_or_glossary_is_empty(self):
        self.assertEqual(describe_stream(None), StreamContext())
        self.assertEqual(describe_stream({"title": "  a \n b  "}).lines(), ["配信タイトル：a b"])
        self.assertEqual(describe_stream({"channel": "Unknown Ch."}, GLOSSARY).host, "")

    def test_dialogue_window_grows_then_restarts_at_half(self):
        obj = fake_translator()
        obj.cfg = TranslateConfig(context_lines=4)
        history = []
        windows = []
        for i in range(7):
            windows.append([ja for ja, _ in obj._window(history)])
            history = (history + [(f"文{i}です", "")])[-4:]
        self.assertEqual(windows, [[], ["文0です"], ["文0です", "文1です"], ["文0です", "文1です", "文2です"],
                                   ["文0です", "文1です", "文2です", "文3です"], ["文3です", "文4です"],
                                   ["文3です", "文4です", "文5です"]])
        # A history reset (seek or gap) starts from the new lines.
        self.assertEqual(obj._window([("別の話です", "")]), [("別の話です", "")])

    def test_kana_aliases_keep_source_spelling_and_canonical_identity(self):
        for spoken, translated, canonical in (
            ("スイちゃん", "Sui-chan", "星街すいせい"),
            ("ミコチ", "Mikochi", "さくらみこ"),
            ("シシロン", "Shishiron", "獅白ぼたん"),
            ("ルイちゃん", "Lui-chan", "鷹嶺ルイ"),
            ("カリオペ", "Calliope", "森カリオペ"),
        ):
            with self.subTest(spoken=spoken):
                self.assertIn((spoken, translated), GLOSSARY.name_terms(f"昨日{spoken}が来た。"))
                self.assertEqual(GLOSSARY.proper_names(spoken), [canonical])
                self.assertIn(canonical, GLOSSARY.recognition_names(spoken))

    def test_misheard_suisei_is_a_name_unless_planets_are_discussed(self):
        for source, pair in (("犯人は水星先輩らしいって。", ("水星先輩", "Suisei-senpai")),
                             ("あの殺された水星さん。", ("水星さん", "Suisei-san")),
                             ("水星が来た。", ("水星", "Suisei")),
                             ("すいせい先輩が来た。", ("すいせい先輩", "Suisei-senpai"))):
            with self.subTest(source=source):
                self.assertIn(pair, GLOSSARY.name_terms(source))
                self.assertEqual(GLOSSARY.proper_names(source), ["星街すいせい"])
        for source in ("水星の魔女見た？", "水星と金星は惑星だよ"):
            with self.subTest(source=source):
                self.assertEqual(GLOSSARY.name_terms(source), [])

    def test_gta_topic_terms_apply_only_to_gta_streams(self):
        gta = GLOSSARY.for_stream("【#holoGTA】勤務だ！DAY3")
        for source, pair in (("店舗強盗行ける人いますか。", ("店舗強盗", "store robbery")),
                             ("店舗ゴートめちゃめちゃいけ。", ("店舗ゴート", "store robbery")),
                             ("締め手配しちゃった。", ("締め手配", "wanted"))):
            with self.subTest(source=source):
                self.assertIn(pair, gta.find(source))
                self.assertNotIn(pair, GLOSSARY.find(source))
        self.assertIn(("水星先輩", "Suisei-senpai"), gta.name_terms("犯人は水星先輩らしいって。"))
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "g.toml"
            path.write_text('[topics.x]\nasr = ["a"]\n', encoding="utf-8")
            Glossary.load(path)

    def test_joined_names_and_false_compounds(self):
        terms = GLOSSARY.name_terms("ルイちゃんカリオペがスイちゃんが撃ったって言った。")
        for pair in (("ルイちゃん", "Lui-chan"), ("カリオペ", "Calliope"), ("スイちゃん", "Sui-chan")):
            self.assertIn(pair, terms)
        for text in ("ルイージが来た", "ボタンを押して", "祭りに行こう", "みこという言葉"):
            with self.subTest(text=text):
                self.assertEqual(GLOSSARY.name_terms(text), [])

    def test_literal_button_and_festival_override_matching_talent_context(self):
        for source, title in (("ボタンを押して。", "獅白ぼたん"), ("電源ボタンがある。", "獅白ぼたん"),
                              ("ぼたんをおす。", "獅白ぼたん"), ("祭りに行こう。", "夏色まつり"),
                              ("まつりへいこう。", "夏色まつり")):
            with self.subTest(source=source):
                self.assertEqual(GLOSSARY.name_terms(source, context=title), [])
        self.assertIn(("ボタン", "Botan"), GLOSSARY.name_terms("ボタンが来た。", context="獅白ぼたん"))
        self.assertIn(("まつり", "Matsuri"), GLOSSARY.name_terms("まつりが来た。", context="夏色まつり"))
        obj = fake_translator("Press the button.")
        obj.stream = StreamContext(title="獅白ぼたん")
        self.assertEqual(obj.translate("ボタンを押して。", []), "Press the button.")
        obj._chat.assert_called_once()

    def test_generic_vocabulary_does_not_use_kana_folding(self):
        self.assertNotIn(("えいむ", "aim"), GLOSSARY.find("えいむ"))

    def test_name_terms_take_priority_over_general_glossary(self):
        obj = fake_translator()
        source = "ナイスショット フレームレート ソロキュー ヘッドショット リスポーン地点 "
        source += "クールタイム スキル回し マッチング エイムアシスト 範囲攻撃 リロード 弾薬 スイちゃんが来た"
        self.assertIn("スイちゃん 翻译成 Sui-chan", obj.prompt_for(source, []))

    def test_short_nicknames_are_preserved_not_expanded(self):
        # Spoken nicknames map to proper names, not ordinary vocabulary.
        self.assertIn(("ラプ", "Lap"), GLOSSARY.name_terms("ラプが来た。", context="ラプラス・ダークネス"))
        self.assertIn(("ラプ様", "Lap-sama"), GLOSSARY.name_terms("ラプ様が来た。", context="ラプラス・ダークネス"))
        self.assertIn(("みこ", "Miko"), GLOSSARY.name_terms("みこが来た。", context="さくらみこ"))
        self.assertIn(("みこち", "Mikochi"), GLOSSARY.name_terms("みこちが来た。", context="さくらみこ"))
        self.assertIn(("まつり", "Matsuri"), GLOSSARY.name_terms("まつりが来た。", context="夏色まつり"))
        self.assertIn(("ぼたん", "Botan"), GLOSSARY.name_terms("ぼたんが来た。", context="獅白ぼたん"))
        self.assertIn(("わため", "Watame"), GLOSSARY.name_terms("わためが来た。", context="角巻わため"))

    def test_nicknames_are_not_literal_ordinary_words(self):
        # Without talent context, ambiguous terms resolve to ordinary meaning.
        self.assertEqual(GLOSSARY.name_terms("ボタンを押して。"), [])
        self.assertEqual(GLOSSARY.name_terms("祭りに行こう。"), [])
        self.assertEqual(GLOSSARY.name_terms("みこという言葉。"), [])
        # With talent context, they are proper names.
        self.assertIn(("ボタン", "Botan"), GLOSSARY.name_terms("ボタンが来た。", context="獅白ぼたん"))
        self.assertIn(("まつり", "Matsuri"), GLOSSARY.name_terms("まつりが来た。", context="夏色まつり"))

    def test_terms_prompt_includes_short_nicknames(self):
        obj = fake_translator()
        # Short nickname should appear in the prompt when it matches the source.
        prompt = obj.prompt_for("ラプが来た。", [])
        self.assertIn("ラプ 翻译成 Lap", prompt)

    def test_terms_prioritizes_exact_matches_under_term_cap(self):
        obj = fake_translator()
        # Many terms match; short nickname exact match must not be crowded out.
        source = "ラプ"
        prompt = obj.prompt_for(source, [])
        self.assertIn("ラプ 翻译成 Lap", prompt)



class NameConsistencyTests(unittest.TestCase):
    def test_missing_name_gets_one_deterministic_retry_with_context(self):
        obj = fake_translator("Suixian shot me.", "Sui-chan shot me.")
        history = [("前の会話", "Incorrect prior English.")]
        self.assertEqual(obj.translate("スイちゃんに撃たれた。", history), "Sui-chan shot me.")
        self.assertEqual(obj._chat.call_count, 2)
        retry = obj._chat.call_args_list[1]
        self.assertEqual(retry.args[0], obj._chat.call_args_list[0].args[0])
        self.assertIn("スイちゃん 翻译成 Sui-chan", retry.args[0])
        self.assertIn("前の会話", retry.args[0])
        self.assertNotIn("Incorrect prior English.", retry.args[0])
        self.assertNotIn("Suixian", retry.args[0])
        self.assertEqual(retry.kwargs["temperature"], 0.0)

    def test_failed_name_retry_keeps_only_a_name_omission(self):
        for initial, expected in (("Su-chan shot me.", ""), ("スイちゃんに撃たれた。", ""),
                                  ("She shot me.", "She shot me.")):
            with self.subTest(initial=initial):
                obj = fake_translator(initial, "Suixian shot me.")
                self.assertEqual(obj.translate("すいちゃんに撃たれた。", []), expected)
                self.assertEqual(obj._chat.call_count, 2)

    def test_best_attempt_prefers_valid_then_named_then_earliest(self):
        names = [("スイちゃん", "Sui-chan")]
        self.assertEqual(HymtTranslator._best(["She shot me.", "Sui-chan shot me."], names), "Sui-chan shot me.")
        self.assertEqual(HymtTranslator._best(["She shot me.", "Someone shot me."], names), "She shot me.")
        self.assertEqual(HymtTranslator._best(["撃たれた", "She shot me."], names), "She shot me.")
        self.assertEqual(HymtTranslator._best(["撃たれた", "Suixian shot me."], names), "")

    def test_garbled_names_are_near_misses_not_other_words(self):
        self.assertTrue(garbled_name("Suixian shot me.", ["Sui-chan"]))
        self.assertTrue(garbled_name("Sue-chan shot me.", ["Sui-chan"]))
        self.assertTrue(garbled_name("Sui-chan met Kalliope.", ["Calliope"], ["Sui-chan"]))
        self.assertFalse(garbled_name("She shot me.", ["Sui-chan"]))
        # A correctly rendered name is not a near miss for another missing name.
        self.assertFalse(garbled_name("Sui-chan is here.", ["Suisei"], ["Sui-chan"]))
        # A shared honorific alone does not make another name a near miss.
        self.assertFalse(garbled_name("Mio-san is here.", ["Sui-san"]))

    def test_case_and_hyphen_variants_are_token_safe(self):
        names = [("スイちゃん", "Sui-chan"), ("カリオペ", "Calliope")]
        for english in ("SUI-CHAN and Calliope.", "Sui chan and Calliope's car.", "Sui‑chan and Calliope!"):
            self.assertEqual(missing_names(english, names), [])
        for english in ("Suichan and Kalliope.", "Suixian and Kariope.", "XSui-chan and CalliopeX."):
            self.assertEqual(missing_names(english, names), ["Sui-chan", "Calliope"])

    def test_source_names_not_just_context_are_required(self):
        obj = fake_translator("She is here.")
        obj.stream = StreamContext(title="星街すいせい")
        self.assertEqual(obj.translate("ここにいる", []), "She is here.")
        obj._chat.assert_called_once()

    def test_close_during_bad_completion_prevents_name_retry(self):
        obj = fake_translator()
        def complete(*args, **kwargs):
            obj.interrupt()
            return "Suixian."
        obj._chat.side_effect = complete
        self.assertEqual(obj.translate("スイちゃん", []), "")
        obj._chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
