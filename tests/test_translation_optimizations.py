"""Offline tests for HY-MT prompt construction and context-gated glossary hints."""
from pathlib import Path
import unittest

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.hymt import build_context, build_prompt, clean_output, missing_names
from test_english_reliability import fake_translator

GLOSSARY = Glossary.load(Path("glossary.toml"))


class PromptTemplateTests(unittest.TestCase):
    def test_plain_template_without_terms_or_context(self):
        self.assertEqual(build_prompt("こんにちは", [], ""),
                         "Translate the following segment into English, without additional explanation."
                         "\n\nこんにちは")

    def test_terminology_template(self):
        prompt = build_prompt("ぺこらだよ", [("ぺこら", "Pekora")], "")
        self.assertEqual(prompt, "参考下面的翻译：\nぺこら 翻译成 Pekora\n\n"
                                 "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\nぺこらだよ")

    def test_contextual_template_combined_with_terms(self):
        prompt = build_prompt("草", [("草", "lol")], "前の文\nEarlier line.")
        self.assertTrue(prompt.startswith("参考下面的翻译：\n草 翻译成 lol\n\n前の文\nEarlier line.\n"))
        self.assertTrue(prompt.endswith("参考上面的信息，把下面的文本翻译成英语，注意不需要翻译上文，也不要额外解释：\n草"))

    def test_context_keeps_recent_japanese_only_within_budget(self):
        history = [(f"文{i}", f"Line {i}.") for i in range(6)]
        self.assertEqual(build_context("Title", history, 3, 300), "Title\n文3\n文4\n文5")
        self.assertEqual(build_context("", history, 3, 4), "文5")
        self.assertEqual(build_context("Title", history, 0, 300), "Title")

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
        obj.title = "【宝鐘マリン】雑談"
        obj.cfg = TranslateConfig(context_lines=2)
        prompt = obj.prompt_for("船長が来た", [("おはよう", "Good morning."), ("元気？", "How are you?")])
        self.assertIn("船長 翻译成 Senchou", prompt)
        self.assertIn("【宝鐘マリン】雑談\nおはよう\n元気？", prompt)
        self.assertNotIn("Good morning.", prompt)
        self.assertNotIn("How are you?", prompt)
        self.assertTrue(prompt.endswith("船長が来た"))

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
        obj.title = "獅白ぼたん"
        self.assertEqual(obj.translate("ボタンを押して。", []), "Press the button.")
        obj._chat.assert_called_once()

    def test_generic_vocabulary_does_not_use_kana_folding(self):
        self.assertNotIn(("えいむ", "aim"), GLOSSARY.find("えいむ"))

    def test_name_terms_take_priority_over_general_glossary(self):
        obj = fake_translator()
        source = "ナイスショット フレームレート ソロキュー ヘッドショット リスポーン地点 "
        source += "クールタイム スキル回し マッチング エイムアシスト 範囲攻撃 リロード 弾薬 スイちゃんが来た"
        self.assertIn("スイちゃん 翻译成 Sui-chan", obj.prompt_for(source, []))


class NameConsistencyTests(unittest.TestCase):
    def test_missing_name_gets_one_terminology_only_retry(self):
        obj = fake_translator("Suixian shot me.", "Sui-chan shot me.")
        history = [("前の会話", "Incorrect prior English.")]
        self.assertEqual(obj.translate("スイちゃんに撃たれた。", history), "Sui-chan shot me.")
        self.assertEqual(obj._chat.call_count, 2)
        retry = obj._chat.call_args_list[1]
        self.assertIn("スイちゃん 翻译成 Sui-chan", retry.args[0])
        self.assertNotIn("前の会話", retry.args[0])
        self.assertNotIn("Suixian", retry.args[0])
        self.assertEqual(retry.kwargs["temperature"], 0.0)

    def test_failed_name_retry_is_suppressed_without_rewriting(self):
        for initial in ("Su-chan shot me.", "スイちゃんに撃たれた。", "She shot me."):
            with self.subTest(initial=initial):
                obj = fake_translator(initial, "Suixian shot me.")
                self.assertEqual(obj.translate("すいちゃんに撃たれた。", []), "")
                self.assertEqual(obj._chat.call_count, 2)

    def test_case_and_hyphen_variants_are_token_safe(self):
        names = [("スイちゃん", "Sui-chan"), ("カリオペ", "Calliope")]
        for english in ("SUI-CHAN and Calliope.", "Sui chan and Calliope's car.", "Sui‑chan and Calliope!"):
            self.assertEqual(missing_names(english, names), [])
        for english in ("Suichan and Kalliope.", "Suixian and Kariope.", "XSui-chan and CalliopeX."):
            self.assertEqual(missing_names(english, names), ["Sui-chan", "Calliope"])

    def test_source_names_not_just_context_are_required(self):
        obj = fake_translator("She is here.")
        obj.title = "星街すいせい"
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
