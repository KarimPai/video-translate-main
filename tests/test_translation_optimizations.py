"""Offline tests for HY-MT prompt construction and context-gated glossary hints."""
from pathlib import Path
import unittest

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.hymt import build_context, build_prompt, clean_output
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

    def test_context_keeps_most_recent_pairs_in_order_within_budget(self):
        history = [(f"文{i}", f"Line {i}.") for i in range(6)]
        self.assertEqual(build_context("Title", history, 3, 600),
                         "Title\n文3\nLine 3.\n文4\nLine 4.\n文5\nLine 5.")
        self.assertEqual(build_context("", history, 3, 20), "文5\nLine 5.")
        self.assertEqual(build_context("Title", history, 0, 600), "Title")

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
        self.assertIn("【宝鐘マリン】雑談\nおはよう\nGood morning.\n元気？\nHow are you?", prompt)
        self.assertTrue(prompt.endswith("船長が来た"))


if __name__ == "__main__":
    unittest.main()
