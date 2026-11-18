"""Deterministic output safeguards; these do not measure model accuracy."""
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.subtitle_clean import (
    clean_english_output, english_output_issue, sanitize_english_output,
    unmatched_double_quote_cleanup,
)
from auto_translate.translate.hymt import HymtTranslator, clean_output


def fake_translator(*outputs, glossary=None):
    obj = object.__new__(HymtTranslator)
    obj.cfg = TranslateConfig()
    obj.glossary = glossary or Glossary.load(Path("glossary.toml"))
    obj.stream = StreamContext()
    obj._position = obj._anchor = obj._newest = None
    obj.gpu_layers = obj.layer_count = 32
    obj._lock = threading.Lock()
    obj._closed = False
    obj._relief = threading.Event()
    obj.server = Mock(alive=Mock(return_value=True))
    obj._chat = Mock(side_effect=outputs)
    return obj


class EnglishReliabilityTests(unittest.TestCase):
    def test_remove_only_orphan_double_quotes(self):
        cases = [
            ('He said, "Let\'s go.', "He said, Let's go."),
            ('"Where is Botan?', "Where is Botan?"),
            ('I want to "help Matsuri.', "I want to help Matsuri."),
            ('"Don\'t worry!"', '"Don\'t worry!"'),
            ('He said “go” and then “wait.', 'He said “go” and then wait.'),
            ('Botan’s coat isn’t mine.', 'Botan’s coat isn’t mine.'),
            ('”Where is Botan?', 'Where is Botan?'),
            ('‟Where is Botan?', 'Where is Botan?'),
            ('„Don\'t worry!“', '„Don\'t worry!“'),
            ('‟Don\'t worry!”', '‟Don\'t worry!”'),
        ]
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(unmatched_double_quote_cleanup(original), expected)
                self.assertEqual(clean_output(original), expected)

    def test_adjacent_cues_cannot_borrow_quotes(self):
        self.assertEqual(sanitize_english_output('He said "go'), 'He said go')
        self.assertEqual(sanitize_english_output('right now."'), 'right now.')

    def test_japanese_and_mixed_outputs_are_hard_failures(self):
        for text in ("ボタンを押して", "Matsuriはもうここにいる", "Press the ボタン.", "ﾎﾞﾀﾝ"):
            with self.subTest(text=text):
                self.assertEqual(english_output_issue(text), "untranslated_japanese")
                self.assertEqual(clean_english_output(text), "")

    def test_romanized_names_honorifics_and_terms_are_valid(self):
        for text in ("Botan-chan arrived.", "Lap-sama, kusa!", "Mikochi's senpai is here.", "50!"):
            self.assertIsNone(english_output_issue(text))
            self.assertEqual(sanitize_english_output(text), text)

    def test_invalid_initial_response_is_retried_with_context_then_plainly(self):
        history = [("前の文", "Earlier line.")]
        obj = fake_translator("今日は晴れです", "It's sunny today.")
        self.assertEqual(obj.translate("今日は晴れです", history), "It's sunny today.")
        self.assertEqual(obj._chat.call_count, 2)
        retry_prompt = obj._chat.call_args_list[1].args[0]
        self.assertIn("前の文", retry_prompt)
        self.assertEqual(obj._chat.call_args_list[1].kwargs["temperature"], 0.0)

        obj = fake_translator("今日は晴れです", "今日は晴れです", "It's sunny today.")
        self.assertEqual(obj.translate("今日は晴れです", history), "It's sunny today.")
        self.assertEqual(obj._chat.call_count, 3)
        plain = obj._chat.call_args_list[2]
        self.assertTrue(plain.args[0].startswith("请将以下文本翻译为 英语"))
        self.assertIn("【简洁、自然的口语，保留原文的脏话，不要审查】", plain.args[0])
        self.assertNotIn("前の文", plain.args[0])
        self.assertEqual(plain.kwargs["temperature"], 0.0)

    def test_all_invalid_candidates_suppress_and_continue(self):
        obj = fake_translator("今日は晴れです", "It is 晴れです", "Hello!")
        self.assertEqual(obj.translate("今日は晴れです", []), "")
        self.assertEqual(obj.translate("こんにちは", []), "Hello!")
        self.assertEqual(obj._chat.call_count, 3)

    def test_valid_output_is_not_retried(self):
        obj = fake_translator("EN: Matsuri is going.\n")
        self.assertEqual(obj.translate("まつりが行く", []), "Matsuri is going.")
        obj._chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
