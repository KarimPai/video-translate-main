"""Deterministic output safeguards; these do not measure model accuracy."""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.subtitle_clean import (
    clean_english_output, english_output_issue, sanitize_english_output,
    unmatched_double_quote_cleanup,
)
from auto_translate.translate.plamo import PlamoTranslator, _clean, _trusted_history_pair
from auto_translate.translation_guard import suspicious_translation


class EnglishReliabilityTests(unittest.TestCase):
    def translator(self, *outputs):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.cfg.sentence_context = False
        obj.glossary = Glossary.load(Path("glossary.toml"))
        obj.memory_lines = 0
        obj.vod_title = ""
        obj._chat = Mock(side_effect=outputs)
        return obj

    def http_translator(self, *choices):
        obj = self.translator()
        del obj._chat
        obj.client = MagicMock()
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.client.post.return_value.json.side_effect = [
            {"choices": [choice], "usage": {"prompt_tokens": 128}} for choice in choices]
        return obj

    @staticmethod
    def choice(content, finish_reason="stop"):
        return {"message": {"content": content}, "finish_reason": finish_reason}

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
                self.assertEqual(_clean(original), expected)

    def test_adjacent_cues_cannot_borrow_quotes(self):
        self.assertEqual(sanitize_english_output('He said "go'), 'He said go')
        self.assertEqual(sanitize_english_output('right now."'), 'right now.')

    def test_japanese_and_mixed_outputs_are_hard_failures(self):
        for text in ("ボタンを押して", "Matsuriはもうここにいる", "Press the ボタン.", "ﾎﾞﾀﾝ"):
            with self.subTest(text=text):
                self.assertEqual(english_output_issue(text), "untranslated_japanese")
                self.assertEqual(suspicious_translation("ボタンを押して", text), "untranslated_japanese")
                self.assertEqual(clean_english_output(text), "")

    def test_romanized_names_honorifics_and_terms_are_valid(self):
        for text in ("Botan-chan arrived.", "Lap-sama, kusa!", "Mikochi's senpai is here.", "50!"):
            self.assertIsNone(english_output_issue(text))
            self.assertEqual(sanitize_english_output(text), text)

    def test_invalid_initial_response_can_be_retried_once(self):
        obj = self.translator("今日は晴れです", "It's sunny today.")
        self.assertEqual(obj.translate("今日は晴れです", []), "It's sunny today.")
        self.assertEqual(obj._chat.call_count, 2)

    def test_failed_retry_cannot_replace_valid_original(self):
        # A literal-loss warning is advisory: retain the valid English original
        # if its context-reduced retry merely copies the Japanese.
        obj = self.translator("Matsuri is going.", "まつりがお祭りに行く")
        self.assertEqual(obj.translate("まつりがお祭りに行く", [("夏色まつりだよ", "")]),
                         "Matsuri is going.")
        self.assertFalse(obj.last_name_diagnostics["retry_accepted"])

    def test_all_invalid_candidates_suppress_and_continue(self):
        obj = self.translator("今日は晴れです", "It is 晴れです", "Hello!")
        self.assertEqual(obj.translate("今日は晴れです", []), "")
        self.assertEqual(obj.last_name_diagnostics["invalid_english"], "untranslated_japanese")
        self.assertEqual(obj.translate("こんにちは", []), "Hello!")
        self.assertEqual(obj._chat.call_count, 3)

    def test_disabled_retry_still_suppresses_japanese(self):
        obj = self.translator("今日は晴れです")
        obj.cfg.retry_suspicious = False
        self.assertEqual(obj.translate("今日は晴れです", []), "")
        obj._chat.assert_called_once()

    def test_language_failure_cannot_enter_trusted_history(self):
        glossary = Glossary({})
        self.assertFalse(_trusted_history_pair("こんにちは", "こんにちは", "", glossary))
        self.assertFalse(_trusted_history_pair("こんにちは", "Hello こんにちは", "", glossary))

    def test_quotes_are_clean_before_sentence_history(self):
        obj = self.translator('He said "go now.', 'Already there.')
        obj.cfg.sentence_context = True
        observed = []
        outputs = iter(['He said "go now.', 'Already there.'])
        def capture(_ja, history, **_kwargs):
            observed.append(list(history))
            return next(outputs)
        obj._chat.side_effect = capture
        self.assertEqual(obj.translate("彼は今すぐ行くと言った。もうそこにいるんです。", []),
                         'He said go now. Already there.')
        self.assertEqual(observed[1][-1][1], "He said go now.")

    def test_invalid_sentence_suppresses_entire_cue_without_extra_whole_pass(self):
        obj = self.translator("Already there.", "もうそこにいるんです", "もうそこにいるんです")
        obj.cfg.sentence_context = True
        self.assertEqual(obj.translate("今すぐ行ってください。もうそこにいるんです。", []), "")
        self.assertEqual(obj._chat.call_count, 3)

    def test_confirmed_truncation_retries_larger_budget_once(self):
        obj = self.http_translator(self.choice("It is a", "length"), self.choice("It is a nice day."))
        self.assertEqual(obj.translate("今日はいい天気です", []), "It is a nice day.")
        bodies = [call.kwargs["json"] for call in obj.client.post.call_args_list]
        self.assertEqual([body["max_tokens"] for body in bodies], [192, 384])
        self.assertEqual(bodies[0]["messages"], bodies[1]["messages"])

    def test_repeated_truncation_is_suppressed_without_another_semantic_retry(self):
        obj = self.http_translator(self.choice("It is a", "length"), self.choice("It is a very", "length"))
        self.assertEqual(obj.translate("今日はいい天気です", []), "")
        self.assertEqual(obj.client.post.call_count, 2)
        self.assertEqual(obj.last_name_diagnostics["invalid_english"], "output_truncated")

    def test_larger_completion_requires_measured_free_context_capacity(self):
        for usage in ({}, {"prompt_tokens": 3900}):
            with self.subTest(usage=usage):
                obj = self.http_translator(self.choice("It is a", "length"))
                obj.client.post.return_value.json.side_effect = [
                    {"choices": [self.choice("It is a", "length")], "usage": usage}]
                self.assertEqual(obj.translate("今日はいい天気です", []), "")
                obj.client.post.assert_called_once()

    def test_unfinished_speech_is_not_token_truncation(self):
        obj = self.http_translator(self.choice("And then"))
        self.assertEqual(obj.translate("それで", []), "And then")
        obj.client.post.assert_called_once()

    def test_overlay_history_is_bounded_trusted_and_glossary_is_relevant(self):
        obj = self.http_translator(self.choice("Botan-chan arrived."))
        obj.cfg.context_profile = "overlay"
        obj.cfg.context_lines = 10
        obj.memory_lines = 16
        obj.vod_title = "Unrelated title"
        history = [("了解", "Roger!"), ("ありがとう", "Thanks!"), ("はい", "Yes!"),
                   ("こんにちは", "こんにちは")]
        obj._chat("ボタンちゃんが来た", history, timeout=1)
        body = obj.client.post.call_args.kwargs["json"]
        self.assertLessEqual(sum(message["role"] == "assistant" for message in body["messages"]), 3)
        self.assertNotIn("Unrelated title", body["chat_template_kwargs"]["context"])
        self.assertNotIn({"role": "assistant", "content": "こんにちは"}, body["messages"])
        self.assertEqual(body["chat_template_kwargs"]["source_lang"], "ja")
        self.assertEqual(body["chat_template_kwargs"]["target_lang"], "en")

    def test_default_profile_remains_unchanged(self):
        self.assertEqual(TranslateConfig().context_profile, "compact")
        self.assertEqual(TranslateConfig(context_profile="hybrid").context_profile, "hybrid")
        with self.assertRaises(ValueError):
            TranslateConfig(context_profile="unknown")

    def test_alternative_profiles_select_three_trusted_pairs_before_recent_bad_examples(self):
        trusted = [("了解", "Roger!"), ("ありがとう", "Thanks!"), ("はい", "Yes!")]
        history = trusted + [("こんにちは", "こんにちは"), ("もう出た", "He left.")]
        for profile in ("overlay", "hybrid"):
            with self.subTest(profile=profile):
                obj = self.http_translator(self.choice("Done."))
                obj.cfg.context_profile = profile
                # Exercise the complete cue path as well as _chat's selection:
                # a short context_lines must not pre-trim candidate history.
                self.assertEqual(obj.translate("終わった", history), "Done.")
                body = obj.client.post.call_args.kwargs["json"]
                actual = [message["content"] for message in body["messages"]
                          if message["role"] == "assistant"]
                self.assertEqual(actual, [translation for _, translation in trusted])


if __name__ == "__main__":
    unittest.main()
