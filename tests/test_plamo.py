import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.translate.plamo import PlamoTranslator, _clean, _translation_context


class FakeGlossary:
    def find(self, text):
        return [("ぺこら", "Pekora")] if "ぺこら" in text else []


class PlamoTranslatorTests(unittest.TestCase):
    def test_context_without_video_title_keeps_subject_neutrality(self):
        context = _translation_context("")
        self.assertIn("Japanese often omits the subject or possessor", context)
        self.assertIn("instead of guessing I, you, my, your", context)
        self.assertIn("Preserve people and possessives explicitly stated", context)
        self.assertNotIn("Video title:", context)

    def test_clean_removes_prefix_and_quotes(self):
        self.assertEqual(_clean('EN: "Hello!"\n'), "Hello!")

    def test_translation_guard_fixes_stream_example_without_second_request(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(timeout_s=5)

        with patch.object(
            PlamoTranslator,
            "_chat",
            return_value="My driving's just not that great, so I ended up hitting it.",
        ) as chat:
            result = translator.translate(
                "なんか運転がまぁ上手くないから当たっちゃったんだよなぁ",
                [],
            )

        self.assertEqual(
            result,
            "The driving's just not that great, so there was a collision.",
        )
        chat.assert_called_once()

    def test_old_english_history_is_neutralized_before_new_request(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            context_lines=2, plamo_model="plamo-3-translate", max_tokens=120,
        )
        translator.glossary = FakeGlossary()
        translator.vod_title = ""
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"

        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Okay."}}]
        }).encode("utf-8"))
        with patch(
            "auto_translate.translate.plamo.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            translator._chat(
                "次の話",
                [("運転が下手", "My driving is bad.")],
                timeout=5,
            )

        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            payload["messages"][1]["content"],
            "The driving is bad.",
        )

    def test_repeated_short_katakana_term_reuses_translation_without_model_call(self):
        translator = object.__new__(PlamoTranslator)
        translator.memory_lines = 24
        translator.cfg = SimpleNamespace(timeout_s=5)
        with patch.object(PlamoTranslator, "_chat") as chat:
            english = translator.translate(
                "シャッター！",
                [("シャッター", "Shutter!")],
            )
        self.assertEqual(english, "Shutter!")
        chat.assert_not_called()

    def test_memory_context_uses_older_japanese_but_not_old_english_guesses(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            context_lines=2, memory_lines=24,
            plamo_model="plamo-3-translate", max_tokens=120,
        )
        translator.glossary = FakeGlossary()
        translator.vod_title = ""
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Okay."}}]
        }).encode("utf-8"))
        with patch(
            "auto_translate.translate.plamo.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            translator._chat(
                "次は何",
                [("ぺこらとレースをしてる", "My race with Pekora."),
                 ("シャッター", "Shutter!"),
                 ("まだある", "There is more."),
                 ("次のゲーム", "Next game.")],
                timeout=5,
            )
        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        context = payload["chat_template_kwargs"]["context"]
        self.assertIn("ぺこら = Pekora", context)
        self.assertIn("シャッター ~ Shutter", context)
        self.assertIn("ぺこらとレースをしてる", context)
        self.assertNotIn("My race with Pekora.", context)
        self.assertEqual(len(payload["messages"]), 5)  # last two pairs + new Japanese

    def test_request_uses_plamo_options_history_and_glossary(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            context_lines=2,
            plamo_model="plamo-3-translate",
            max_tokens=120,
            timeout_s=5,
        )
        translator.glossary = FakeGlossary()
        translator.vod_title = "【ホロGTA】大空スバル視点"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"

        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Hello, Pekora!"}}]
        }).encode("utf-8"))

        with patch(
            "auto_translate.translate.plamo.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            result = translator.translate(
                "ぺこら、こんにちは",
                [("昨日はありがとう", "Thanks for yesterday.")],
            )

        self.assertEqual(result, "Hello, Pekora!")
        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "plamo-3-translate")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["chat_template_kwargs"]["source_lang"], "ja")
        self.assertEqual(payload["chat_template_kwargs"]["target_lang"], "en")
        self.assertEqual(
            payload["chat_template_kwargs"]["glossary"],
            [{"source": "ぺこら", "target": "Pekora"}],
        )
        context = payload["chat_template_kwargs"]["context"]
        self.assertIn("Video title: 【ホロGTA】大空スバル視点", context)
        self.assertIn("subject-neutral English", context)
        self.assertIn("The driving is bad", context)
        self.assertEqual(len(payload["messages"]), 3)


class SelectiveRecheckTests(unittest.TestCase):
    @staticmethod
    def translator(**settings):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            timeout_s=10.0, quality_recheck_enabled=True,
            quality_recheck_timeout_s=3.0,
            quality_recheck_min_interval_s=0.0, **settings,
        )
        translator.memory_lines = 0
        translator.glossary = FakeGlossary()
        return translator

    def test_clean_translation_uses_only_one_model_call(self):
        translator = self.translator()
        with patch.object(PlamoTranslator, "_chat", return_value="Watame is here.") as chat:
            self.assertEqual(translator.translate("わためが来た", []), "Watame is here.")
        chat.assert_called_once()

    def test_nonverbal_loop_needs_no_model_call(self):
        translator = self.translator()
        with patch.object(PlamoTranslator, "_chat") as chat:
            self.assertEqual(translator.translate("タタタタあっとたっとと", []), "[SKIP]")
        chat.assert_not_called()

    def test_possible_negation_loss_gets_one_bounded_second_pass(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["It's sunny!", "It's not sunny!"],
        ) as chat:
            self.assertEqual(translator.translate("晴れてないわ", []), "It's not sunny!")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(
            chat.call_args.kwargs["review_hint"], "possible_missing_negation"
        )

    def test_nonenglish_gibberish_gets_one_retry(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["Ta-ta-ta-ta!", "The police are here!"],
        ) as chat:
            self.assertEqual(translator.translate("警察が来た", []), "The police are here!")
        self.assertEqual(chat.call_count, 2)

    def test_repeated_bad_second_pass_keeps_initial_output(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["Ta-ta-ta-ta!", "Ta-ta-ta-ta!"],
        ) as chat:
            self.assertEqual(translator.translate("警察が来た", []), "Ta-ta-ta-ta!")
        self.assertEqual(chat.call_count, 2)

    def test_failed_recheck_does_not_fail_first_translation(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["It's sunny!", TimeoutError("slow")],
        ) as chat:
            self.assertEqual(translator.translate("晴れてないわ", []), "It's sunny!")
        self.assertEqual(chat.call_count, 2)

    def test_meaningful_japanese_skip_requires_retry(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["[SKIP]", "Good morning."],
        ) as chat:
            self.assertEqual(translator.translate("おはよう", []), "Good morning.")
        self.assertEqual(chat.call_count, 2)

    def test_false_skip_is_not_silently_suppressed(self):
        translator = self.translator()
        with patch.object(
            PlamoTranslator, "_chat", side_effect=["[SKIP]", "[SKIP]"],
        ):
            self.assertEqual(translator.translate("おはよう", []), "[translation failed]")

    def test_can_disable_second_pass(self):
        translator = self.translator()
        translator.cfg.quality_recheck_enabled = False
        with patch.object(PlamoTranslator, "_chat", return_value="It's sunny!") as chat:
            self.assertEqual(translator.translate("晴れてないわ", []), "It's sunny!")
        chat.assert_called_once()

    def test_recheck_respects_minimum_interval(self):
        translator = self.translator()
        translator.cfg.quality_recheck_min_interval_s = 3600
        with patch.object(PlamoTranslator, "_chat", side_effect=[
            "It's sunny!", "It's not sunny!", "It's sunny!"
        ]) as chat:
            self.assertEqual(translator.translate("晴れてないわ", []), "It's not sunny!")
            self.assertEqual(translator.translate("晴れてないわ", []), "It's sunny!")
        self.assertEqual(chat.call_count, 3)

    def test_review_context_keeps_title_glossary_and_japanese_history(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            context_lines=2, memory_lines=0, plamo_model="plamo-3-translate",
            max_tokens=120,
        )
        translator.glossary = FakeGlossary()
        translator.vod_title = "Hololive livestream"
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Pekora is here."}}]
        }).encode("utf-8"))
        with patch(
            "auto_translate.translate.plamo.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            translator._chat(
                "ぺこらが来た", [("昨日", "Yesterday")],
                timeout=3.0, review_hint="untranslated_japanese",
            )
        body = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            body["chat_template_kwargs"]["glossary"],
            [{"source": "ぺこら", "target": "Pekora"}],
        )
        self.assertIn("Video title: Hololive livestream", body["chat_template_kwargs"]["context"])
        self.assertIn("untranslated_japanese", body["chat_template_kwargs"]["context"])
        self.assertEqual(len(body["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
