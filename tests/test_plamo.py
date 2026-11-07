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


if __name__ == "__main__":
    unittest.main()
