import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.translate.plamo import PlamoTranslator, _clean


class FakeGlossary:
    def find(self, text):
        return [("ぺこら", "Pekora")] if "ぺこら" in text else []


class PlamoTranslatorTests(unittest.TestCase):
    def test_clean_removes_prefix_and_quotes(self):
        self.assertEqual(_clean('EN: "Hello!"\n'), "Hello!")

    def test_request_uses_plamo_options_history_and_glossary(self):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = SimpleNamespace(
            context_lines=2,
            plamo_model="plamo-3-translate",
            max_tokens=120,
            timeout_s=5,
        )
        translator.glossary = FakeGlossary()
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
        self.assertEqual(len(payload["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
