"""Qwen request construction, safeguards and clean-slate tests (no GPU required)."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.translate.qwen import (
    QwenTranslator, _clean, _model_ids, _remove_unmentioned_full_names,
)


class FakeGlossary:
    def find(self, jp):
        return [("ぺこちゃん", "Peko-chan")] if "ぺこちゃん" in jp else []


def translator(**kwargs):
    settings = {
        "qwen_model": "qwen3.5-4b", "context_lines": 3,
        "timeout_s": 10.0, "max_tokens": 128,
        "quality_recheck_enabled": True, "quality_recheck_timeout_s": 3.0,
        "quality_recheck_min_interval_s": 0.0,
    }
    settings.update(kwargs)
    instance = object.__new__(QwenTranslator)
    instance.cfg = SimpleNamespace(**settings)
    instance.url = "http://127.0.0.1:8001/v1/chat/completions"
    instance.glossary = FakeGlossary()
    instance.vod_title = "ARK stream with Pekora"
    instance.memory_lines = 0
    return instance


class QwenTranslatorTests(unittest.TestCase):
    def test_clean_discards_leaked_thinking(self):
        self.assertEqual(_clean("<think>speculation</think>\nEN: Peko-chan!"), "Peko-chan!")
        self.assertEqual(_clean("EN: \"Hello!\""), "Hello!")
        self.assertEqual(_clean(""), "")

    def test_current_glossary_only_and_no_past_english_leakage(self):
        instance = translator()
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Peko-chan, hello!"}}]
        }).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response) as request:
            en = instance.translate(
                "ぺこちゃん、こんにちは",
                [("猫ちゃんはどこ", "A hallucinated Usada Pekora is here.")],
            )
        self.assertEqual(en, "Peko-chan, hello!")
        data = json.loads(request.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(data["model"], "qwen3.5-4b")
        self.assertEqual(data["messages"][0]["role"], "system")
        self.assertIn("Japanese", data["messages"][0]["content"])
        self.assertEqual(data["messages"][1]["role"], "user")
        self.assertIn("ぺこちゃん => Peko-chan", data["messages"][1]["content"])
        self.assertIn("猫ちゃんはどこ", data["messages"][1]["content"])
        self.assertNotIn("hallucinated", data["messages"][1]["content"])
        self.assertEqual(data["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("source_lang", data["chat_template_kwargs"])
        self.assertNotIn("stop", data)

    def test_hololive_name_expansion_guard(self):
        self.assertEqual(
            _remove_unmentioned_full_names("他の人", "Ichijou Ririka"), ""
        )
        self.assertEqual(
            _remove_unmentioned_full_names("一条莉々華が来た", "Ichijou Ririka is here"),
            "Ichijou Ririka is here",
        )

    def test_isolated_exclamation_needs_no_model_call(self):
        instance = translator()
        with patch.object(QwenTranslator, "_chat") as chat:
            self.assertEqual(instance.translate("すごい", []), "Wow!")
        chat.assert_not_called()

    def test_selective_recheck_is_bounded(self):
        instance = translator()
        with patch.object(
            QwenTranslator, "_chat", side_effect=["It's sunny.", "It's not sunny."],
        ) as chat:
            self.assertEqual(instance.translate("晴れてないわ", []), "It's not sunny.")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args.kwargs["review_hint"], "possible_missing_negation")
        self.assertEqual(chat.call_args.kwargs["timeout"], 3.0)

    def test_bad_skip_is_not_silently_output(self):
        instance = translator(quality_recheck_enabled=False)
        with patch.object(QwenTranslator, "_chat", return_value="[SKIP]"):
            self.assertEqual(instance.translate("警察が来た", []), "[translation failed]")

    def test_qwen_server_models_are_checked_before_reuse(self):
        response = io.BytesIO(json.dumps({"data": [{"id": "qwen3.5-4b"}]}).encode())
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response):
            self.assertEqual(_model_ids("http://127.0.0.1:8001/v1"), ["qwen3.5-4b"])

    def test_ark_glossary_in_qwen_prompt(self):
        instance = translator()
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "Giganotosaurus!"}},
        ]}).encode())
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response) as call:
            self.assertEqual(instance._chat("ギガノト!", [], 5), "Giganotosaurus!")
        data = json.loads(call.call_args.args[0].data.decode())
        self.assertIn("ギガノト => Giganotosaurus", data["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
