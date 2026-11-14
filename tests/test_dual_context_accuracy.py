"""Offline accuracy-first context and conversation tests."""
import json
import unittest
from unittest.mock import MagicMock, patch

from auto_translate.config import TranslateConfig
from auto_translate.context_memory import split_completed_japanese_sentences
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import PlamoTranslator, _translation_context


class DualContextAccuracyTests(unittest.TestCase):
    def make_translator(self):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.glossary = Glossary({"わため": "Watame"})
        obj.vod_title = "【ホロGTA】警察無線"
        obj.memory_lines = 0
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        return obj

    def test_accuracy_defaults_keep_original_models(self):
        cfg = TranslateConfig()
        self.assertEqual((cfg.context_profile, cfg.context_lines, cfg.memory_lines),
                         ("compact", 3, 0))
        self.assertTrue(cfg.sentence_context)
        self.assertIn("radio", _translation_context("ホロGTA", profile="compact"))

    def test_only_trusted_bilingual_examples_reach_plamo(self):
        obj = self.make_translator()
        history = [
            ("犯人はそっちに行った！", "Suspect headed your way!"),
            ("もう出た", "He left."),   # invented gender
            ("待って！", "Don't wait!"),  # reversed imperative; should not be used as an example
            ("了解！", "Roger!"),
        ]
        answer = MagicMock()
        answer.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "I'll circle around."}}]}
        ).encode()
        with patch("auto_translate.translate.plamo.urllib.request.urlopen", return_value=answer) as request:
            PlamoTranslator._chat(obj, "回り込む！", history, timeout=1.0)
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "回り込む！"})
        self.assertIn({"role": "assistant", "content": "Roger!"}, body["messages"])
        self.assertNotIn("He left.", str(body["messages"]))
        self.assertNotIn("Don't wait!", str(body["messages"]))
        self.assertIn("もう出た", body["chat_template_kwargs"]["context"])

    def test_glossary_name_mismatch_excluded_from_examples(self):
        obj = self.make_translator()
        history = [("わためが来た", "Watanabe arrived.")]
        answer = MagicMock()
        answer.__enter__.return_value.read.return_value = b'{"choices":[{"message":{"content":"Watame arrived."}}]}'
        with patch("auto_translate.translate.plamo.urllib.request.urlopen", return_value=answer) as request:
            PlamoTranslator._chat(obj, "わためが来た", history, timeout=1)
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["messages"], [{"role": "user", "content": "わためが来た"}])

    def test_punctuation_is_required_for_sentence_split(self):
        self.assertEqual(split_completed_japanese_sentences("犯人は北に向かってる今すぐ追って"), ["犯人は北に向かってる今すぐ追って"])
        self.assertEqual(split_completed_japanese_sentences("了解！すぐ行きます。"), ["了解！すぐ行きます。"])
        self.assertEqual(split_completed_japanese_sentences("「犯人が北だ！」と聞いた。すぐ追いかける！"),
                         ["「犯人が北だ！」と聞いた。", "すぐ追いかける！"])

    def test_sentence_translation_uses_preceding_verified_sentence(self):
        obj = self.make_translator()
        calls = []
        def fake_single(ja, history):
            calls.append((ja, list(history)))
            return "Suspect is heading north!" if len(calls) == 1 else "Move to intercept!"
        obj._translate_one = fake_single
        result = obj.translate("犯人は北に向かってる！すぐに回り込んで！", [])
        self.assertEqual(result, "Suspect is heading north! Move to intercept!")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1][-1], ("犯人は北に向かってる！", "Suspect is heading north!"))

    def test_legacy_can_restore_previous_japanese_only_context(self):
        obj = self.make_translator()
        obj.cfg.context_profile = "legacy"
        answer = MagicMock()
        answer.__enter__.return_value.read.return_value = b'{"choices":[{"message":{"content":"Roger."}}]}'
        with patch("auto_translate.translate.plamo.urllib.request.urlopen", return_value=answer) as request:
            PlamoTranslator._chat(obj, "了解", [("どこだ", "Where?")], timeout=1)
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["messages"], [{"role": "user", "content": "了解"}])


if __name__ == "__main__":
    unittest.main()
