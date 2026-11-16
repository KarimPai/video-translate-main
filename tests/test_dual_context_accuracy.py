"""Offline accuracy-first context and conversation tests."""
import json
from pathlib import Path
from auto_translate.text_filter import rejection_reason
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
                         ("compact", 3, 16))
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
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "Done."}}]}
        with patch.object(obj.client, "post", wraps=obj.client.post) as request:
            PlamoTranslator._chat(obj, "回り込む！", history, timeout=1.0)
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "回り込む！"})
        self.assertIn({"role": "assistant", "content": "Roger!"}, body["messages"])
        self.assertNotIn("He left.", str(body["messages"]))
        self.assertNotIn("Don't wait!", str(body["messages"]))
        self.assertIn("もう出た", body["chat_template_kwargs"]["context"])

    def test_kusa_sense_comes_from_dialogue(self):
        obj = self.make_translator()
        obj.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
        obj.client = MagicMock()
        obj.memory_lines = 16
        for current, prior in [
            ("草を刈って", "庭の手入れをしよう"),
            ("草。", "羊は何を食べるの？"),
            ("クサ", "羊は何を食べるの？"),
            ("それは草", "そのオチ面白すぎる"),
        ]:
            with self.subTest(current=current):
                history = [(current, "LOL"), (prior, "")]
                self.assertIsNone(rejection_reason(current))
                obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
                obj.translate(current, history)
                body = obj.client.post.call_args.kwargs["json"]
                options = body["chat_template_kwargs"]
                self.assertIn(prior, str(body))
                self.assertIn("grass or weeds", options["context"])
                self.assertIn("amusement", options["context"])
                self.assertNotIn({"source": "草", "target": "LOL"}, options.get("glossary", []))
                obj.client.post.reset_mock()
        for current in ("面倒くさい", "草原を走る"):
            obj._chat(current, [], timeout=1)
            options = obj.client.post.call_args.kwargs["json"]["chat_template_kwargs"]
            self.assertNotIn("Resolve kusa", options["context"])
        self.assertEqual(rejection_reason("草草草"), "laughter_only")

    def test_glossary_name_mismatch_excluded_from_examples(self):
        obj = self.make_translator()
        history = [("わためが来た", "Watanabe arrived.")]
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "Done."}}]}
        with patch.object(obj.client, "post", wraps=obj.client.post) as request:
            PlamoTranslator._chat(obj, "わためが来た", history, timeout=1)
        body = request.call_args.kwargs["json"]
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
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "Done."}}]}
        with patch.object(obj.client, "post", wraps=obj.client.post) as request:
            PlamoTranslator._chat(obj, "了解", [("どこだ", "Where?")], timeout=1)
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["messages"], [{"role": "user", "content": "了解"}])


if __name__ == "__main__":
    unittest.main()
