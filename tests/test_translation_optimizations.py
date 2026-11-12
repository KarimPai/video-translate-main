"""Offline tests for terminology repair, glossary, and selective PLaMo retries."""
import json
import unittest
from unittest.mock import patch, MagicMock

from auto_translate.asr_correction import correct_known_asr_terms
from auto_translate.context_memory import build_recent_context
from auto_translate.glossary import Glossary
from auto_translate.translation_guard import suspicious_translation
from auto_translate.translate.plamo import PlamoTranslator, _current_glossary_matches
from auto_translate.config import TranslateConfig


class QualityOptimizationsTests(unittest.TestCase):
    def test_context_verified_hologta_recognition(self):
        title = "【ホロGTA】1日目まとめ - さくらみこ"
        self.assertEqual(
            correct_known_asr_terms("コロナイブgta", video_context=title),
            "ホロライブGTA",
        )
        self.assertEqual(
            correct_known_asr_terms("50人弱のフォロメンが。", video_context=title),
            "50人弱のホロメンが。",
        )
        self.assertEqual(
            correct_known_asr_terms("コロナイブgta", video_context="unrelated gaming stream"),
            "コロナイブgta",
        )
        self.assertEqual(
            correct_known_asr_terms("コロナイブgta", video_context=title,
                                    caption="コロナイブgta"),
            "コロナイブgta",
        )
        self.assertEqual(
            correct_known_asr_terms("いらないよ！", video_context=title),
            "いらないよ！",
        )

    def test_scene_glossary_only_on_matching_video_and_utterance(self):
        glossary = Glossary({"ホロライブ": "Hololive", "わため": "Watame"})
        title = "ホロGTA 1日目"
        items = _current_glossary_matches("副署長とホロライブGTA", title, glossary)
        self.assertIn(("副署長", "Deputy Police Chief"), items)
        self.assertIn(("ホロライブGTA", "Hololive GTA"), items)
        self.assertNotIn(("ホロライブ", "Hololive"), items)  # masked by longer term
        self.assertNotIn(("警察", "police"), items)  # absent from current utterance
        self.assertEqual(
            _current_glossary_matches("副署長", "unrelated video", glossary),
            [],
        )

    def test_memory_never_repeats_untrusted_english_hints(self):
        glossary = Glossary({"わため": "Watame"})
        history = [
            ("わため", "Watanabe"),
            ("シャッター", "shudder"),
            ("やった", "I caused an accident"),
            ("そうなんだ", "Yes"),
            ("なるほど", "Right"),
            ("ありがとう", "Thanks"),
        ]
        context = build_recent_context(history, glossary, context_lines=4, memory_lines=8)
        self.assertIn("わため = Watame", context)
        self.assertNotIn("Watanabe", context)
        self.assertNotIn("shudder", context)
        self.assertNotIn("I caused", context)
        self.assertIn("Earlier Japanese", context)

    def test_guard_short_reactions_negation_and_numbers(self):
        self.assertEqual(
            suspicious_translation("すごい!", "That's bad!"), "positive_reaction_reversed")
        self.assertIsNone(suspicious_translation("すごい!", "That's amazing!"))
        self.assertIsNone(suspicious_translation("すごい!", "Not bad!"))
        self.assertEqual(
            suspicious_translation("できないよ", "I can do it."), "possible_negation_loss")
        self.assertIsNone(suspicious_translation("できないよ", "I can't do it."))
        self.assertEqual(
            suspicious_translation("50人", "There are 40 people."), "numeric_mismatch")
        self.assertIsNone(suspicious_translation("50人", "There are fifty people."))
        self.assertEqual(
            suspicious_translation(
                "ホロライブ", "Corona Eve",
                names=[("ホロライブ", "Hololive")],
            ),
            "proper_name_mismatch",
        )
        self.assertIsNone(suspicious_translation(
            "ホロライブ", "Hololive", names=[("ホロライブ", "Hololive")]))
        self.assertEqual(
            suspicious_translation("テスト", "Earlier Japanese for scene continuity: ..."),
            "prompt_echo_or_loop")

    @staticmethod
    def translator(responses, title="ホロGTA 1日目"):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.glossary = Glossary({"ホロライブ": "Hololive"})
        obj.vod_title = title
        obj.memory_lines = 0
        calls = []
        responses = iter(responses)

        def fake_chat(ja, history, timeout, *, neutral_retry=False):
            calls.append((ja, list(history), timeout))
            return next(responses)

        obj._chat = fake_chat
        return obj, calls

    def test_suspicious_translation_retry_without_context(self):
        translator, calls = self.translator(["That's bad!", "That's amazing!"])
        english = translator.translate("すごい!", [("前", "The previous wrong pronoun")])
        self.assertEqual(english, "That's amazing!")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], [])
        self.assertEqual(calls[1][2], 15.0)

    def test_good_translation_uses_single_pass(self):
        translator, calls = self.translator(["That's amazing!"])
        self.assertEqual(translator.translate("すごい!", []), "That's amazing!")
        self.assertEqual(len(calls), 1)

    def test_second_suspicious_translation_never_overwrites_first(self):
        translator, calls = self.translator(["I can do it.", "I will do it."])
        self.assertEqual(translator.translate("できない", []), "I can do it.")
        self.assertEqual(len(calls), 2)

    def test_retry_can_be_disabled(self):
        translator, calls = self.translator(["That's bad!"])
        translator.cfg.retry_suspicious = False
        self.assertEqual(translator.translate("すごい!", []), "That's bad!")
        self.assertEqual(len(calls), 1)

    def test_chat_passes_only_matched_scene_terms(self):
        translator, calls = self.translator([])
        translator.glossary = Glossary({"ホロライブ": "Hololive"})
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        history = [("昔", "I crashed"), ("いや", "No"), ("え", "Huh")]
        answer = MagicMock()
        answer.__enter__.return_value.read.return_value = json.dumps(
            {"choices":[{"message":{"content":"the deputy chief"}}]}
        ).encode("utf-8")
        with patch("auto_translate.translate.plamo.urllib.request.urlopen", return_value=answer) as post:
            text = PlamoTranslator._chat(translator, "副署長", history, timeout=1.0)
        self.assertEqual(text, "the deputy chief")
        req = post.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertIn({"source":"副署長","target":"Deputy Police Chief"},
                      body["chat_template_kwargs"]["glossary"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "副署長"}])
        self.assertNotIn("I crashed", body["chat_template_kwargs"]["context"])
        self.assertIn("Recent Japanese", body["chat_template_kwargs"]["context"])
        self.assertIn("Video title", body["chat_template_kwargs"]["context"])
        self.assertNotIn("署長", [x["source"] for x in body["chat_template_kwargs"]["glossary"]
                                  if x["source"] != "副署長"])


if __name__ == "__main__":
    unittest.main()
