"""CPU-only checks for direct Japanese -> Standard Malay translation branch."""
import json
import queue
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_translate.config import Config
from auto_translate.glossary import Glossary
from auto_translate.malay_translation_guard import suspicious_malay_translation
from auto_translate.pipeline import run_translate
from auto_translate.subtitle_clean import clean_malay_output
from auto_translate.translate.plamo import PlamoTranslator, _current_glossary_matches
from auto_translate.types import Transcript


class MalayOutputTests(unittest.TestCase):
    def test_malay_defaults_and_glossary(self):
        cfg = Config()
        self.assertEqual(cfg.translate.target_lang, "ms")
        self.assertEqual(cfg.translate.glossary, "glossary-ms.toml")
        self.assertFalse(cfg.translate.neutral_subjects)
        glossary = Glossary.load(Path(__file__).parents[1] / cfg.translate.glossary)
        self.assertIn(("わため", "Watame"), glossary.find("わため"))
        self.assertEqual(glossary.find("卒業"), [])
        self.assertGreater(len(glossary.terms), 70)

    def test_glossary_hologta_uses_malay(self):
        matches = _current_glossary_matches(
            "副署長と警察", "ホロGTA", Glossary({}), target_lang="ms",
        )
        self.assertIn(("副署長", "timbalan ketua polis"), matches)
        self.assertIn(("警察", "polis"), matches)

    def test_malay_prompt_targets_ms_and_preserves_context(self):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = Config().translate
        obj.glossary = Glossary({})
        obj.vod_title = "ホロGTA"
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.memory_lines = 0
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices":[{"message":{"content":"Timbalan ketua polis"}}]}
        ).encode("utf-8")
        with patch("auto_translate.translate.plamo.urllib.request.urlopen", return_value=response) as post:
            result = obj._chat("副署長", [("分からない", "Saya tak tahu")], 1.0)
        self.assertEqual(result, "Timbalan ketua polis")
        payload = json.loads(post.call_args[0][0].data.decode("utf-8"))
        opts = payload["chat_template_kwargs"]
        self.assertEqual(opts["source_lang"], "ja")
        self.assertEqual(opts["target_lang"], "ms")
        self.assertIn("Bahasa Melayu", opts["context"])
        self.assertIn({"source":"副署長","target":"timbalan ketua polis"}, opts["glossary"])
        self.assertEqual(payload["messages"][1]["content"], "Saya tak tahu")

    def test_no_english_cleaning_changes_malay_negation(self):
        self.assertEqual(
            clean_malay_output('  "Tak, tak, jangan pergi!"  '),
            "Tak, tak, jangan pergi!",
        )
        self.assertEqual(
            clean_malay_output("Saya belum faham lagi."),
            "Saya belum faham lagi.",
        )

    def test_malay_guard(self):
        self.assertIsNone(suspicious_malay_translation("行かない", "Saya tak pergi."))
        self.assertEqual(
            suspicious_malay_translation("行かない", "Saya akan pergi."),
            "possible_negation_loss",
        )
        self.assertEqual(
            suspicious_malay_translation("50人", "Ada 40 orang."),
            "numeric_mismatch",
        )
        self.assertEqual(
            suspicious_malay_translation("わため", "Watanabe", names=[("わため", "Watame")]),
            "proper_name_mismatch",
        )

    def test_pipeline_preserves_short_malay_interjections(self):
        class FakeTranslator:
            cfg = Config().translate
            memory_lines = 0
            def translate(self, ja, history):
                return "Tak, tak, jangan!"
        inp, out = queue.Queue(), queue.Queue()
        inp.put(Transcript(1, 2, "ダメダメ"))
        inp.put(None)
        run_translate(FakeTranslator(), 0, inp, out)
        self.assertEqual(out.get().en, "Tak, tak, jangan!")


if __name__ == "__main__":
    unittest.main()
