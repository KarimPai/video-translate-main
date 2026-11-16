"""Matsuri's person name survives translation; real festivals remain events."""

from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import Mock

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import PlamoTranslator


class MatsuriTranslationTests(unittest.TestCase):
    def translator(self, responses, *, title="ホロGTA", **options):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig(sentence_context=False, **options)
        translator.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
        translator.vod_title = title
        translator.memory_lines = 0
        translator.validation_stats = Counter()
        translator._chat = Mock(side_effect=responses)
        return translator

    def test_natural_name_syntax_and_honorifics(self):
        for japanese, target in (("まつりはどうした", "Matsuri"), ("マツリはどうした", "Matsuri"),
                                  ("まつりによろしく", "Matsuri"), ("まつりさんはどうした", "Matsuri-san"),
                                  ("夏色まつりはどうした", "Natsuiro Matsuri"),
                                  ("夏色まつりさんはどうした", "Natsuiro Matsuri-san")):
            with self.subTest(japanese=japanese):
                translator = self.translator(["What happened to the festival?"])
                self.assertEqual(translator.translate(japanese, []), f"What happened to {target}?")
                self.assertEqual(translator.last_translation_flags, [])
                translator._chat.assert_called_once()

    def test_shrine_location_does_not_veto_the_person(self):
        translator = self.translator(["The festival went to the shrine."])
        self.assertEqual(translator.translate("まつりは神社に行った", []),
                         "Matsuri went to the shrine.")

    def test_observed_roster_cue_preserves_matsuri(self):
        # User's Haas/PLaMo output at 7585.29-7593.06 in Q8RQPpzKapQ.
        japanese = "okいろは祭りあとビジューカリスマ"
        translator = self.translator(["Ok Iroha Festival, then Bijou Charisma"])
        self.assertEqual(translator.translate(japanese, []), "Ok Iroha Matsuri, then Bijou Charisma")
        self.assertEqual(translator._chat.call_args.args[0], "okいろはMatsuriあとビジューカリスマ")
        self.assertIn(("Matsuri", "Matsuri"), translator._chat.call_args.kwargs["names"])
        self.assertEqual(translator.last_name_resolutions[0]["japanese"], japanese)
        self.assertEqual(translator.last_name_resolutions[0]["source"], "祭り")

    def test_observed_self_correction_preserves_negation_and_corrected_reading(self):
        # User's source at 7794.63-7799.45; まとり must not become Matsuri.
        japanese = "何もいない祭りあー祭りじゃないまとりサーマル使って"
        expected = "No one here. Matsuri—ah, not Matsuri. Matori, use the thermal."
        translator = self.translator([
            "Nothing's here—it's not a festival. Use the Matori Thermal.", expected,
        ])
        self.assertEqual(translator.translate(japanese, []), expected)
        self.assertEqual(translator._chat.call_count, 2)
        self.assertEqual(translator._chat.call_args.args[0],
                         "何もいないMatsuriあーMatsuriじゃないまとりサーマル使って")
        self.assertEqual(translator._chat.call_args.args[1], [])
        self.assertEqual([item["source"] for item in translator.last_name_resolutions], ["祭り", "祭り"])
        self.assertEqual(translator.last_translation_flags, [])

    def test_repeated_calls_are_not_left_as_festivals(self):
        translator = self.translator(["Festival! Festival!"])
        self.assertEqual(translator.translate("まつり！まつり！", []), "Matsuri! Matsuri!")
        translator._chat.assert_called_once()

    def test_person_and_real_event_survive_together(self):
        english = "Going to the summer festival with Matsuri."
        translator = self.translator([english])
        self.assertEqual(translator.translate("まつりと夏祭りに行く", []), english)
        self.assertEqual(translator.last_translation_flags, [])

    def test_person_can_come_to_a_deictic_festival(self):
        translator = self.translator(["Matsuri is coming to this festival."])
        self.assertEqual(translator.translate("まつりはこのお祭りに来る", []),
                         "Matsuri is coming to this festival.")
        self.assertEqual([item["source"] for item in translator.last_name_resolutions], ["まつり"])

    def test_same_spelling_name_and_event_uses_only_the_name_span(self):
        translator = self.translator(["Matsuri is coming to the festival."])
        japanese = "まつりはこのまつりに来る"
        self.assertEqual(translator.translate(japanese, []), "Matsuri is coming to the festival.")
        self.assertEqual(translator._chat.call_args.args[0], "Matsuriはこのまつりに来る")
        self.assertIn(("Matsuri", "Matsuri"), translator._chat.call_args.kwargs["names"])
        self.assertNotIn(("まつり", "Matsuri"), translator._chat.call_args.kwargs["names"])

    def test_missing_full_name_or_honorific_retries_and_is_reviewable(self):
        for japanese in ("夏色まつりが来た", "夏色まつりさんが来た"):
            with self.subTest(japanese=japanese):
                translator = self.translator(["Someone arrived.", "Someone arrived."])
                self.assertEqual(translator.translate(japanese, []), "Someone arrived.")
                self.assertEqual(translator._chat.call_count, 2)
                self.assertEqual(translator.last_translation_flags, ["proper_name_mismatch"])

    def test_recent_japanese_name_is_available_after_an_intervening_cue(self):
        translator = self.translator(["Where's the festival?"], title="Minecraft")
        history = [("夏色まつりが来た", "Natsuiro Matsuri arrived."), ("ちょっと待って", "Wait.")]
        self.assertEqual(translator.translate("マツリはどこ", history), "Where's Matsuri?")

    def test_english_guess_does_not_establish_person_context(self):
        translator = self.translator(["The festival arrived."], title="Japanese culture tour")
        self.assertEqual(translator.translate("祭りが来た", [("了解", "Natsuiro Matsuri.")]),
                         "The festival arrived.")
        self.assertEqual(translator.last_name_resolutions, [])

    def test_literal_festivals_and_compounds_remain_unchanged(self):
        for japanese, english in (("お祭りに行く", "Going to the festival."),
                                  ("夏祭りが来た", "The summer festival arrived."),
                                  ("祭りの準備", "Preparing for the festival."),
                                  ("まつりは明日だ", "The festival is tomorrow."),
                                  ("まつりが楽しみ", "Looking forward to the festival."),
                                  ("ひなまつりです", "It's the Doll Festival."),
                                  ("しまつりさんが来た", "Shimatsuri-san arrived.")):
            with self.subTest(japanese=japanese):
                translator = self.translator([english])
                self.assertEqual(translator.translate(japanese, []), english)
                self.assertEqual(translator.last_name_resolutions, [])

    def test_native_request_uses_person_glossary_and_explains_separate_event(self):
        translator = self.translator([])
        del translator._chat
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        translator.client = Mock()
        translator.client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Going to the summer festival with Matsuri."}}],
        }
        translator.translate("まつりと夏祭りに行く", [])
        options = translator.client.post.call_args.kwargs["json"]["chat_template_kwargs"]
        self.assertIn({"source": "まつり", "target": "Matsuri"}, options["glossary"])
        self.assertIn("person names for Natsuiro Matsuri", options["context"])
        self.assertIn("separate literal festival", options["context"])


if __name__ == "__main__":
    unittest.main()
