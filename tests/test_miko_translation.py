"""Miko translation requests and repairs, without loading ASR or PLaMo."""

from collections import Counter
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

import httpx

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.base import TranslationCancelled
from auto_translate.translate.plamo import PlamoTranslator


class MikoTranslationTests(unittest.TestCase):
    def translator(self, responses, *, title="ホロGTA", **options):
        translator = object.__new__(PlamoTranslator)
        translator.cfg = TranslateConfig(sentence_context=False, **options)
        translator.glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
        translator.vod_title = title
        translator.memory_lines = 0
        translator.validation_stats = Counter()
        translator._chat = Mock(side_effect=responses)
        return translator

    def test_kana_and_honorifics_repair_existing_shrine_maiden_output(self):
        for japanese, expected in (("ミコはどうした", "What happened to Miko?"),
                                   ("みこさんはどうした", "What happened to Miko-san?"),
                                   ("ミコ様はどうした", "What happened to Miko-sama?"),
                                   ("ミコチはどうした", "What happened to Mikochi?")):
            with self.subTest(japanese=japanese):
                translator = self.translator(["What happened to the shrine maiden?"])
                self.assertEqual(translator.translate(japanese, []), expected)
                translator._chat.assert_called_once()
                self.assertEqual(translator.last_translation_flags, [])

    def test_shrine_location_does_not_block_a_confirmed_name(self):
        translator = self.translator(["The shrine maiden went to the shrine."])
        self.assertEqual(translator.translate("ミコは神社に行った", []),
                         "Miko went to the shrine.")

    def test_mixed_name_and_role_retries_without_replacing_real_role(self):
        translator = self.translator([
            "The shrine maiden is a shrine maiden idol.",
            "Miko is a shrine maiden idol.",
        ])
        self.assertEqual(translator.translate("みこは巫女アイドルです", []),
                         "Miko is a shrine maiden idol.")
        self.assertEqual(translator._chat.call_count, 2)
        self.assertEqual(translator._chat.call_args.args[1], [])
        self.assertEqual(translator.last_translation_flags, [])

    def test_kanji_homophone_changes_only_translation_input(self):
        translator = self.translator(["The shrine maiden arrived."], title="Sakura Miko Ch.")
        japanese = "巫女が来た"
        self.assertEqual(translator.translate(japanese, []), "Miko arrived.")
        self.assertEqual(japanese, "巫女が来た")
        self.assertEqual(translator._chat.call_args.args[0], "Mikoが来た")
        self.assertEqual(translator._chat.call_args.kwargs["names"], [("Miko", "Miko")])
        evidence = translator.last_name_resolutions[0]
        self.assertEqual((evidence["source"], evidence["target"]), ("巫女", "Miko"))
        self.assertEqual(evidence["japanese"], japanese)
        self.assertEqual(evidence["translation_source"], "Mikoが来た")

    def test_same_surface_name_and_literal_role_are_disambiguated_by_span(self):
        for japanese, expected_input in (
            ("みこが来た、神社のみこも来た", "Mikoが来た、神社のみこも来た"),
        ):
            with self.subTest(japanese=japanese):
                english = "Miko arrived, and the shrine maiden at the shrine arrived too."
                translator = self.translator([english], title="Sakura Miko Ch.")
                self.assertEqual(translator.translate(japanese, []), english)
                self.assertEqual(translator._chat.call_args.args[0], expected_input)
                self.assertEqual(translator._chat.call_args.kwargs["names"], [("Miko", "Miko")])
                self.assertEqual(len(translator.last_name_resolutions), 1)

    def test_literal_roles_and_unrelated_words_are_unchanged(self):
        for japanese, english in (("神社の巫女さん", "The shrine maiden at the shrine."),
                                  ("巫女アイドルです", "A shrine maiden idol."),
                                  ("みこしが来た", "The portable shrine arrived."),
                                  ("みこと", "Mikoto."),
                                  ("今日は晴れ", "Sunny today.")):
            with self.subTest(japanese=japanese):
                translator = self.translator([english], title="Sakura Miko Ch.")
                self.assertEqual(translator.translate(japanese, []), english)
                self.assertEqual(translator._chat.call_args.args[0], japanese)
                self.assertEqual(translator.last_name_resolutions, [])

    def test_shrine_topic_across_clauses_keeps_ambiguous_kanji_literal(self):
        translator = self.translator(["Going to the shrine today. The shrine maiden arrived."],
                                     title="Sakura Miko Ch.")
        japanese = "今日は神社に行く。巫女が来た"
        self.assertEqual(translator.translate(japanese, []),
                         "Going to the shrine today. The shrine maiden arrived.")
        self.assertEqual(translator._chat.call_args.args[0], japanese)
        self.assertEqual(translator.last_name_resolutions, [])

    def test_polite_role_without_name_context_stays_literal(self):
        translator = self.translator(["The shrine maiden arrived."], title="Japanese culture tour")
        self.assertEqual(translator.translate("みこさんが来た", []), "The shrine maiden arrived.")
        self.assertEqual(translator.last_name_resolutions, [])

    def test_full_name_honorific_is_preserved(self):
        translator = self.translator(["The shrine maiden arrived."])
        self.assertEqual(translator.translate("さくらみこさんが来た", []), "Sakura Miko-san arrived.")
        translator._chat.assert_called_once()

    def test_general_hololive_context_does_not_promote_kanji_role(self):
        translator = self.translator(["The shrine maiden arrived."])
        self.assertEqual(translator.translate("巫女が来た", []), "The shrine maiden arrived.")
        self.assertEqual(translator._chat.call_args.args[0], "巫女が来た")
        self.assertEqual(translator.last_name_resolutions, [])

    def test_recent_japanese_identity_survives_one_retry_without_history(self):
        translator = self.translator(["Someone arrived.", "Miko arrived."], title="Minecraft")
        history = [("さくらみこだよ", "Sakura Miko."), ("ちょっと待って", "Wait.")]
        self.assertEqual(translator.translate("巫女が来た", history), "Miko arrived.")
        self.assertEqual([call.args[0] for call in translator._chat.call_args_list],
                         ["Mikoが来た", "Mikoが来た"])
        self.assertEqual(translator._chat.call_args.args[1], [])
        self.assertEqual(translator.last_translation_flags, [])

    def test_old_or_english_only_identity_does_not_create_a_name(self):
        for history in (
            [("了解", "Sakura Miko is here.")],
            [("さくらみこだよ", "Sakura Miko.")] + [("了解", "Okay.")] * 4,
        ):
            with self.subTest(history=history):
                translator = self.translator(["The shrine maiden arrived."], title="Minecraft")
                self.assertEqual(translator.translate("巫女が来た", history), "The shrine maiden arrived.")
                self.assertEqual(translator.last_name_resolutions, [])

    def test_unresolved_full_name_is_reviewable_and_flags_reset_next_cue(self):
        translator = self.translator(["Someone arrived.", "Someone came.", "Hello."])
        self.assertEqual(translator.translate("さくらみこが来た", []), "Someone arrived.")
        self.assertEqual(translator._chat.call_count, 2)
        self.assertEqual(translator.last_translation_flags, ["proper_name_mismatch"])
        self.assertEqual(translator.translate("こんにちは", []), "Hello.")
        self.assertEqual(translator.last_translation_flags, [])
        self.assertEqual(translator.last_name_resolutions, [])

    def test_disabled_retry_still_marks_missing_name_for_review(self):
        translator = self.translator(["Someone arrived."], retry_suspicious=False)
        self.assertEqual(translator.translate("ミコが来た", []), "Someone arrived.")
        translator._chat.assert_called_once()
        self.assertEqual(translator.last_translation_flags, ["proper_name_mismatch"])

    def test_cancelled_cue_clears_previous_name_diagnostics(self):
        translator = self.translator(["Miko arrived."])
        translator.translate("みこが来た", [])
        self.assertTrue(translator.last_name_resolutions)
        with self.assertRaises(TranslationCancelled):
            translator.translate_cancellable("ミコが来た", [], cancelled=lambda: True)
        self.assertEqual(translator.last_name_resolutions, [])
        self.assertEqual(translator.last_translation_flags, [])

    def test_owned_server_recovery_discards_failed_attempt_diagnostics(self):
        translator = self.translator([])
        translator._close_lock = threading.Lock()
        translator._closed = translator._restart_used = False
        translator.process = Mock()
        translator.process.poll.return_value = 1
        translator._server_log = None
        translator._start_server = Mock()
        calls = []

        def translate_cue(*_):
            calls.append(True)
            if len(calls) == 1:
                translator.last_name_resolutions.append({"source": "failed attempt"})
                translator.last_translation_flags.append("proper_name_mismatch")
                raise httpx.ConnectError("owned server exited")
            translator.last_name_resolutions.append({"source": "さくらみこ", "target": "Sakura Miko"})
            return "Sakura Miko arrived."

        translator._translate_cue = translate_cue
        self.assertEqual(translator.translate("さくらみこが来た", []), "Sakura Miko arrived.")
        self.assertEqual(translator.last_translation_flags, [])
        self.assertEqual(translator.last_name_resolutions,
                         [{"source": "さくらみこ", "target": "Sakura Miko"}])
        translator._start_server.assert_called_once()

    def test_native_request_carries_person_glossary_and_separate_literal_role(self):
        translator = self.translator([], title="Sakura Miko Ch.")
        del translator._chat
        translator.url = "http://127.0.0.1:8000/v1/chat/completions"
        translator.client = Mock()
        translator.client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Miko arrived. The shrine maiden is at work."}}],
        }
        japanese = "みこが来た、神社のみこは仕事中"
        translator.translate(japanese, [])
        body = translator.client.post.call_args.kwargs["json"]
        self.assertEqual(body["messages"][-1]["content"], "Mikoが来た、神社のみこは仕事中")
        self.assertIn({"source": "Miko", "target": "Miko"},
                      body["chat_template_kwargs"]["glossary"])
        self.assertNotIn({"source": "巫女", "target": "Miko"},
                         body["chat_template_kwargs"]["glossary"])
        self.assertIn("separate literal occupation", body["chat_template_kwargs"]["context"])

    def test_recent_name_window_does_not_expand_other_role_policies(self):
        translator = self.translator(["The captain."], title="ホロライブ")
        translator.translate("船長", [("宝鐘マリンだよ", "Houshou Marine."), ("了解", "Okay.")])
        self.assertNotIn(("船長", "Senchou"), translator._chat.call_args.kwargs["names"])


if __name__ == "__main__":
    unittest.main()
