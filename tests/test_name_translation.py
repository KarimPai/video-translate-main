"""Deterministic name preservation tests; no acoustic/model accuracy claims."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.proper_names import resolve_names, name_glossary
from auto_translate.translate.plamo import (
    PlamoTranslator, _current_glossary_matches, _trusted_history_pair,
)
from auto_translate.translation_guard import resolved_name_issue, suspicious_translation


class NameTranslationTests(unittest.TestCase):
    def translator(self, *responses):
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.cfg.sentence_context = False
        obj.glossary = Glossary.load(Path("glossary.toml"))
        obj.vod_title = "ホロライブ まつり みこ"
        obj.memory_lines = 0
        obj._chat = Mock(side_effect=responses)
        return obj

    def test_title_only_ambiguous_word_never_becomes_confirmed_glossary(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for ja in ("まつりが来た", "みこが来た", "まつり", "みこ"):
            with self.subTest(ja=ja):
                self.assertFalse(_current_glossary_matches(ja, "ホロライブ まつり みこ", glossary))

    def test_confirmed_matsuri_and_miko_literal_repairs(self):
        for ja, previous, raw, expected in (
            ("まつりが来た", "夏色まつりだよ", "The festival arrived.", "Matsuri arrived."),
            ("みこが来た", "みこちだよ", "The shrine maiden arrived.", "Miko arrived."),
            ("祭りが来た", "夏色まつりだよ", "The festival arrived.", "Matsuri arrived."),
            ("巫女が来た", "みこちだよ", "The shrine maiden arrived.", "Miko arrived."),
        ):
            with self.subTest(ja=ja):
                obj = self.translator(raw)
                self.assertEqual(obj.translate(ja, [(previous, "")]), expected)
                obj._chat.assert_called_once()
                self.assertEqual(obj.last_name_diagnostics["raw_english"], raw)
                self.assertEqual(obj.last_name_diagnostics["corrected_english"], expected)

    def test_common_noun_keeps_literal_meaning(self):
        for ja, raw in (("お祭りに行く", "Going to the festival."),
                        ("巫女さんが神社にいる", "A shrine maiden is at the shrine.")):
            obj = self.translator(raw)
            self.assertEqual(obj.translate(ja, []), raw)
            obj._chat.assert_called_once()

    def test_spoken_honorifics_and_multiple_names(self):
        obj = self.translator("Natsuiro Matsuri and Sakura Miko arrived.")
        self.assertEqual(obj.translate("まつりちゃんとみこちゃんが来た", []),
                         "Matsuri-chan and Miko-chan arrived.")
        obj._chat.assert_called_once()

    def test_mixed_meanings_retry_instead_of_global_noun_replacement(self):
        obj = self.translator("The festival came to the school festival.",
                              "Matsuri came to the school festival.")
        ja = "まつりが文化祭に来た"
        self.assertEqual(obj.translate(ja, [("夏色まつりだよ", "")]),
                         "Matsuri came to the school festival.")
        self.assertEqual(obj._chat.call_count, 2)
        self.assertTrue(obj.last_name_diagnostics["retry"])
        self.assertTrue(obj.last_name_diagnostics["retry_accepted"])

    def test_mixed_identical_source_omits_global_glossary_and_still_validates(self):
        ja = "まつりが来た。おまつりに行く。"
        mentions = resolve_names(ja, previous_ja="夏色まつりだよ")
        self.assertNotIn(("まつり", "Matsuri"), name_glossary(mentions))
        self.assertEqual(resolved_name_issue(ja, "The festival came. Going to the festival.", mentions),
                         "proper_name_mismatch")
        self.assertIsNone(resolved_name_issue(ja, "Matsuri came. Going to the festival.", mentions))

    def test_failed_name_retry_abstains_and_allows_next_cue(self):
        obj = self.translator("The festival came to the school festival.",
                              TimeoutError("classification budget"), "Hello!")
        ja = "まつりが文化祭に来た"
        self.assertEqual(obj.translate(ja, [("夏色まつりだよ", "")]), "")
        self.assertTrue(obj.last_name_diagnostics["abstained"])
        self.assertEqual(obj.translate("こんにちは", []), "Hello!")

    def test_mixed_literal_cannot_become_second_person_name(self):
        ja = "まつりが文化祭に来た"
        mentions = resolve_names(ja, previous_ja="夏色まつりだよ")
        self.assertEqual(resolved_name_issue(ja, "Matsuri came to Matsuri.", mentions),
                         "literal_noun_mismatch")

    def test_retry_romanizes_only_confirmed_person_spans_in_translation_input(self):
        obj = self.translator()
        del obj._chat
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.client = MagicMock()
        obj.client.post.return_value.json.side_effect = [
            {"choices": [{"message": {"content": "The festival came to the school festival."}}]},
            {"choices": [{"message": {"content": "Matsuri came to the school festival."}}]},
        ]
        ja = "まつりが文化祭に来た"
        self.assertEqual(obj.translate(ja, [("夏色まつりだよ", "")]),
                         "Matsuri came to the school festival.")
        first, retry = [call.kwargs["json"] for call in obj.client.post.call_args_list]
        self.assertEqual(first["messages"][-1]["content"], ja)
        self.assertEqual(retry["messages"][-1]["content"], "Matsuriが文化祭に来た")
        self.assertIn(ja, retry["chat_template_kwargs"]["context"])
        self.assertIsNone(obj._name_context)

    def test_literal_noun_mistranslated_as_name_gets_retry(self):
        obj = self.translator("Going to Matsuri.", "Going to the festival.")
        self.assertEqual(obj.translate("お祭りに行く", []), "Going to the festival.")
        self.assertTrue(obj.last_name_diagnostics["retry"])
        self.assertEqual(obj.last_name_diagnostics["retry_reason"], "literal_noun_mismatch")

    def test_kanji_source_names_are_validated(self):
        ja = "巫女が来た"
        mentions = resolve_names(ja, previous_ja="みこちだよ")
        self.assertEqual(suspicious_translation(ja, "Someone arrived.", mentions=mentions),
                         "proper_name_mismatch")
        self.assertIsNone(suspicious_translation(ja, "Miko arrived.", mentions=mentions))

    def test_repeated_name_requires_each_english_occurrence(self):
        ja = "まつりちゃん、まつりちゃん！"
        mentions = resolve_names(ja)
        self.assertEqual(resolved_name_issue(ja, "Matsuri-chan!", mentions), "proper_name_mismatch")
        self.assertIsNone(resolved_name_issue(ja, "Matsuri-chan, Matsuri-chan!", mentions))

    def test_full_name_cannot_also_satisfy_separate_short_name(self):
        for ja, raw in (("夏色まつりとまつりが来た", "Natsuiro Matsuri arrived."),
                        ("さくらみことみこが来た", "Sakura Miko arrived.")):
            with self.subTest(ja=ja):
                mentions = resolve_names(ja)
                self.assertEqual(resolved_name_issue(ja, raw, mentions), "proper_name_mismatch")

    def test_names_absent_from_speech_are_retried_and_rejected(self):
        for ja in ("今日は晴れ", "まつりが来た"):
            obj = self.translator("Matsuri said it is sunny.", "Matsuri arrived.")
            self.assertEqual(obj.translate(ja, []), "")
            self.assertEqual(obj._chat.call_count, 2)
            self.assertEqual(obj.last_name_diagnostics["unresolved_name_issue"], "unsupported_proper_name")

    def test_literal_omission_is_review_signal_not_global_replacement(self):
        ja = "まつりがお祭りに行く"
        mentions = resolve_names(ja, previous_ja="夏色まつりだよ")
        self.assertEqual(resolved_name_issue(ja, "Matsuri is going.", mentions), "possible_literal_noun_loss")
        obj = self.translator("Matsuri is going.", "Matsuri is going to the festival.")
        self.assertEqual(obj.translate(ja, [("夏色まつりだよ", "")]), "Matsuri is going to the festival.")
        self.assertTrue(obj.last_name_diagnostics["retry"])

    def test_short_form_does_not_accept_expanded_stage_name(self):
        mentions = resolve_names("みこちゃん")
        self.assertEqual(resolved_name_issue("みこちゃん", "Sakura Miko", mentions),
                         "proper_name_mismatch")
        mentions = resolve_names("みこが来た", previous_ja="みこちだよ")
        self.assertEqual(resolved_name_issue("みこが来た", "Sakura Miko arrived.", mentions),
                         "proper_name_expansion")

    def test_full_fuwawa_name_is_supported_and_short_form_stays_short(self):
        full_source = "フワワ・アビスガード"
        self.assertIsNone(suspicious_translation(full_source, "Fuwawa Abyssgard",
                                                names=[(full_source, "Fuwawa Abyssgard")], mentions=[]))
        obj = self.translator("Fuwawa Abyssgard")
        self.assertEqual(obj.translate(full_source, []), "Fuwawa Abyssgard")
        obj._chat.assert_called_once()
        obj = self.translator("Fuwawa Abyssgard")
        self.assertEqual(obj.translate("フワワ", []), "Fuwawa")
        obj._chat.assert_called_once()
        self.assertEqual(suspicious_translation("フワワ", "Fuwawa Abyssgard",
                                                names=[("フワワ", "Fuwawa")], mentions=[]),
                         "proper_name_expansion")

    def test_contaminated_or_ambiguous_history_is_rejected(self):
        glossary = Glossary.load(Path("glossary.toml"))
        for ja, en in (("まつり", "Matsuri"), ("みこ", "Miko"),
                       ("まつりちゃんが来た", "The festival arrived."),
                       ("巫女さんが神社にいる", "Miko is at the shrine.")):
            with self.subTest(ja=ja):
                self.assertFalse(_trusted_history_pair(ja, en, "ホロライブ", glossary))

    def test_name_context_consumed_once_and_exact_source_only(self):
        obj = self.translator("The festival arrived.", "The festival arrived.")
        ja = "まつりが来た"
        obj.set_name_context(ja, resolve_names(ja, previous_ja="夏色まつりだよ"))
        self.assertEqual(obj.translate(ja, []), "Matsuri arrived.")
        self.assertIsNone(obj._name_context)
        self.assertEqual(obj.translate(ja, []), "The festival arrived.")
        self.assertTrue(obj.last_name_diagnostics["abstained"])

    def test_resolver_failure_is_advisory(self):
        obj = self.translator("Hello!")
        with patch.object(obj, "_mentions_for", side_effect=RuntimeError("resolver unavailable")):
            self.assertEqual(obj.translate("こんにちは", []), "Hello!")
        self.assertEqual(obj.last_name_diagnostics["error"], "resolver:RuntimeError")
        self.assertTrue(obj.last_name_diagnostics["abstained"])

    def test_resolver_failure_does_not_break_actual_chat_or_history_filtering(self):
        obj = self.translator()
        del obj._chat
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        obj.memory_lines = 16
        with patch("auto_translate.translate.plamo.resolve_names", side_effect=RuntimeError("resolver unavailable")):
            self.assertEqual(obj.translate("こんにちは", [("みこちが来た", "Mikochi arrived.")]), "Hello!")
        self.assertTrue(obj.last_name_diagnostics["abstained"])

    def test_model_payload_uses_supported_glossary_schema_and_occurrence_context(self):
        obj = self.translator()
        del obj._chat
        obj.url = "http://127.0.0.1:8000/v1/chat/completions"
        obj.client = MagicMock()
        obj.client.post.return_value.json.return_value = {"choices": [{"message": {"content": "Matsuri-chan"}}]}
        PlamoTranslator._chat(obj, "まつりちゃん", [], timeout=1)
        body = obj.client.post.call_args.kwargs["json"]
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["messages"][-1]["content"], "まつりちゃん")
        self.assertEqual(body["chat_template_kwargs"]["glossary"],
                         [{"source": "まつりちゃん", "target": "Matsuri-chan"}])
        self.assertIn("Current Japanese name occurrences", body["chat_template_kwargs"]["context"])

    def test_sentence_context_diagnostics_keep_each_name_and_full_english(self):
        obj = self.translator("Matsuri-chan has arrived.", "Mikochi is already here.")
        obj.cfg.sentence_context = True
        ja = "まつりちゃんが来た。みこちはもうここにいる。"
        self.assertEqual(obj.translate(ja, []), "Matsuri-chan has arrived. Mikochi is already here.")
        diagnostics = obj.last_name_diagnostics
        self.assertEqual(len(diagnostics["segments"]), 2)
        self.assertEqual(diagnostics["raw_english"], "Matsuri-chan has arrived. Mikochi is already here.")
        self.assertEqual(diagnostics["corrected_english"], diagnostics["raw_english"])
        self.assertEqual({m["target"] for m in diagnostics["mentions"]}, {"Matsuri-chan", "Mikochi"})
        self.assertTrue(all(ja[m["start"]:m["end"]] == m["source"] for m in diagnostics["mentions"]))


if __name__ == "__main__":
    unittest.main()
