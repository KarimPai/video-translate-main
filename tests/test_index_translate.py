"""Offline Index-Translate-2B tests: model, prompt, glossary, cleanup."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.index import (
    IndexTranslator,
    _clean,
    build_index_prompt,
)


class IndexTranslationTests(unittest.TestCase):
    def test_short_translation_prompt_has_no_leak_prone_constraint_headers(self):
        glossary = Glossary({"角巻わため": "Tsunomaki Watame", "兎田ぺこら": "Usada Pekora"})
        prompt = build_index_prompt(
            "角巻わためと行くよ", [("兎田ぺこらだよ", "I'm Pekora")],
            glossary, vod_title="Hololive GTA RP", context_lines=4,
        )
        self.assertIn("角巻わためと行くよ", prompt)
        self.assertIn("角巻わため = Tsunomaki Watame", prompt)
        self.assertNotIn("兎田ぺこら = Usada Pekora", prompt)
        self.assertNotIn("I'm Pekora", prompt)
        self.assertNotIn("〖", prompt)
        self.assertNotIn("Constraint Requirements", prompt)
        self.assertNotIn("硬性要求", prompt)
        self.assertIn("被打断或未说完的句子只翻译已经说出的内容", prompt)
        self.assertNotIn("Hololive GTA RP", prompt)
        self.assertEqual(prompt.count("角巻わためと行くよ"), 1)
        self.assertTrue(prompt.endswith("\n\n角巻わためと行くよ"))


    def test_reference_context_uses_two_japanese_lines_only(self):
        glossary = Glossary({"現行犯": "caught in the act"})
        history = [
            ("古い発言", "OLD ENGLISH"),
            ("警察が来た", "POLICE COMING"),
            ("現行犯だ", "Fugitive"),
        ]
        prompt = build_index_prompt("どうするの", history, glossary)
        self.assertIn("警察が来た", prompt)
        self.assertIn("現行犯だ", prompt)
        self.assertNotIn("古い発言", prompt)
        self.assertNotIn("OLD ENGLISH", prompt)
        self.assertNotIn("POLICE COMING", prompt)
        self.assertNotIn("Fugitive", prompt)
        self.assertTrue(prompt.endswith("\n\nどうするの"))
        self.assertNotIn("〖", prompt)

    def test_context_is_disabled_by_zero_and_has_no_target_leak(self):
        glossary = Glossary({})
        prompt = build_index_prompt("今どこ", [("前の発言", "Where now")],
                                    glossary, context_lines=0)
        self.assertNotIn("前の発言", prompt)
        self.assertNotIn("Where now", prompt)
        self.assertTrue(prompt.endswith("今どこ"))

    def test_retry_adds_short_quality_instruction_not_previous_english(self):
        prompt = build_index_prompt(
            "現行犯だ", [("張り込み", "Confront me")], Glossary({}),
            review_reason="missing_priority_term:現行犯",
        )
        self.assertIn("重新核对本句的否定", prompt)
        self.assertNotIn("Confront me", prompt)
        self.assertIn("張り込み", prompt)
        self.assertNotIn("〖", prompt)

    @staticmethod
    def _unit():
        obj = object.__new__(IndexTranslator)
        obj.cfg = TranslateConfig()
        obj.glossary = Glossary({"角巻わため": "Tsunomaki Watame"})
        obj.vod_title = "Hololive GTA"
        obj.base_url = obj.cfg.index_base_url
        obj.url = obj.base_url + "/chat/completions"
        obj.memory_lines = 0
        return obj

    def test_request_uses_one_user_message_greedy_no_thinking(self):
        obj = self._unit()
        reply = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "I'm off!"}}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.index.urllib.request.urlopen",
                   return_value=reply) as urlopen:
            result = obj._chat("行ってきます", [], timeout=4.0)
        self.assertEqual(result, "I'm off!")
        request = urlopen.call_args.args[0]
        data = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8002/v1/chat/completions")
        self.assertEqual(data["model"], "index-translate-2b")
        self.assertEqual(data["temperature"], 0)
        self.assertEqual(data["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual([message["role"] for message in data["messages"]], ["user"])
        self.assertNotIn("system", str([msg["role"] for msg in data["messages"]]))
        self.assertNotIn("top_p", data)
        self.assertNotIn("qwen", data)
        self.assertIn("〖", data["stop"])
        self.assertIn("Constraint Requirements", data["stop"])
        self.assertNotIn("〖源文〗", data["messages"][0]["content"])

    def test_short_casual_line_is_preserved_without_model_rewrite(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="I'm off!") as call:
            self.assertEqual(obj.translate("行ってきます", []), "I'm off!")
        call.assert_called_once()

    def test_exact_original_song_is_not_romanized(self):
        obj = self._unit()
        with patch.object(obj, "_chat") as call:
            self.assertEqual(obj.translate("オリ曲!", []), "Original song!")
            self.assertEqual(obj.translate("オリジナル曲", []), "Original song")
        call.assert_not_called()

    def test_suspicious_roleplay_term_gets_only_one_bounded_retry(self):
        obj = self._unit()
        with patch.object(
            obj, "_chat",
            side_effect=["So, he's a fugitive.", "So, he was caught in the act."],
        ) as call:
            result = obj.translate("現行犯ということで", [])
        self.assertEqual(result, "So, he was caught in the act.")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[1].kwargs["review_reason"],
                         "missing_priority_term:現行犯")
        self.assertLessEqual(call.call_args_list[1].kwargs["timeout"], 8.0)

    def test_suspicious_translation_keeps_first_if_retry_is_still_wrong(self):
        obj = self._unit()
        with patch.object(
            obj, "_chat", side_effect=["Confront me.", "Go away."]
        ) as call:
            result = obj.translate("張り込みしてほしい", [])
        self.assertEqual(result, "Confront me.")
        self.assertEqual(call.call_count, 2)

    def test_quality_retry_can_be_disabled_without_other_translator(self):
        obj = self._unit()
        obj.cfg.quality_retry_enabled = False
        with patch.object(obj, "_chat", return_value="They called my name.") as call:
            result = obj.translate("指名手配されてますよ", [])
        self.assertEqual(result, "They called my name.")
        call.assert_called_once()

    def test_recheck_errors_keep_first_translation(self):
        obj = self._unit()
        with patch.object(obj, "_chat", side_effect=[
            "They called my name.", TimeoutError("recheck time budget")
        ]) as call:
            result = obj.translate("指名手配されてますよ", [])
        self.assertEqual(result, "They called my name.")
        self.assertEqual(call.call_count, 2)

    def test_clean_translation_does_not_retry(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="You're wanted by police.") as call:
            result = obj.translate("指名手配されてますよ", [])
        self.assertEqual(result, "You're wanted by police.")
        call.assert_called_once()

    def test_glossary_priority_anchors_not_triggered_without_source_terms(self):
        obj = self._unit()
        self.assertIsNone(obj._review_reason(
            "何をしてる", "They're wanted by police."
        ))

    def test_unintelligible_repetitions_skip_model_call(self):
        obj = self._unit()
        with patch.object(obj, "_chat") as chat:
            self.assertEqual(obj.translate("", []), "[SKIP]")
        chat.assert_not_called()

    def test_clean_thinking_and_headers(self):
        self.assertEqual(_clean("<think>notes</think>Translation: Let's go!"), "Let's go!")


    def test_real_stream_leakage_english_instructions_are_truncated(self):
        observed = (
            "Come on, come on! This is it!  〖Constraint Requirements〗 "
            "1. 〖Hard Requirement〗Only translate the Japanese in〖Source〗into English"
        )
        self.assertEqual(_clean(observed), "Come on, come on! This is it!")

    def test_real_stream_leakage_chinese_instructions_are_truncated(self):
        observed = (
            "I'm coming with you!  〖约束要求〗 "
            "1. 〖硬性要求〗只翻译日语"
        )
        self.assertEqual(_clean(observed), "I'm coming with you!")

    def test_meta_only_output_is_never_used_as_a_subtitle(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value=(
            "〖Constraint Requirements〗 1. 〖Hard Requirement〗Only translate..."
        )):
            self.assertEqual(
                obj.translate("行ってきます", []), "[translation failed]"
            )

    def test_leaked_instruction_is_removed_before_translation_corrections(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value=(
            "It's okay!  〖Source〗 Go away  〖Constraint Requirements〗"
        )):
            result = obj.translate("大丈夫です", [])
        self.assertEqual(result, "It's okay!")
        self.assertNotIn("〖", result)

    def test_clean_handles_plain_english_metadata_and_preserves_dialogue(self):
        self.assertEqual(
            _clean("Let's go.  Constraint Requirements: 1. Hard Requirement: translate"),
            "Let's go."
        )
        self.assertEqual(_clean("No, no, no!"), "No, no, no!")
        self.assertEqual(
            _clean("Translation: I'll head to the carjacking scene."),
            "I'll head to the carjacking scene."
        )

    def test_managed_server_selects_index_hf_model_and_new_port(self):
        obj = self._unit()
        obj.process = None
        obj._job = None
        obj._server_log = None
        obj._registered_exit = False
        process = Mock()
        process.poll.return_value = None
        process.pid = 1234
        with patch("auto_translate.translate.index._model_ids",
                   side_effect=[None, ["index-translate-2b"]]), patch(
            "auto_translate.translate.index.shutil.which",
            return_value="llama-server.exe",
        ), patch(
            "auto_translate.translate.index.subprocess.Popen",
            return_value=process,
        ) as popen, patch(
            "auto_translate.translate.index._create_kill_job",
            return_value=None,
        ), patch("auto_translate.translate.index.time.sleep"):
            obj._start_server()
            args = popen.call_args.args[0]
            self.assertEqual(args[args.index("-hf") + 1],
                             "IndexTeam/Index-Translate-2B-GGUF:Q4_K_M")
            self.assertEqual(args[args.index("--port") + 1], "8002")
            self.assertEqual(args[args.index("--alias") + 1], "index-translate-2b")
            self.assertIn("--jinja", args)
            self.assertIn("--no-mmproj", args)
            self.assertIn("--reasoning", args)
            obj.close()
        process.terminate.assert_called_once()

    def test_existing_server_is_never_adopted(self):
        obj = self._unit()
        with patch("auto_translate.translate.index._model_ids",
                   return_value=["stale-other-model"]), patch(
                   "auto_translate.translate.index.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "existing llama.cpp server"):
                obj._start_server()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
