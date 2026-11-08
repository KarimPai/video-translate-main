"""CAT-Translate-3.3B official-prompt and managed-server regression checks."""
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.cat import CatTranslator, _clean, build_cat_prompt, cat_output_problem


class CatTranslationTests(unittest.TestCase):
    def test_native_cat_prompt_is_exact_and_source_only(self):
        self.assertEqual(
            build_cat_prompt(" こらー逃げるなー "),
            "Translate the following Japanese text into English.\n\nこらー逃げるなー",
        )
        for excluded in ("〖", "Constraint Requirements", "Index", "Qwen", "Past dialogue"):
            self.assertNotIn(excluded, build_cat_prompt("わためだよ"))

    @staticmethod
    def _unit():
        obj = object.__new__(CatTranslator)
        obj.cfg = TranslateConfig()
        obj.glossary = Glossary({"わため": "Watame"})
        obj.vod_title = "Hololive GTA"
        obj.base_url = obj.cfg.cat_base_url
        obj.url = obj.base_url + "/chat/completions"
        obj.memory_lines = 0
        return obj

    def test_request_is_one_user_message_without_model_specific_controls(self):
        obj = self._unit()
        reply = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Don't run!"}}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.cat.urllib.request.urlopen",
                   return_value=reply) as urlopen:
            result = obj._chat("逃げるな", timeout=5.0)
        self.assertEqual(result, "Don't run!")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8003/v1/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "cat-translate-3.3b")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["repeat_penalty"], 1.05)
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual([part["role"] for part in body["messages"]], ["user"])
        self.assertEqual(
            body["messages"][0]["content"],
            "Translate the following Japanese text into English.\n\n逃げるな",
        )
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["stop"], ["</s>"])
        self.assertNotIn("top_p", body)
        self.assertNotIn("quality_review", str(body))

    def test_past_dialogue_never_enters_model_prompt(self):
        obj = self._unit()
        history = [("スバルが来た", "I saw a helicopter."),
                   ("車が故障した", "This is a casino.")]
        with patch.object(obj, "_chat", return_value="I'll go.") as call:
            result = obj.translate("行ってきます", history)
        self.assertEqual(result, "I'll go.")
        call.assert_called_once_with("行ってきます", timeout=obj.cfg.timeout_s)

    def test_no_inference_for_empty_or_nonverbal(self):
        obj = self._unit()
        with patch.object(obj, "_chat") as chat:
            self.assertEqual(obj.translate("", []), "[SKIP]")
        chat.assert_not_called()

    def test_one_call_only_even_for_suspect_translation(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="Let's run!") as chat:
            obj.translate("逃げるな", [])
        chat.assert_called_once()

    def test_sarashina_template_renders_exact_official_single_turn(self):
        from jinja2 import Template
        template = Path(__file__).resolve().parents[1] / (
            "src/auto_translate/translate/cat_sarashina.jinja"
        )
        source = build_cat_prompt("なるほど")
        rendered = Template(template.read_text(encoding="utf-8")).render(
            messages=[{"role": "user", "content": source}],
            eos_token="</s>",
            add_generation_prompt=True,
        )
        self.assertEqual(rendered, "<|user|>" + source + "</s><|assistant|>")

    def test_short_general_phrases_are_not_given_to_generating_model(self):
        obj = self._unit()
        with patch.object(obj, "_chat") as request:
            self.assertEqual(obj.translate("なるほど", []), "I see.")
            self.assertEqual(obj.translate("あかんか", []), "So that won't work?")
            self.assertEqual(obj.translate("なるほど。", []), "I see.")
        request.assert_not_called()

    def test_meta_reply_and_runaway_punctuation_are_rejected(self):
        self.assertEqual(
            cat_output_problem("なるほど", "I don't see the 10-character Japanese phrase to translate."),
            "assistant_meta_reply",
        )
        self.assertEqual(
            cat_output_problem("あかんか", "I'm ready to help translate, but please provide text."),
            "assistant_meta_reply",
        )
        self.assertEqual(
            cat_output_problem("服をセーブする", "Save it. " + ". " * 50),
            "punctuation_loop",
        )
        self.assertIsNone(cat_output_problem(
            "じゃあ行こう", "Okay, let's go."
        ))
        self.assertIsNone(cat_output_problem(
            "翻訳してほしい", "I would like you to translate it."
        ))

    def test_invalid_completion_retries_once_with_same_japanese(self):
        obj = self._unit()
        with patch.object(obj, "_chat", side_effect=[
            "I don't see the 10-character Japanese phrase.",
            "Oh, is that so?",
        ]) as call:
            self.assertEqual(obj.translate("そうなんだ?", []), "Oh, is that so?")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].args, ("そうなんだ?",))
        self.assertEqual(call.call_args_list[1].args, ("そうなんだ?",))
        self.assertTrue(call.call_args_list[1].kwargs["retry"])
        self.assertLessEqual(call.call_args_list[1].kwargs["timeout"], 4.0)

    def test_two_invalid_completions_drop_the_subtitle(self):
        obj = self._unit()
        with patch.object(obj, "_chat", side_effect=[
            "Save. " + ". " * 30,
            "I don't see the Japanese text to translate.",
        ]) as call:
            self.assertEqual(obj.translate("セーブしてね", []), "[SKIP]")
        self.assertEqual(call.call_count, 2)

    def test_valid_short_translation_still_uses_only_one_inference(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="Where's Watame?") as call:
            self.assertEqual(obj.translate("わためどこ", []), "Where's Watame?")
        call.assert_called_once_with("わためどこ", timeout=obj.cfg.timeout_s)

    def test_retry_sampler_is_bounded(self):
        obj = self._unit()
        reply = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Oh, is that so?"}}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.cat.urllib.request.urlopen",
                   return_value=reply) as urlopen:
            obj._chat("そうなんだ?", timeout=3.0, retry=True)
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["temperature"], 0.35)
        self.assertEqual(body["repeat_penalty"], 1.08)
        self.assertLessEqual(body["max_tokens"], 96)
        self.assertEqual(len(body["messages"]), 1)

    def test_thinking_and_translation_header_cleanup(self):
        self.assertEqual(
            _clean("<think>internal</think>English: Original song!"),
            "Original song!",
        )
        self.assertEqual(_clean("No, don't run!"), "No, don't run!")
        self.assertEqual(_clean("Well!</s>"), "Well!")
        self.assertEqual(_clean("Wait 30 seconds. </s> extra output"), "Wait 30 seconds.")
        self.assertEqual(_clean("English: Hello there.</s>"), "Hello there.")
        self.assertEqual(_clean("</s>"), "")

    def test_eos_artifact_is_removed_before_subtitle_display(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="Good morning!</s>"):
            self.assertEqual(obj.translate("おはようございます", []), "Good morning!")

    def test_eos_only_is_suppressed_after_one_bounded_retry(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="</s>") as chat:
            self.assertEqual(obj.translate("おはようございます", []), "[SKIP]")
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(chat.call_args_list[1].kwargs["retry"])
        self.assertLessEqual(chat.call_args_list[1].kwargs["timeout"], 4.0)

    def test_missing_output_is_suppressed_after_one_bounded_retry(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="") as chat:
            self.assertEqual(obj.translate("行ってきます", []), "[SKIP]")
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(chat.call_args_list[1].kwargs["retry"])

    def test_empty_first_response_can_recover_on_retry(self):
        obj = self._unit()
        with patch.object(obj, "_chat", side_effect=["", "I'm heading out."]) as chat:
            self.assertEqual(obj.translate("行ってきます", []), "I'm heading out.")
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(chat.call_args_list[1].kwargs["retry"])

    def test_server_selects_exact_nonbeta_3b_file_and_safe_port(self):
        obj = self._unit()
        obj.process = None
        obj._job = None
        obj._server_log = None
        obj._registered_exit = False
        process = Mock()
        process.poll.return_value = None
        process.pid = 4321
        with patch("auto_translate.translate.cat._model_ids",
                   side_effect=[None, ["cat-translate-3.3b"]]), patch(
                   "auto_translate.translate.cat.shutil.which",
                   return_value="llama-server.exe",
        ), patch(
            "auto_translate.translate.cat.subprocess.Popen",
            return_value=process,
        ) as popen, patch(
            "auto_translate.translate.cat._create_kill_job",
            return_value=None,
        ), patch("auto_translate.translate.cat.time.sleep"):
            obj._start_server()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("-hf") + 1],
                             "Sundowner123/cat-translate-3.3b-gguf")
            self.assertEqual(command[command.index("-hff") + 1],
                             "cat-translate-3.3b-Q4_K_M.gguf")
            self.assertEqual(command[command.index("--port") + 1], "8003")
            self.assertIn("--jinja", command)
            self.assertIn("--chat-template-file", command)
            chat_template = Path(command[command.index("--chat-template-file") + 1])
            self.assertTrue(chat_template.is_file())
            template_text = chat_template.read_text(encoding="utf-8")
            self.assertIn("<|user|>", template_text)
            self.assertIn("<|assistant|>", template_text)
            self.assertIn("eos_token", template_text)
            self.assertNotIn("--reasoning", command)
            obj.close()
        process.terminate.assert_called_once()

    def test_existing_server_not_adopted_or_terminated(self):
        obj = self._unit()
        with patch("auto_translate.translate.cat._model_ids",
                   return_value=["old-model"]), patch(
                   "auto_translate.translate.cat.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "already occupies"):
                obj._start_server()
        popen.assert_not_called()

    def test_external_server_urls_are_rejected(self):
        obj = self._unit()
        obj.base_url = "https://remote.example.com/v1"
        with patch("auto_translate.translate.cat._model_ids", return_value=None), patch(
            "auto_translate.translate.cat.shutil.which", return_value="llama-server.exe"
        ):
            with self.assertRaisesRegex(ValueError, "local"):
                obj._start_server()

    def test_hololive_names_still_pass_through_shared_fixes(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="Watame is here."):
            self.assertEqual(obj.translate("わためが来た", []), "Watame is here.")


if __name__ == "__main__":
    unittest.main()
