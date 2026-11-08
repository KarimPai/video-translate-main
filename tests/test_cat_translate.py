"""CAT-Translate-3.3B official-prompt and managed-server regression checks."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.cat import CatTranslator, _clean, build_cat_prompt


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
        self.assertEqual(body["max_tokens"], 192)
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

    def test_eos_only_is_not_shown_as_translation(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="</s>"):
            self.assertEqual(obj.translate("おはようございます", []), "[translation failed]")

    def test_missing_output_is_not_displayed_as_valid_english(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value=""):
            self.assertEqual(obj.translate("行ってきます", []), "[translation failed]")

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
