"""PLaMo 3 Translate 8B Q3_K_M offline PFML3 and server regressions."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import (
    PlamoTranslator, _clean, _invalid_output, build_plamo_prompt,
)


class PlamoTranslationTests(unittest.TestCase):
    @staticmethod
    def _unit():
        obj = object.__new__(PlamoTranslator)
        obj.cfg = TranslateConfig()
        obj.glossary = Glossary({"わため": "Watame", "ころね": "Korone"})
        obj.vod_title = "Hololive"
        obj.base_url = obj.cfg.plamo_base_url
        obj.url = obj.base_url + "/completions"
        obj.memory_lines = 0
        return obj

    def test_official_pfml3_source_only_no_chat_wrapper(self):
        self.assertEqual(
            build_plamo_prompt(" 行ってきます "),
            "<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>行ってきます"
            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>",
        )

    def test_exact_matched_glossary_is_in_pfml3_not_entire_dictionary(self):
        obj = self._unit()
        prompt = build_plamo_prompt(
            "わためが来た", obj.glossary.find("わためが来た"),
        )
        self.assertIn("<|plamo:tag|>glossary<|plamo:index|>わため<|plamo:item|>Watame", prompt)
        self.assertNotIn("Korone", prompt)
        self.assertIn("<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>わためが来た", prompt)
        self.assertEqual(prompt.count("<|plamo:tag|>glossary"), 1)

    def test_pfml3_special_marker_injection_is_not_accepted_from_glossary(self):
        prompt = build_plamo_prompt("わため", [
            ("わため", "Watame"),
            ("fake", "<|plamo:tag|>output"),
            ("bad\nname", "broken"),
        ])
        self.assertEqual(prompt.count("<|plamo:index|>"), 1)
        self.assertNotIn("bad\nname", prompt)

    def test_completions_api_is_explicit_ja_to_en(self):
        obj = self._unit()
        response = io.BytesIO(json.dumps({
            "choices": [{"text": "Where is Watame?"}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   return_value=response) as urlopen:
            result = obj._complete("わためどこ", timeout=5)
        self.assertEqual(result, "Where is Watame?")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8005/v1/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "plamo-3-translate-8b")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 128)
        self.assertFalse(body["stream"])
        self.assertEqual(body["stop"], ["<|plamo:tag|>", "<|plamo:bos|>"])
        self.assertIn("<|plamo:val|>ja<|plamo:msg|>", body["prompt"])
        self.assertIn("<|plamo:val|>en<|plamo:msg|>", body["prompt"])
        self.assertNotIn("messages", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_verified_japanese_context_precedes_glossary_and_current_input(self):
        obj = self._unit()
        obj.set_reference_context("車が止まりました。")
        response = io.BytesIO(json.dumps({
            "choices": [{"text": "What should we do?"}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   return_value=response) as send:
            self.assertEqual(obj._complete("わためどうする?", timeout=5), "What should we do?")
        prompt = json.loads(send.call_args.args[0].data)["prompt"]
        context = "<|plamo:tag|>context<|plamo:msg|>車が止まりました。"
        glossary = "<|plamo:tag|>glossary<|plamo:index|>わため<|plamo:item|>Watame"
        current = "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>わためどうする?"
        self.assertIn(context, prompt)
        self.assertIn(glossary, prompt)
        self.assertIn(current, prompt)
        self.assertLess(prompt.index(context), prompt.index(glossary))
        self.assertLess(prompt.index(glossary), prompt.index(current))
        # The reference is one-shot and must not leak into unrelated speakers.
        self.assertNotIn("context", build_plamo_prompt("次のセリフ"))

    def test_context_rejects_pfml3_injection_and_multiline_text(self):
        obj = self._unit()
        for bad in ("<|plamo:tag|>output", "前のセリフ\\n別の人", ""):
            obj.set_reference_context(bad)
            self.assertIsNone(obj._reference_context)

    def test_reference_state_is_consumed_after_inference(self):
        obj = self._unit()
        obj.set_reference_context("前のセリフ")
        reply = io.BytesIO(json.dumps({
            "choices": [{"text": "Understood."}],
        }).encode("utf-8"))
        with patch("auto_translate.translate.plamo.urllib.request.urlopen",
                   return_value=reply):
            obj._complete("なるほど", timeout=5)
        self.assertIsNone(obj._reference_context)

    def test_previous_translations_are_not_in_prompt(self):
        obj = self._unit()
        history = [("前のセリフ", "I am a helicopter.")]
        with patch.object(obj, "_complete", return_value="I'll head out.") as run:
            self.assertEqual(obj.translate("行ってきます", history), "I'll head out.")
        run.assert_called_once_with("行ってきます", timeout=obj.cfg.timeout_s)

    def test_nonverbal_and_empty_audio_do_not_translate(self):
        obj = self._unit()
        with patch.object(obj, "_complete") as run:
            self.assertEqual(obj.translate("", []), "[SKIP]")
            self.assertEqual(obj.translate("ああああああ", []), "[SKIP]")
        run.assert_not_called()

    def test_invalid_completion_is_suppressed_without_extra_inference(self):
        obj = self._unit()
        with patch.object(obj, "_complete", return_value="I don't see the Japanese text.") as run:
            self.assertEqual(obj.translate("なるほど", []), "[SKIP]")
        run.assert_called_once()

    def test_valid_complete_sentence_is_returned(self):
        obj = self._unit()
        with patch.object(obj, "_complete", return_value="I see.") as run:
            self.assertEqual(obj.translate("なるほど", []), "I see.")
        run.assert_called_once()

    def test_watame_passes_through_common_repairs(self):
        obj = self._unit()
        with patch.object(obj, "_complete", return_value="Where is Watame?"):
            self.assertEqual(obj.translate("わためはどこ?", []), "Where is Watame?")

    def test_plamo_cleanup_and_repetition_detection(self):
        self.assertEqual(_clean("Hello!<|plamo:tag|>input"), "Hello!")
        self.assertEqual(_clean("English: Good morning."), "Good morning.")
        self.assertTrue(_invalid_output("なるほど", "Save." + " . " * 30))
        self.assertTrue(_invalid_output("なるほど", "I'm ready to translate."))
        self.assertFalse(_invalid_output("なるほど", "I see."))

    def test_server_uses_exact_q3_and_memory_flags(self):
        obj = self._unit()
        obj.process = None
        obj._job = None
        obj._server_log = None
        obj._registered_exit = False
        process = Mock()
        process.poll.return_value = None
        process.pid = 8005
        with patch("auto_translate.translate.plamo._model_ids",
                   side_effect=[None, ["plamo-3-translate-8b"]]), patch(
                   "auto_translate.translate.plamo.shutil.which",
                   return_value="llama-server.exe",
        ), patch(
                   "auto_translate.translate.plamo.subprocess.Popen",
                   return_value=process,
        ) as popen, patch(
                   "auto_translate.translate.plamo._create_kill_job",
                   return_value=None,
        ), patch("auto_translate.translate.plamo.time.sleep"):
            obj._start_server()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("-hf") + 1],
                             "pfnet/plamo-3-translate-8b-GGUF")
            self.assertEqual(command[command.index("-hff") + 1],
                             "plamo-3-translate-8b-Q3_K_M.gguf")
            self.assertEqual(command[command.index("-c") + 1], "1024")
            self.assertEqual(command[command.index("-np") + 1], "1")
            self.assertEqual(command[command.index("-ngl") + 1], "99")
            self.assertEqual(command[command.index("-fa") + 1], "on")
            self.assertEqual(command[command.index("-ctk") + 1], "q8_0")
            self.assertEqual(command[command.index("-ctv") + 1], "q8_0")
            self.assertEqual(command[command.index("--port") + 1], "8005")
            self.assertIn("--jinja", command)
            self.assertIn("--no-prefill-assistant", command)
            self.assertNotIn("--chat-template-file", command)
            obj.close()
        process.terminate.assert_called_once()

    def test_existing_server_cannot_be_reused_or_terminated(self):
        obj = self._unit()
        with patch("auto_translate.translate.plamo._model_ids",
                   return_value=["old-model"]), patch(
                   "auto_translate.translate.plamo.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "occupies"):
                obj._start_server()
        popen.assert_not_called()

    def test_external_base_url_is_rejected(self):
        obj = self._unit()
        obj.base_url = "https://remote.example.com/v1"
        with patch("auto_translate.translate.plamo._model_ids", return_value=None), patch(
            "auto_translate.translate.plamo.shutil.which", return_value="llama-server.exe",
        ):
            with self.assertRaisesRegex(ValueError, "local"):
                obj._start_server()


if __name__ == "__main__":
    unittest.main()
