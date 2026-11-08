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
    def test_official_insttrans_prompt_is_single_source_with_glossary(self):
        glossary = Glossary({"角巻わため": "Tsunomaki Watame", "兎田ぺこら": "Usada Pekora"})
        prompt = build_index_prompt(
            "角巻わためと行くよ", [("兎田ぺこらだよ", "I'm Pekora")],
            glossary, vod_title="Hololive GTA RP", context_lines=4,
        )
        self.assertIn("〖源文〗\n角巻わためと行くよ\n\n〖约束要求〗", prompt.replace("〖約束要求〗", "〖约束要求〗"))
        self.assertIn("角巻わため→Tsunomaki Watame", prompt)
        self.assertNotIn("兎田ぺこら→Usada Pekora", prompt)
        self.assertIn("先前的日语对话", prompt)
        self.assertIn("只输出译文", prompt)
        self.assertEqual(prompt.count("〖源文〗\n"), 1)
        self.assertNotIn("I'm Pekora", prompt)

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

    def test_short_casual_line_is_preserved_without_model_rewrite(self):
        obj = self._unit()
        with patch.object(obj, "_chat", return_value="I'm off!") as call:
            self.assertEqual(obj.translate("行ってきます", []), "I'm off!")
        call.assert_called_once()

    def test_unintelligible_repetitions_skip_model_call(self):
        obj = self._unit()
        with patch.object(obj, "_chat") as chat:
            self.assertEqual(obj.translate("", []), "[SKIP]")
        chat.assert_not_called()

    def test_clean_thinking_and_headers(self):
        self.assertEqual(_clean("<think>notes</think>Translation: Let's go!"), "Let's go!")

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
