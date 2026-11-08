"""Qwen request construction, safeguards and clean-slate tests (no GPU required)."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.translate.qwen import (
    QwenTranslator, _clean, _model_ids, _remove_unmentioned_full_names,
    _remove_unspoken_insult,
)


class FakeGlossary:
    def find(self, jp):
        return [("ぺこちゃん", "Peko-chan")] if "ぺこちゃん" in jp else []


def translator(**kwargs):
    settings = {
        "qwen_model": "qwen3.5-4b", "context_lines": 3,
        "timeout_s": 10.0, "max_tokens": 128,
        "quality_recheck_enabled": True, "quality_recheck_timeout_s": 3.0,
        "quality_recheck_min_interval_s": 0.0,
    }
    settings.update(kwargs)
    instance = object.__new__(QwenTranslator)
    instance.cfg = SimpleNamespace(**settings)
    instance.url = "http://127.0.0.1:8001/v1/chat/completions"
    instance.glossary = FakeGlossary()
    instance.vod_title = "ARK stream with Pekora"
    instance.memory_lines = 0
    return instance


class QwenTranslatorTests(unittest.TestCase):
    def test_clean_discards_leaked_thinking(self):
        self.assertEqual(_clean("<think>speculation</think>\nEN: Peko-chan!"), "Peko-chan!")
        self.assertEqual(_clean("EN: \"Hello!\""), "Hello!")
        self.assertEqual(_clean(""), "")

    def test_current_glossary_only_and_no_past_english_leakage(self):
        instance = translator()
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Peko-chan, hello!"}}]
        }).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response) as request:
            en = instance.translate(
                "ぺこちゃん、こんにちは",
                [("猫ちゃんはどこ", "A hallucinated Usada Pekora is here.")],
            )
        self.assertEqual(en, "Peko-chan, hello!")
        data = json.loads(request.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(data["model"], "qwen3.5-4b")
        self.assertEqual(data["messages"][0]["role"], "system")
        self.assertIn("Japanese", data["messages"][0]["content"])
        self.assertEqual(data["messages"][1]["role"], "user")
        self.assertIn("ぺこちゃん => Peko-chan", data["messages"][1]["content"])
        self.assertIn("猫ちゃんはどこ", data["messages"][1]["content"])
        self.assertNotIn("hallucinated", data["messages"][1]["content"])
        self.assertEqual(data["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("source_lang", data["chat_template_kwargs"])
        self.assertNotIn("stop", data)

    def test_scene_memory_in_qwen_request_beyond_last_five_lines(self):
        instance = translator()
        instance.cfg.context_lines = 5
        instance.observe_scene("馬券を買う")
        instance.observe_scene("オッズは7倍")
        history = [("なるほど", "Mistaken earlier translation")] * 12
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "What should we do next?"}},
        ]}).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=response) as call:
            self.assertEqual(instance._chat("次どうする?", history, 5),
                             "What should we do next?")
        body = json.loads(call.call_args.args[0].data.decode())
        prompt = body["messages"][1]["content"]
        self.assertIn("Scene memory (prior recognized Japanese", prompt)
        self.assertIn("Japanese horse racing and betting", prompt)
        self.assertIn("betting tickets", prompt)
        self.assertNotIn("Mistaken earlier translation", prompt)
        self.assertIn("CURRENT JAPANESE (translate only this line)", prompt)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(len(body["messages"]), 2)

    def test_scene_memory_can_be_disabled_and_reset_without_model_call(self):
        instance = translator()
        with patch.object(QwenTranslator, "_chat", return_value="Hello.") as call:
            instance.translate("馬券買った", [])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(instance._scene_topic(), "horse_racing")
        instance.reset_scene_context()
        self.assertIsNone(instance._scene_topic())
        instance.cfg.scene_context_enabled = False
        instance.observe_scene("オッズを見た")
        self.assertIsNone(instance._scene_topic())

    def test_gist_style_is_default_and_faithful_style_is_selectable(self):
        instance = translator()
        response = json.dumps({
            "choices": [{"message": {"content": "A Giganotosaurus attacked the base."}}]
        }).encode("utf-8")

        def prompt_for(style=None):
            instance.cfg.subtitle_style = style or "gist"
            with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                       return_value=io.BytesIO(response)) as urlopen:
                result = instance._chat("ギガノトが拠点を襲った", [], timeout=5)
            self.assertEqual(result, "A Giganotosaurus attacked the base.")
            return json.loads(urlopen.call_args.args[0].data)["messages"][0]["content"]

        gist = prompt_for("gist")
        faithful = prompt_for("faithful")
        self.assertIn("concise gist", gist)
        self.assertIn("Preserve every meaningful negation", gist)
        self.assertIn("faithful natural translation", faithful)
        self.assertNotIn("concise gist", faithful)

    def test_hololive_name_expansion_guard(self):
        self.assertEqual(
            _remove_unmentioned_full_names("他の人", "Ichijou Ririka"), ""
        )
        self.assertEqual(
            _remove_unmentioned_full_names("一条莉々華が来た", "Ichijou Ririka is here"),
            "Ichijou Ririka is here",
        )

    def test_ambiguous_cat_nickname_is_not_expanded_to_hololive_name(self):
        self.assertEqual(
            _remove_unmentioned_full_names("あ、聞こえます?猫ちゃん",
                                           "Can you hear me? Nekomata-chan"),
            "Can you hear me? Neko-chan",
        )
        self.assertEqual(
            _remove_unmentioned_full_names("猫又おかゆが来た", "Nekomata-chan came"),
            "Nekomata-chan came",
        )
        self.assertEqual(
            _remove_unmentioned_full_names("ぺこちゃん", "Peko-chan"),
            "Peko-chan",
        )

    def test_unspoken_trailing_insult_removed(self):
        self.assertEqual(
            _remove_unspoken_insult(
                "もう罠から出さなくていいんだよなきっと",
                "You don't have to get out of the trap, you idiot."
            ), "You don't have to get out of the trap."
        )
        self.assertEqual(
            _remove_unspoken_insult("バカだな", "You're an idiot, you idiot."),
            "You're an idiot, you idiot.",
        )

    def test_isolated_exclamation_needs_no_model_call(self):
        instance = translator()
        with patch.object(QwenTranslator, "_chat") as chat:
            self.assertEqual(instance.translate("すごい", []), "Wow!")
        chat.assert_not_called()

    def test_selective_recheck_is_bounded(self):
        instance = translator()
        with patch.object(
            QwenTranslator, "_chat", side_effect=["It's sunny.", "It's not sunny."],
        ) as chat:
            self.assertEqual(instance.translate("晴れてないわ", []), "It's not sunny.")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args.kwargs["review_hint"], "possible_missing_negation")
        self.assertEqual(chat.call_args.kwargs["timeout"], 3.0)

    def test_bad_skip_is_not_silently_output(self):
        instance = translator(quality_recheck_enabled=False)
        with patch.object(QwenTranslator, "_chat", return_value="[SKIP]"):
            self.assertEqual(instance.translate("警察が来た", []), "[translation failed]")


    def test_owned_llama_server_is_terminated_and_job_released(self):
        instance = translator()
        process = Mock()
        process.poll.return_value = None
        process.pid = 1234
        instance.process = process
        instance._job = "fake-windows-job"
        instance._server_log = Mock()
        with patch("auto_translate.translate.qwen._close_job") as close_job, patch(
            "auto_translate.translate.qwen.atexit.unregister"
        ):
            instance.close()
            instance.close()  # cleanup is idempotent
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3)
        close_job.assert_called_once_with("fake-windows-job")
        self.assertIsNone(instance.process)

    def test_close_does_not_stop_external_server(self):
        instance = translator()
        instance.process = None
        instance._job = None
        instance._server_log = None
        with patch("auto_translate.translate.qwen._close_job") as close_job, patch(
            "auto_translate.translate.qwen.atexit.unregister"
        ):
            instance.close()
        close_job.assert_not_called()

    def test_start_server_rejects_unowned_existing_server(self):
        instance = translator()
        instance.base_url = "http://127.0.0.1:8001/v1"
        instance.process = None
        instance._job = None
        with patch("auto_translate.translate.qwen._model_ids", return_value=["qwen3.5-4b"]):
            with self.assertRaisesRegex(RuntimeError, "already listening"):
                instance._start_server()

    def test_qwen_server_models_are_checked_before_reuse(self):
        response = io.BytesIO(json.dumps({"data": [{"id": "qwen3.5-4b"}]}).encode())
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response):
            self.assertEqual(_model_ids("http://127.0.0.1:8001/v1"), ["qwen3.5-4b"])

    def test_carjacking_is_defined_not_kidnapping_in_current_cue(self):
        instance = translator()
        instance.vod_title = "Hololive GTA"
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "We're going to the carjacking scene."}},
        ]}).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response) as call:
            self.assertEqual(instance._chat("カージャックされた現場に行く", [], 5),
                             "We're going to the carjacking scene.")
        data = json.loads(call.call_args.args[0].data.decode())
        self.assertIn("カージャック => carjack / carjacking", data["messages"][1]["content"])
        self.assertIn("not kidnapping", data["messages"][1]["content"])

    def test_racing_context_guidance_preserves_horse_and_names(self):
        instance = translator()
        instance.vod_title = "Hololive horse racing"
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "The first horse is Millefeuille d'Or."}},
        ]}).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=response) as call:
            self.assertEqual(
                instance._chat("1番のミルフィーユドールは前走8着", [], 5),
                "The first horse is Millefeuille d'Or.",
            )
        data = json.loads(call.call_args.args[0].data.decode())
        self.assertIn("Event context: Japanese horse racing and betting.", data["messages"][1]["content"])
        self.assertIn("not 'Milky Way Doll'", data["messages"][1]["content"])

    def test_horse_wide_bet_and_number_sequence_are_not_straight_bet_or_money(self):
        instance = translator()
        instance.vod_title = "Hololive horse racing"
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "Just the key horse for the wide bet."}},
        ]}).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=response) as call:
            self.assertEqual(
                instance._chat("ワイドに軸だけ。", [], 5),
                "Just the key horse for the wide bet.",
            )
        data = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertIn("ワイド => wide bet", data["messages"][1]["content"])
        self.assertIn("NOT a straight win bet", data["messages"][1]["content"])
        self.assertIn("key or anchor horse", data["messages"][1]["content"])

        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "62468? How about 468?"}},
        ]}).encode("utf-8"))
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=response) as call:
            self.assertEqual(instance._chat("62468どうする?468。", [], 5),
                             "62468? How about 468?")
        data = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertIn("do not add thousands commas", data["messages"][0]["content"])
        self.assertIn("never turn 62468 into 62,468", data["messages"][1]["content"])

    def test_betting_payout_not_retail_refund(self):
        instance = translator()
        instance.vod_title = "Horse racing event"
        raw = json.dumps({"choices": [
            {"message": {"content": "Let's go get our refund."}},
        ]}).encode("utf-8")
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=io.BytesIO(raw)):
            result = instance.translate("払い戻し行こうか", [("馬券買ったの?", "Did you bet?")])
        self.assertEqual(result, "Let's go get our payout.")
        # A new video starts with new scene memory; never carry the previous
        # horse-racing state into unrelated retail dialogue.
        instance.reset_scene_context()
        instance.vod_title = "Shopping"
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=io.BytesIO(raw)):
            result = instance.translate("払い戻し行こうか", [])
        self.assertEqual(result, "Let's go get our refund.")
        self.assertIsNone(instance._scene_topic())

        # A fresh translator with a shopping title must behave identically,
        # even without an explicit reset or any previous source context.
        fresh = translator()
        fresh.vod_title = "Shopping"
        with patch("auto_translate.translate.qwen.urllib.request.urlopen",
                   return_value=io.BytesIO(raw)):
            self.assertEqual(
                fresh.translate("払い戻し行こうか", []),
                "Let's go get our refund.",
            )
        self.assertIsNone(fresh._scene_topic())

    def test_ark_glossary_in_qwen_prompt(self):
        instance = translator()
        response = io.BytesIO(json.dumps({"choices": [
            {"message": {"content": "Giganotosaurus!"}},
        ]}).encode())
        with patch("auto_translate.translate.qwen.urllib.request.urlopen", return_value=response) as call:
            self.assertEqual(instance._chat("ギガノト!", [], 5), "Giganotosaurus!")
        data = json.loads(call.call_args.args[0].data.decode())
        self.assertIn("ギガノト => Giganotosaurus", data["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
