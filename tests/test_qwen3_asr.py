import base64
import io
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np

from auto_translate.asr import asr_context
from auto_translate.asr.qwen3 import AsrError, Qwen3ASR, is_japanese, parse_output, wav_bytes
from auto_translate.config import AsrConfig
from auto_translate.glossary import Glossary


def response(text, reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": reason}]}


def fake_asr(*responses, cfg=None, context=""):
    obj = object.__new__(Qwen3ASR)
    obj.cfg = cfg or AsrConfig()
    obj.context = context
    obj.last_raw_transcript = ""
    obj.last_rejection = obj.last_language = None
    obj._state_lock = threading.Lock()
    obj._inference_lock = threading.Lock()
    obj._generation = obj._failures = obj._restarts = 0
    obj._closed = False
    obj.server = Mock(alive=Mock(return_value=True))
    obj._request = Mock(side_effect=responses)
    return obj


class QwenProtocolTests(unittest.TestCase):
    def test_language_and_transcript_parsing(self):
        self.assertEqual(parse_output("language Japanese<asr_text>こんにちは。"), ("Japanese", "こんにちは。"))
        self.assertEqual(parse_output("language None<asr_text>"), (None, ""))
        self.assertEqual(parse_output("language English<asr_text>Hello."), ("English", "Hello."))
        self.assertEqual(parse_output("<asr_text>はい"), (None, "はい"))

    def test_language_gate(self):
        self.assertTrue(is_japanese("Japanese", "はい"))
        self.assertTrue(is_japanese("Chinese", "ありがとう"))
        self.assertTrue(is_japanese(None, "猫"))
        self.assertFalse(is_japanese("Chinese", "你好"))
        self.assertFalse(is_japanese("English", "ありがとう"))
        self.assertFalse(is_japanese(None, "hello"))

    def test_mono_wav_and_bounded_quiet_gain(self):
        for level, expected in ((0, 0), (.0001, .0001), (.002, .008), (.03, .03)):
            with self.subTest(level=level):
                with wave.open(io.BytesIO(wav_bytes(np.full(16000, level))), "rb") as wav:
                    self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
                    pcm = np.frombuffer(wav.readframes(16000), dtype="<i2") / 32767
                    self.assertAlmostEqual(float(pcm.mean()), expected, delta=1 / 32767)
        with self.assertRaises(ValueError):
            wav_bytes(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            wav_bytes(np.array([float("nan")]))

    def test_context_uses_metadata_selected_names_not_raw_titles(self):
        glossary = Glossary({"兎田ぺこら": "Usada Pekora", "ホロライブ": "Hololive"}, nicknames={"ボタン": "Botan"})
        text = asr_context(AsrConfig(context="マイクラ", max_context_chars=50), glossary, "兎田ぺこら配信")
        self.assertTrue(text.startswith("マイクラ\n人名："))
        self.assertNotIn("配信", text)
        self.assertIn("兎田ぺこら", text)
        self.assertNotIn("ボタン", text)
        self.assertLessEqual(len(text), 50)
        self.assertEqual(asr_context(AsrConfig(max_context_chars=0), glossary), "")

    def test_audio_request_contains_no_prefill_and_greedy_sampling(self):
        obj = fake_asr(context="ぺこら 配信")
        del obj._request
        obj.server.chat.return_value = response("language Japanese<asr_text>こんにちは")
        obj._request(b"RIFF", 4, timeout=15)
        body = obj.server.chat.call_args.args[0]
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["temperature"], 0)
        self.assertFalse(body["stream"])
        part = body["messages"][1]["content"][0]["input_audio"]
        self.assertEqual(base64.b64decode(part["data"]), b"RIFF")
        self.assertEqual(part["format"], "wav")
        self.assertEqual(obj.server.chat.call_args.kwargs["timeout"], 15)

    def test_constructor_uses_model_projector_and_owned_server(self):
        server = Mock()
        with patch("auto_translate.asr.qwen3.require_gguf", side_effect=[Path("model.gguf"), Path("mmproj.gguf")]), patch(
                "auto_translate.asr.qwen3.resolve_executable", return_value=Path("llama-server.exe")), patch(
                "auto_translate.asr.qwen3.LlamaServer", return_value=server) as launch:
            obj = Qwen3ASR(AsrConfig(), "ぺこら")
        args = launch.call_args.args[2]
        self.assertEqual(args[args.index("--mmproj") + 1], "mmproj.gguf")
        self.assertEqual(args[args.index("--device") + 1], "CUDA0")
        server.start.assert_called_once()
        self.assertEqual(server.chat.call_count, 1)
        obj.close()
        server.close.assert_called_once()


class QwenRecognitionTests(unittest.TestCase):
    pcm = np.full(32000, .03, dtype=np.float32)

    def test_accepts_japanese_and_records_raw_language(self):
        obj = fake_asr(response("language Japanese<asr_text>ちょっと待って！"))
        self.assertEqual(obj.transcribe(self.pcm), "ちょっと待って！")
        self.assertEqual(obj.last_language, "Japanese")
        self.assertIn("<asr_text>", obj.last_raw_transcript)
        self.assertIsNone(obj.last_rejection)

    def test_rejected_utterances_never_produce_transcripts(self):
        cases = [("language English<asr_text>Hello!", "language:English", "stop"),
                 ("language None<asr_text>", "no_speech", "stop"),
                 ("language Japanese<asr_text>ちょっと待って！", "truncated", "length"),
                 ("language Japanese<asr_text>ご視聴ありがとうございました", "phantom_outro", "stop")]
        for raw, expected, reason in cases:
            with self.subTest(raw=raw):
                obj = fake_asr(response(raw, reason))
                self.assertEqual(obj.transcribe(self.pcm), "")
                self.assertEqual(obj.last_rejection, expected)

    def test_context_echo_blocklist_and_probe(self):
        text = "兎田ぺこらと宝鐘マリンとさくらみこ"
        obj = fake_asr(response("language Japanese<asr_text>" + text), context=text)
        self.assertEqual(obj.transcribe(self.pcm), "")
        self.assertEqual(obj.last_rejection, "context_echo")
        obj = fake_asr(response("language Japanese<asr_text>これは禁止です"), cfg=AsrConfig(blocklist=["禁止"]))
        self.assertEqual(obj.transcribe(self.pcm), "")
        self.assertEqual(obj.last_rejection, "blocklist")
        obj = fake_asr(response("language Chinese<asr_text>ありがとう"))
        self.assertEqual(obj.transcribe(self.pcm, probe=True), "")
        self.assertEqual(obj.last_rejection, "probe_unconfirmed")

    def test_cancel_during_request_discards_completion(self):
        obj = fake_asr()
        def complete(*args, **kwargs):
            obj.cancel_pending()
            return response("language Japanese<asr_text>古い発話です")
        obj._request.side_effect = complete
        self.assertEqual(obj.transcribe(self.pcm), "")
        self.assertEqual(obj.last_raw_transcript, "")

    def test_closed_asr_does_not_restart_or_submit_requests(self):
        obj = fake_asr()
        obj.close()
        self.assertEqual(obj.transcribe(self.pcm), "")
        obj._request.assert_not_called()
        obj.server.start.assert_not_called()

    def test_three_consecutive_transport_failures_are_fatal(self):
        obj = fake_asr(*[httpx.ReadTimeout("timed out")] * 3)
        self.assertEqual(obj.transcribe(self.pcm), "")
        self.assertEqual(obj.transcribe(self.pcm), "")
        with self.assertRaises(AsrError):
            obj.transcribe(self.pcm)


if __name__ == "__main__":
    unittest.main()
