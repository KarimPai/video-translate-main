"""Offline Qwen3-ASR speech.cpp worker tests; no GPU or download required."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr import create_asr
from auto_translate.asr.qwen_asr import (
    MODEL_BYTES, MODEL_FILENAME, MODEL_REPO, MODEL_SHA256,
    QwenASR, resolve_qwen_asr_model,
)
from auto_translate.config import AsrConfig


def worker_stream(*messages):
    proc = Mock()
    proc.poll.return_value = None
    proc.stdin = io.StringIO()
    proc.stdout = io.StringIO("".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in (
            {"type": "ready", "version": "0.10.1"},
            *messages,
        )
    ))
    return proc


class QwenASRTests(unittest.TestCase):
    def test_selected_proper_noun_finetune_is_speechcpp_specific(self):
        self.assertEqual(MODEL_REPO, "kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF")
        self.assertEqual(MODEL_FILENAME, "Qwen3-ASR-1.7B-JA-Q8_0.gguf")
        self.assertEqual(MODEL_BYTES, 2_176_109_280)
        self.assertEqual(len(MODEL_SHA256), 64)

    def test_local_model_checksum_and_size_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qwen3-asr.gguf"
            path.write_bytes(b"gguf-test-weights")
            cfg = AsrConfig(model_path=str(path))
            data = path.read_bytes()
            with patch("auto_translate.asr.qwen_asr.MODEL_BYTES", len(data)), patch(
                "auto_translate.asr.qwen_asr.MODEL_SHA256",
                hashlib.sha256(data).hexdigest(),
            ):
                self.assertEqual(resolve_qwen_asr_model(cfg), path)
            with patch("auto_translate.asr.qwen_asr.MODEL_BYTES", len(data)), patch(
                "auto_translate.asr.qwen_asr.MODEL_SHA256", "0" * 64,
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    resolve_qwen_asr_model(cfg)

    def test_hf_model_download_uses_expected_single_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.gguf"
            path.write_bytes(b"verified weights")
            from huggingface_hub import hf_hub_download
            with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as fetch, patch(
                "auto_translate.asr.qwen_asr.MODEL_BYTES", path.stat().st_size,
            ), patch(
                "auto_translate.asr.qwen_asr.MODEL_SHA256",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ):
                self.assertEqual(resolve_qwen_asr_model(AsrConfig()), path)
            fetch.assert_called_once_with(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    def test_worker_forces_japanese_on_vulkan_without_haas_keywords(self):
        proc = worker_stream({
            "type": "end", "id": "1", "text": "わため、大丈夫?",
            "stop": "complete", "languages": ["ja"],
        })
        with patch("auto_translate.asr.qwen_asr.resolve_qwen_asr_model",
                   return_value=Path("model.gguf")), patch(
            "auto_translate.asr.qwen_asr.shutil.which", return_value="speech.exe",
        ), patch("auto_translate.asr.qwen_asr.subprocess.Popen",
                 return_value=proc) as popen, patch(
            "auto_translate.asr.qwen_asr._create_kill_job", return_value=None,
        ):
            recognizer = create_asr(AsrConfig())
            self.assertIsInstance(recognizer, QwenASR)
            self.assertEqual(recognizer.transcribe(
                np.zeros(16000, dtype=np.float32),
            ), "わため、大丈夫?")
            commands = [json.loads(x) for x in proc.stdin.getvalue().splitlines()]
            args = popen.call_args.args[0]
            self.assertEqual(args[:3], ["speech.exe", "worker", "model.gguf"])
            self.assertEqual(args[args.index("--language") + 1], "ja")
            self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
            self.assertNotIn("--keywords", args)
            self.assertEqual([x["type"] for x in commands], ["chunk", "transcribe"])
            self.assertEqual(commands[1]["sample_rate"], 16000)
            self.assertNotIn("keywords", commands[1])
            self.assertEqual(
                len(base64.b64decode(commands[0]["pcm"])), 16000 * 2,
            )
            recognizer.close()
            recognizer.close()
        proc.terminate.assert_called_once()

    def test_optional_short_qwen_prompt_and_probe_still_japanese(self):
        proc = worker_stream({
            "type": "end", "id": "1", "text": "角巻わため",
        })
        with patch("auto_translate.asr.qwen_asr.resolve_qwen_asr_model",
                   return_value=Path("model.gguf")), patch(
            "auto_translate.asr.qwen_asr.shutil.which", return_value="speech.exe",
        ), patch("auto_translate.asr.qwen_asr.subprocess.Popen",
                 return_value=proc) as popen:
            recognizer = QwenASR(AsrConfig(prompt="角巻わため、風真いろは"))
            self.assertEqual(recognizer.transcribe(
                np.ones(8000, dtype=np.float32), probe=True,
            ), "角巻わため")
            recognizer.close()
        args = popen.call_args.args[0]
        self.assertEqual(args[args.index("--prompt") + 1], "角巻わため、風真いろは")

    def test_recognition_error_does_not_fall_back_to_other_model(self):
        proc = worker_stream({
            "type": "error", "id": "1", "error": {
                "code": "unsupported", "message": "model failure",
            },
        })
        with patch("auto_translate.asr.qwen_asr.resolve_qwen_asr_model",
                   return_value=Path("model.gguf")), patch(
            "auto_translate.asr.qwen_asr.shutil.which", return_value="speech.exe",
        ), patch("auto_translate.asr.qwen_asr.subprocess.Popen",
                 return_value=proc):
            recognizer = QwenASR(AsrConfig())
            with self.assertRaisesRegex(RuntimeError, "Qwen ASR recognition failed"):
                recognizer.transcribe(np.ones(2000, dtype=np.float32))
            recognizer.close()

    def test_failed_worker_startup_closes_owned_process(self):
        proc = worker_stream()
        proc.stdout = io.StringIO('{"type":"fatal","error":{"code":"model_file"}}\n')
        with patch("auto_translate.asr.qwen_asr.resolve_qwen_asr_model",
                   return_value=Path("model.gguf")), patch(
            "auto_translate.asr.qwen_asr.shutil.which", return_value="speech.exe",
        ), patch("auto_translate.asr.qwen_asr.subprocess.Popen",
                 return_value=proc):
            with self.assertRaisesRegex(RuntimeError, "worker failed to load"):
                QwenASR(AsrConfig())
        proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
