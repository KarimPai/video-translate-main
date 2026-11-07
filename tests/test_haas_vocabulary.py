"""Haas keyword loading and speech.cpp worker request protocol tests.

No real model, FFmpeg or GPU is needed.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.asr.haas import HaasASR, load_haas_keywords
from auto_translate.config import AsrConfig, load_config


READY = {"type": "ready", "version": "0.10.1"}
def fake_worker(*answers):
    proc = Mock()
    proc.poll.return_value = None
    proc.stdin = io.StringIO()
    proc.stdout = io.StringIO("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in (READY, *answers)))
    return proc


class HaasVocabularyTests(unittest.TestCase):
    def test_keyword_boost_enabled_by_default(self):
        self.assertTrue(AsrConfig().haas_keywords_enabled)

    def test_default_vocabulary_contains_general_terms_but_no_gta_profile(self):
        file = Path(__file__).resolve().parents[1] / "haas_vocabulary.toml"
        words = load_haas_keywords(str(file)).splitlines()
        self.assertEqual(len(words), 29)
        self.assertIn("音乃瀬奏=おとのせかなで", words)
        self.assertIn("リスポーン=りすぽーん", words)
        self.assertNotIn("インパウンド=いんぱうんど", words)

    def test_duplicate_keywords_ignored_and_missing_file_disables_boost(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "terms.toml"
            path.write_text('keywords = ["わため=わため", "わため=わため"]\n', encoding="utf-8")
            self.assertEqual(load_haas_keywords(str(path)), "わため=わため")
            self.assertEqual(load_haas_keywords(str(path.with_name("missing.toml"))), "")

    def test_invalid_keyword_definition_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "terms.toml"
            path.write_text('keywords = ["="]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid keyword"):
                load_haas_keywords(str(path))

    def test_worker_sends_keywords_as_newline_string_with_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            vocab = Path(folder) / "terms.toml"
            vocab.write_text('keywords = ["わため=わため", "ギミック=ぎみっく"]\n', encoding="utf-8")
            cfg = AsrConfig(haas_keywords_enabled=True, haas_keywords_file=str(vocab))
            proc = fake_worker({"type": "end", "id": "1", "text": "わため"})
            with patch("auto_translate.asr.haas.shutil.which", return_value="speech.exe"), patch(
                "auto_translate.asr.haas.subprocess.Popen", return_value=proc
            ):
                worker = HaasASR(cfg)
                result = worker.transcribe(np.zeros(160, dtype=np.float32))
            self.assertEqual(result, "わため")
            sent = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
            self.assertEqual(sent[0]["type"], "chunk")
            self.assertEqual(sent[1]["type"], "transcribe")
            self.assertEqual(sent[1]["keywords"], "わため=わため\nギミック=ぎみっく")
            self.assertEqual(sent[1]["keywords_alpha"], 2.0)
            self.assertIs(sent[1]["keywords_guard"], True)

    def test_probe_and_disabled_vocabulary_do_not_send_keywords(self):
        for enabled, probe in ((False, False), (True, True)):
            with self.subTest(enabled=enabled, probe=probe):
                cfg = AsrConfig(haas_keywords_enabled=enabled, haas_keywords_file=str(Path(__file__).resolve().parents[1] / "haas_vocabulary.toml"))
                proc = fake_worker({"type": "end", "id": "1", "text": "わため"})
                with patch("auto_translate.asr.haas.shutil.which", return_value="speech.exe"), patch(
                    "auto_translate.asr.haas.subprocess.Popen", return_value=proc
                ):
                    worker = HaasASR(cfg)
                    worker.transcribe(np.zeros(160, dtype=np.float32), probe=probe)
                sent = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
                self.assertNotIn("keywords", sent[1])

    def test_unsupported_keyword_setting_retries_without_keywords(self):
        cfg = AsrConfig(haas_keywords_enabled=True, haas_keywords_file=str(Path(__file__).resolve().parents[1] / "haas_vocabulary.toml"))
        proc = fake_worker(
            {"type": "error", "id": "1", "error": {"code": "unsupported", "option": "keywords", "message": "not supported"}},
            {"type": "end", "id": "2", "text": "わため"},
        )
        with patch("auto_translate.asr.haas.shutil.which", return_value="speech.exe"), patch(
            "auto_translate.asr.haas.subprocess.Popen", return_value=proc
        ):
            worker = HaasASR(cfg)
            self.assertEqual(worker.transcribe(np.zeros(160, dtype=np.float32)), "わため")
        sent = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
        self.assertIn("keywords", sent[1])
        self.assertNotIn("keywords", sent[3])
        self.assertEqual(worker.keywords, "")

    def test_local_config_can_disable_and_tune_keywords(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.toml"
            path.write_text(
                '[asr]\nhaas_keywords_enabled = false\nhaas_keywords_file = "custom.toml"\n'
                'haas_keywords_alpha = 1.5\nhaas_keywords_guard = false\n', encoding="utf-8"
            )
            cfg = load_config(path).asr
            self.assertFalse(cfg.haas_keywords_enabled)
            self.assertEqual(cfg.haas_keywords_file, "custom.toml")
            self.assertEqual(cfg.haas_keywords_alpha, 1.5)
            self.assertFalse(cfg.haas_keywords_guard)


if __name__ == "__main__":
    unittest.main()
