"""Offline runtime contracts; these tests do not measure acoustic accuracy."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from auto_translate.asr.haas import HaasASR, _KeywordOptionsRejected, _advertised_options, _keyword_text
from auto_translate.asr.whisper_turbo import _decode, whisper_decode_options
from auto_translate.config import AsrConfig, DualAsrConfig, load_config
from pathlib import Path


class HaasVocabularyTests(unittest.TestCase):
    def worker(self, responses, *, supported=True, enabled=True):
        worker = object.__new__(HaasASR)
        worker.cfg = AsrConfig(haas_keywords_enabled=enabled)
        worker.keyword_support = supported
        worker.request_counter = 0
        worker.last_stop = None
        worker.process = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        iterator = iter(responses)
        worker._read_message = lambda deadline: next(iterator)
        return worker

    @staticmethod
    def requests(worker):
        return [request for line in worker.process.stdin.getvalue().splitlines()
                if (request := json.loads(line))["type"] == "transcribe"]

    def test_actual_model_capabilities_are_required(self):
        self.assertEqual(_advertised_options({"version": "0.11.0"}), set())
        status = {"model": {"options": [{"name": "keywords", "type": "string"},
                                         {"name": "keywords_guard", "type": "bool"}]}}
        self.assertEqual(_advertised_options(status), {"keywords", "keywords_guard"})
        self.assertEqual(_advertised_options({"model": None}), set())
        self.assertEqual(_advertised_options({"model": {"options": None}}), set())

    def test_old_model_omits_all_options_and_preserves_decode(self):
        worker = self.worker([{"type": "end", "id": "1", "text": "まつりちゃん"}], supported=False)
        self.assertEqual(worker.transcribe(np.ones(100), keywords=["まつり=マツリ"]), "まつりちゃん")
        self.assertEqual(self.requests(worker), [{"type": "transcribe", "id": "1", "sample_rate": 16000}])
        self.assertFalse(worker.last_keyword_diagnostics["active"])
        self.assertEqual(worker.last_keyword_diagnostics["fallback"], "unsupported")
        self.assertFalse(worker.last_keyword_diagnostics["attempted"])

    def test_explicit_opt_in_and_guard_required(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                worker = self.worker([{"type": "end", "id": "1", "text": "みこちゃん"}], enabled=enabled)
                worker.transcribe(np.ones(100), keywords=["みこ=ミコ"], keywords_alpha=1.5)
                request = self.requests(worker)[0]
                self.assertEqual("keywords" in request, enabled)
                if enabled:
                    self.assertIs(request["keywords_guard"], True)
                    self.assertEqual(request["keywords_alpha"], 1.5)
                    self.assertEqual(request["keywords"], "みこ=ミコ")

    def test_rejected_option_retries_original_pcm_without_options(self):
        worker = self.worker([
            {"type": "error", "id": "1", "error": {"code": "unsupported", "option": "keywords_guard"}},
            {"type": "end", "id": "2", "text": "お祭りに行く"},
        ])
        self.assertEqual(worker.transcribe(np.ones(100), keywords="まつり=マツリ"), "お祭りに行く")
        first, second = self.requests(worker)
        self.assertEqual((first["id"], second["id"]), ("1", "2"))
        self.assertNotIn("keywords", second)
        chunks = [json.loads(line) for line in worker.process.stdin.getvalue().splitlines()
                  if json.loads(line)["type"] == "chunk"]
        self.assertEqual(chunks[0]["pcm"], chunks[1]["pcm"])
        self.assertFalse(worker.keyword_support)
        self.assertEqual(worker.last_keyword_diagnostics["fallback"], "unsupported")
        self.assertTrue(worker.last_keyword_diagnostics["attempted"])

    def test_decode_error_does_not_trigger_unsupported_retry(self):
        worker = self.worker([{"type": "error", "id": "1", "error": {"code": "decode_failed"}}])
        with self.assertRaises(RuntimeError):
            worker.transcribe(np.ones(100), keywords="みこ=ミコ")
        self.assertEqual(len(self.requests(worker)), 1)

    def test_fallback_keeps_original_deadline_and_cannot_submit_after_expiry(self):
        worker = self.worker([])
        worker._transcribe_request = Mock(side_effect=[_KeywordOptionsRejected("unsupported"), "こんにちは"])
        self.assertEqual(worker.transcribe(np.ones(100), timeout=.5, keywords="みこ=ミコ"), "こんにちは")
        first, second = worker._transcribe_request.call_args_list
        self.assertEqual(first.args[1], second.args[1])
        self.assertEqual(second.args[2], {})
        worker = self.worker([])
        with self.assertRaises(TimeoutError):
            worker._transcribe_request("pcm", 0, {})
        self.assertEqual(self.requests(worker), [])

    def test_probe_never_boosts_names(self):
        worker = self.worker([{"type": "end", "id": "1", "text": "こんにちは"}])
        worker.transcribe(np.ones(100), probe=True, keywords="みこ=ミコ")
        self.assertNotIn("keywords", self.requests(worker)[0])

    def test_vocabulary_is_bounded_and_deduplicated(self):
        self.assertEqual(_keyword_text(["みこ=ミコ", "みこ=ミコ", "まつり=マツリ"]),
                         "みこ=ミコ\nまつり=マツリ")
        for invalid in (["みこ"] * 9, ["みこ\nラプ"], [123], ["a" * 65]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _keyword_text(invalid)


class WhisperDecodingOptionsTests(unittest.TestCase):
    @staticmethod
    def model():
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text="みこちが来た", avg_logprob=-.2,
                                                               no_speech_prob=.1)]), None)
        return model

    def test_exposed_controls_keep_existing_defaults(self):
        options = whisper_decode_options(DualAsrConfig())
        self.assertEqual(options["temperature"], 0)
        self.assertEqual(options["suppress_tokens"], [-1])
        self.assertFalse(options["word_timestamps"])
        self.assertEqual(DualAsrConfig().secondary_prompt_mode, "none")
        self.assertFalse(AsrConfig().haas_keywords_enabled)
        self.assertEqual(DualAsrConfig(secondary_length_penalty=0).secondary_length_penalty, 0)

    def test_config_examples_load_with_exposed_defaults(self):
        for filename in ("config.example.toml", "config.dual-asr.example.toml"):
            with self.subTest(filename=filename):
                config = load_config(Path(filename))
                self.assertEqual(config.dual_asr.secondary_beam_size, 6)
                self.assertEqual(config.dual_asr.secondary_prompt_mode, "none")
                self.assertFalse(config.asr.haas_keywords_enabled)

    def test_supported_controls_and_mode_are_recorded(self):
        model = self.model()
        _, confidence = _decode(model, np.ones(100), 8, hotwords="みこち", with_metadata=True,
                                decode_options={"patience": 1.5, "max_new_tokens": 128})
        request = model.transcribe.call_args.kwargs
        self.assertEqual(request["hotwords"], "みこち")
        self.assertEqual(request["beam_size"], 8)
        self.assertEqual(request["patience"], 1.5)
        self.assertEqual(request["max_new_tokens"], 128)
        self.assertEqual(confidence["decoding"]["prompt_mode"], "hotwords")
        self.assertEqual(confidence["decoding"]["unsupported_options"], [])
        self.assertEqual(request["language"], "ja")
        self.assertFalse(request["condition_on_previous_text"])

    def test_old_runtime_drops_unsupported_controls_and_hint(self):
        calls = []

        class LegacyModel:
            def transcribe(self, samples, *, language, task, beam_size,
                           condition_on_previous_text, vad_filter, without_timestamps):
                calls.append((language, condition_on_previous_text))
                return iter([SimpleNamespace(text="こんにちは")]), None

        text, confidence = _decode(LegacyModel(), np.ones(100), 6, hotwords="みこ", with_metadata=True)
        self.assertEqual(text, "こんにちは")
        self.assertEqual(calls, [("ja", False)])
        self.assertEqual(confidence["decoding"]["prompt_mode"], "none")
        self.assertIn("hotwords", confidence["decoding"]["unsupported_options"])

    def test_hints_and_safe_contract_cannot_be_overridden(self):
        for options in ({"task": "translate"}, {"condition_on_previous_text": True},
                        {"prefix": "Sakura Miko"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _decode(self.model(), np.ones(100), 6, decode_options=options)
        with self.assertRaises(ValueError):
            _decode(self.model(), np.ones(100), 6, "みこ", hotwords="まつり")
        with self.assertRaises(ValueError):
            _decode(self.model(), np.ones(100), 6, hotwords="みこ" * 200)

    def test_config_rejects_invalid_controls(self):
        for options in ({"secondary_patience": float("nan")}, {"secondary_temperature": 2},
                        {"secondary_best_of": 0}, {"secondary_prompt_mode": "title"},
                        {"secondary_suppress_tokens": [True]}, {"secondary_max_new_tokens": 449}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DualAsrConfig(**options)
        for options in ({"haas_keywords_enabled": "yes"}, {"haas_keywords_alpha": 0},
                        {"haas_keywords_alpha": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AsrConfig(**options)


if __name__ == "__main__":
    unittest.main()
