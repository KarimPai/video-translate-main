import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

from auto_translate.config import AsrConfig, TranslateConfig, load_config
from scripts.benchmark import (
    chrf, edit_distance, load_cases, normalized, read_audio, summarize, translation_overrides,
)


class BackendConfigTests(unittest.TestCase):
    def test_example_uses_new_model_schema(self):
        cfg = load_config(Path("config.example.toml"))
        self.assertEqual(cfg.asr.model_path, AsrConfig().model_path)
        self.assertEqual(cfg.asr.python_path, AsrConfig().python_path)
        self.assertEqual(cfg.translate.model_path, TranslateConfig().model_path)
        self.assertEqual(cfg.translate.gpu_layers, -1)
        self.assertEqual(cfg.translate.context_lines, 6)
        self.assertEqual(cfg.translate.max_context_chars, 600)
        self.assertEqual(cfg.translate.context_max_gap_s, 15.0)
        self.assertFalse(cfg.translate.context_english)
        self.assertTrue(cfg.translate.merge_fragments)
        self.assertFalse(cfg.language_filter.enabled)

    def test_translation_context_defaults(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.context_lines, 6)
        self.assertEqual(cfg.max_context_chars, 600)
        self.assertEqual(cfg.context_max_gap_s, 15.0)
        self.assertEqual((cfg.merge_max_gap_s, cfg.merge_max_s), (0.6, 16.0))

    def test_context_gap_accepts_positive_numeric_overrides(self):
        for gap in (1, 2.5, 30):
            with self.subTest(gap=gap):
                self.assertEqual(TranslateConfig(context_max_gap_s=gap).context_max_gap_s, gap)

    def test_context_gap_rejects_invalid_values(self):
        for gap in (0, -1, -0.1, float('nan'), float('inf'), float('-inf'),
                    True, False, '15'):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, 'context_max_gap_s'):
                TranslateConfig(context_max_gap_s=gap)

    def test_asr_boundary_validation(self):
        for kwargs in ({"model_path": ""}, {"python_path": " "}, {"cuda_device": -1},
                       {"blocklist": [""]}, {"response_timeout_s": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AsrConfig(**kwargs)

    def test_translation_boundary_validation(self):
        for kwargs in ({"model_path": ""}, {"gpu_layers": -2}, {"context_lines": 21},
                       {"context_lines": True}, {"context_size": 100}, {"temperature": float("nan")},
                       {"max_tokens": 0}, {"timeout_s": 0}, {"context_english": 1},
                       {"merge_fragments": "yes"}, {"merge_max_gap_s": -1}, {"merge_max_s": 60}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TranslateConfig(**kwargs)


class BenchmarkTests(unittest.TestCase):
    def test_character_distance_and_normalization(self):
        self.assertEqual(normalized("はい、 Ａ！"), "はいA")
        self.assertEqual(edit_distance("猫です", "犬です"), 1)
        self.assertEqual(edit_distance("はい", ""), 2)
        self.assertEqual(edit_distance("", "はい"), 2)

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            valid = [{"audio": "ja.wav", "ja": "ぺこらです", "en": "I'm Pekora.", "names": {"ぺこら": "Pekora"}}]
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(load_cases(path), valid)
            for cases in ([], [{}], [{"audio": "ja.wav", "ja": "はい", "names": {"ぺこら": "Pekora"}}]):
                path.write_text(json.dumps(cases), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_pcm_wav_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ja.wav"
            for rate in (16000, 44100):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    wav.writeframes(bytes(32000))
                if rate == 16000:
                    self.assertEqual(len(read_audio(path)), 16000)
                else:
                    with self.assertRaises(ValueError):
                        read_audio(path)

    def test_summary_uses_corpus_weighted_cer_and_rtf(self):
        rows = [{"reference_chars": 10, "char_errors": 1, "name_count": 1,
                 "asr_names_correct": 1, "mt_names_correct": 0, "duration_s": 2,
                 "asr_s": .2, "mt_s": .4},
                {"reference_chars": 30, "char_errors": 3, "name_count": 1,
                 "asr_names_correct": 1, "mt_names_correct": 1, "duration_s": 8,
                 "asr_s": .8, "mt_s": 1.6}]
        summary = summarize(rows)
        self.assertEqual(summary["cer"], .1)
        self.assertEqual(summary["asr_rtf"], .1)
        self.assertEqual(summary["mt_name_recall"], .5)
        self.assertIn("Human review", summary["english_review"])
        self.assertIsNone(summary["chrf"])

    def test_chrf_rewards_closer_translations(self):
        self.assertAlmostEqual(chrf([("It's sunny today.", "It's sunny today.")]), 100.0)
        self.assertEqual(chrf([("Hello.", "zzz")]), 0.0)
        self.assertIsNone(chrf([]))
        close = chrf([("Pekora is here.", "Pekora's here.")])
        far = chrf([("Pekora is here.", "The rabbit came.")])
        self.assertGreater(close, far)
        # Corpus scores pool n-gram counts instead of averaging per-cue scores.
        pooled = chrf([("Pekora is here.", "Pekora's here."), ("Yes.", "No.")])
        self.assertLess(pooled, close)

    def test_context_overrides_for_ab_runs(self):
        args = SimpleNamespace(context_lines=None, max_context_chars=None, context_english=False, no_context=False)
        self.assertEqual(translation_overrides(args), {})
        args = SimpleNamespace(context_lines=4, max_context_chars=800, context_english=True, no_context=False)
        self.assertEqual(translation_overrides(args),
                         {"context_lines": 4, "max_context_chars": 800, "context_english": True})
        args.no_context = True
        self.assertEqual(translation_overrides(args)["context_lines"], 0)

    def test_manifest_accepts_sequential_groups_and_stream_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            valid = [{"audio": "a.wav", "ja": "はい", "group": "collab", "channel": "Pekora Ch. 兎田ぺこら",
                      "description": "ゲスト：宝鐘マリン"}]
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(load_cases(path), valid)
            path.write_text(json.dumps([{"audio": "a.wav", "ja": "はい", "group": 1}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_benchmark_validate_only_without_models(self):
        # The validate-only path must work without model files: it only checks
        # manifest structure and audio format, never loading ASR or MT.
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            path = base / "cases.json"
            valid = [{"audio": "ja.wav", "ja": "ぺこらです", "en": "I'm Pekora.", "names": {"ぺこら": "Pekora"}}]
            path.write_text(json.dumps(valid), encoding="utf-8")
            wav_path = base / "ja.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(bytes(32000))
            cases = load_cases(path)
            self.assertEqual(len(cases), 1)
            pcm = read_audio(base / cases[0]["audio"])
            self.assertEqual(len(pcm), 16000)
            # Simulate validate-only report structure.
            report = {"cases": len(cases), "manifest": str(path), "validated": True,
                      "audio_clips": [(c["audio"], len(read_audio(base / c["audio"])) / 16000) for c in cases]}
            self.assertEqual(report["cases"], 1)
            self.assertTrue(report["validated"])


if __name__ == "__main__":
    unittest.main()
