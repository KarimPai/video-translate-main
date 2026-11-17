import json
from pathlib import Path
import tempfile
import unittest
import wave

from auto_translate.config import AsrConfig, TranslateConfig, load_config
from scripts.benchmark import edit_distance, load_cases, normalized, read_audio, summarize


class BackendConfigTests(unittest.TestCase):
    def test_example_uses_new_model_schema(self):
        cfg = load_config(Path("config.example.toml"))
        self.assertEqual(cfg.asr.model_path, AsrConfig().model_path)
        self.assertEqual(cfg.asr.mmproj_path, AsrConfig().mmproj_path)
        self.assertEqual(cfg.translate.model_path, TranslateConfig().model_path)
        self.assertEqual(cfg.translate.gpu_layers, -1)
        self.assertEqual(cfg.translate.context_lines, 3)
        self.assertFalse(cfg.language_filter.enabled)

    def test_asr_boundary_validation(self):
        for kwargs in ({"model_path": ""}, {"mmproj_path": " "}, {"cuda_device": -1},
                       {"threads": True}, {"context": []}, {"context_size": 100},
                       {"max_context_chars": 8001}, {"blocklist": [""]},
                       {"response_timeout_s": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AsrConfig(**kwargs)

    def test_translation_boundary_validation(self):
        for kwargs in ({"model_path": ""}, {"gpu_layers": -2}, {"context_lines": 21},
                       {"context_lines": True}, {"context_size": 100}, {"temperature": float("nan")},
                       {"max_tokens": 0}, {"timeout_s": 0}):
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


if __name__ == "__main__":
    unittest.main()
