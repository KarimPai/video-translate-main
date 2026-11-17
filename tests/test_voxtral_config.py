"""Configuration validation independent of model weights and native runtime."""
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from auto_translate.config import AsrConfig, Config, GpuBudgetConfig, load_config


class VoxtralConfigurationTests(unittest.TestCase):
    def test_example_loads_and_retains_cpu_vad_and_hardware_playback(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.example.toml")
        self.assertTrue(cfg.asr.voxtral_model_path.endswith("q4_k.gguf"))
        self.assertTrue(cfg.asr.voxtral_server_path.endswith("audiocpp_server.exe"))
        self.assertEqual(cfg.vad.device, "cpu")
        self.assertEqual(cfg.mpv.hwdec, "auto-safe")
        self.assertEqual(cfg.asr.cuda_device, 0)
        self.assertGreaterEqual(cfg.gpu_budget.startup_headroom_gb, 0.75)

    def test_partial_override_and_relative_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[asr]\nvoxtral_model_path="other/model.gguf"\nthreads=4\n', encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg.asr.voxtral_model_path, "other/model.gguf")
        self.assertEqual(cfg.asr.threads, 4)
        self.assertEqual(cfg.asr.cuda_device, 0)
        self.assertEqual(cfg.translate.glossary, "glossary.toml")

    def test_native_service_fields_reject_invalid_values(self):
        for options in (
            {"voxtral_model_path": " "}, {"voxtral_server_path": ""},
            {"voxtral_model_path": None}, {"cuda_device": -1},
            {"cuda_device": True}, {"threads": 0}, {"threads": 1.5},
            {"startup_timeout_s": float("nan")}, {"response_timeout_s": float("inf")},
            {"live_response_timeout_s": 0}, {"response_timeout_s": True},
            {"blocklist": [""]}, {"blocklist": [1]},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    AsrConfig(**options)

    def test_missing_config_uses_voxtral_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(Path(directory) / "missing.toml")
        self.assertEqual(cfg, Config())
        self.assertEqual(cfg.asr.startup_timeout_s, 180)
        self.assertEqual(cfg.gpu_budget, GpuBudgetConfig())
        self.assertEqual({f.name for f in fields(cfg.asr)}, {
            "voxtral_model_path", "voxtral_server_path", "cuda_device", "threads",
            "blocklist", "response_timeout_s", "live_response_timeout_s", "startup_timeout_s",
            "rescue_retry", "rescue_min_vad_confidence", "rescue_min_duration_s",
        })


if __name__ == "__main__":
    unittest.main()
