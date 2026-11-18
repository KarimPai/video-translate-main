"""Regression tests: no artificial VRAM cap, CPU VAD, safe physical headroom."""
import unittest
from unittest.mock import patch

from auto_translate.config import GpuBudgetConfig, VadConfig
from auto_translate.gpu_budget import GpuBudget
from auto_translate.vad import Segmenter


class AdaptiveMemoryTests(unittest.TestCase):
    @patch("auto_translate.gpu_budget.gpu_memory_gb", return_value=(7.40, 8.59))
    def test_no_fixed_7_5gb_ceiling(self, sample):
        budget = GpuBudget(GpuBudgetConfig())
        self.assertEqual(budget.pressure_level, "normal")
        self.assertTrue(budget.admits(0.15))  # 7.40 GB in use is allowed.
        self.assertFalse(budget.admits(0.30))  # Reserve must fit plus headroom.

    @patch("auto_translate.gpu_budget.gpu_memory_gb", return_value=(8.19, 8.59))
    def test_low_physical_headroom_only(self, sample):
        budget = GpuBudget(GpuBudgetConfig())
        self.assertEqual(budget.pressure_level, "low")
        self.assertFalse(budget.admits(0.0))
        budget.free = 0.20
        self.assertEqual(budget.pressure_level, "critical")

    @patch("auto_translate.gpu_budget.gpu_memory_gb", return_value=(None, None))
    def test_missing_nvidia_smi_never_reports_safe_admission(self, sample):
        budget = GpuBudget(GpuBudgetConfig())
        self.assertEqual(budget.pressure_level, "unknown")
        self.assertFalse(budget.admits(0.0))

    def test_invalid_guard_parameters(self):
        for kwargs in ({"startup_headroom_gb": -0.1},
                       {"low_free_gb": 1.1},
                       {"emergency_free_gb": 0.80},
                       {"sample_interval_s": 0.01}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    GpuBudgetConfig(**kwargs)


class CpuVadTests(unittest.TestCase):
    def test_cpu_is_default_and_preserves_segmenter_parameters(self):
        cfg = VadConfig()
        self.assertEqual(cfg.device, "cpu")
        with patch("auto_translate.vad.StreamingFireRedVAD") as ctor:
            seg = Segmenter(cfg)
            ctor.assert_called_once_with(cfg.threshold, "cpu", cfg.model_path)
            self.assertEqual(seg.min_silence, 55)
            self.assertEqual(seg.pad, 40)  # 400 ms of audio at 10 ms/frame

    def test_configured_model_directory_reaches_loader(self):
        cfg = VadConfig(model_path="shared/Stream-VAD")
        with patch("auto_translate.vad.StreamingFireRedVAD") as ctor:
            Segmenter(cfg)
        ctor.assert_called_once_with(cfg.threshold, "cpu", "shared/Stream-VAD")

    def test_cuda_vad_is_rejected_to_preserve_asr_headroom(self):
        with self.assertRaisesRegex(ValueError, "must use cpu"):
            VadConfig(device="cuda")

    def test_reject_other_devices(self):
        with self.assertRaises(ValueError):
            VadConfig(device="gpu")


if __name__ == "__main__":
    unittest.main()
