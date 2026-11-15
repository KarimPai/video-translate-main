import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_translate.__main__ import main
from auto_translate.config import load_config


class SharedGpuPlaybackTests(unittest.TestCase):
    def test_launcher_profile_applies_to_existing_config_before_source_start(self):
        launcher = Path("Start translate.bat").read_text(encoding="utf-8")
        self.assertIn("--mpv --shared-gpu-playback", launcher)
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                cfg = load_config(Path("config.example.toml"))
                original = (cfg.asr.haas_device, cfg.reconciler.gpu_layers,
                            cfg.ingest.vod_max_audio_lead_s)
                argv = ["auto-translate", "https://example.org/video", "--mpv"]
                if enabled:
                    argv.append("--shared-gpu-playback")
                with (patch.object(sys, "argv", argv),
                      patch("auto_translate.windows_job.bind_process_tree"),
                      patch("auto_translate.cuda_dlls.setup"),
                      patch("auto_translate.config.load_config", return_value=cfg),
                      patch("auto_translate.ingest.resolve_source", side_effect=StopIteration),
                      patch("sys.stdout"), patch("sys.stderr"),
                      self.assertRaises(StopIteration)):
                    main()
                self.assertEqual((cfg.asr.haas_device, cfg.reconciler.gpu_layers,
                                  cfg.ingest.vod_max_audio_lead_s),
                                 ("Vulkan0", 99, 30) if enabled else original)
                self.assertFalse(cfg.asr.haas_decode_on_cpu)
                self.assertTrue(cfg.dual_asr.enabled)
                self.assertEqual(cfg.dual_asr.secondary_device, "cuda")
