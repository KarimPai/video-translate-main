"""Feature settings and truthful, cached diagnostic rendering without models."""
import importlib.util
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_translate.gui_runner import build_backend_command, diagnostic_rows


class DiagnosticRenderingTests(unittest.TestCase):
    def test_missing_and_invalid_gpu_observations_remain_unknown(self):
        for event in ({}, {"gpu": None}, {"gpu": {"used_gb": float("nan"),
                                                   "total_gb": True, "free_gb": -1}}):
            with self.subTest(event=event):
                rows = diagnostic_rows(event)
                self.assertEqual(rows["gpu"], "VRAM unknown used / unknown total; unknown available")
                self.assertIn("decoder unknown", rows["playback"])

    def test_metrics_and_worker_status_are_displayed(self):
        rows = diagnostic_rows({
            "gpu": {"used_gb": 5.25, "total_gb": 8, "free_gb": 2.75,
                    "utilization_pct": 70, "temperature_c": 65, "power_w": 180},
            "stages": {"asr_s": 1.2, "mt_s": .6},
            "counters": {"pending_subtitles": 3, "latency_s": 12.34, "dropped": 2, "expired": 1},
            "playback": {"resolution": "1280×720", "decoder": "h264", "hwdec": "d3d11va"},
            "workers": {"asr": "running", "mt": "owned"},
        })
        self.assertIn("70.0%", rows["activity"])
        self.assertIn("180.0 W", rows["activity"])
        self.assertIn("ASR 1.20 s", rows["stages"])
        self.assertIn("Translation 0.60 s", rows["stages"])
        self.assertIn("expired 1", rows["subtitles"])
        self.assertIn("asr: running", rows["workers"])
        self.assertIn("mt: owned", rows["workers"])
        self.assertIn("d3d11va", rows["playback"])

    def test_feature_choices_override_config_with_explicit_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            interpreter = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            if os.name == "nt":
                interpreter.with_name("pythonw.exe").touch()
            _, args = build_backend_command(
                root, "https://youtu.be/abcd", 720, 42,
                context_lines=12,
                asr_model="models/custom-asr.gguf", mt_model="models/custom-mt.gguf",
                safe_playback=True, diagnostics=False,
            )
            for flag in ("--safe-playback", "--no-diagnostics"):
                self.assertIn(flag, args)
            self.assertEqual(args[args.index("--context-lines") + 1], "12")
            self.assertEqual(args[args.index("--asr-model") + 1], "models/custom-asr.gguf")
            self.assertEqual(args[args.index("--mt-model") + 1], "models/custom-mt.gguf")
            _, defaults = build_backend_command(root, "https://youtu.be/abcd", 720, 42)
            for flag in ("--no-safe-playback", "--diagnostics"):
                self.assertIn(flag, defaults)
            self.assertEqual(defaults[defaults.index("--context-lines") + 1], "6")
            self.assertNotIn("--asr-model", defaults)
            self.assertNotIn("--mt-model", defaults)

    def test_invalid_feature_values_are_rejected_before_launch(self):
        for kwargs in ({"context_lines": 21}, {"context_lines": True},
                       {"asr_model": ""}, {"mt_model": False},
                       {"safe_playback": "yes"}, {"diagnostics": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_backend_command(Path("."), "https://youtu.be/abcd", 720, 42, **kwargs)

    def test_owned_thread_and_subprocess_observations_are_readable(self):
        rows = diagnostic_rows({"workers": {"threads": {"ingest": True, "asr": False},
                                            "processes": [{"name": "llama-server.exe", "pid": 42, "running": True},
                                                          {"name": "ffmpeg.exe", "pid": 43, "running": False,
                                                           "exit_code": 1}]}})
        self.assertIn("ingest running", rows["workers"])
        self.assertIn("asr stopped", rows["workers"])
        self.assertIn("llama-server.exe (PID 42): running", rows["workers"])
        self.assertIn("ffmpeg.exe (PID 43): exited (1)", rows["workers"])


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "PySide6 optional GUI dependency unavailable")
class QtDiagnosticSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from auto_translate.gui import VideoTranslateWindow
        self.folder = tempfile.TemporaryDirectory()
        self.window = VideoTranslateWindow(Path(self.folder.name))

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        self.folder.cleanup()

    def test_feature_defaults_and_context_controls(self):
        window = self.window
        self.assertEqual(window.context_lines.value(), 6)
        self.assertEqual(window.context_lines.minimum(), 0)
        self.assertEqual(window.context_lines.maximum(), 20)
        self.assertTrue(window.context_lines.isEnabled())
        self.assertFalse(window.safe_playback_checkbox.isChecked())
        self.assertTrue(window.diagnostics_checkbox.isChecked())
        self.assertFalse(window.include_windows_events.isChecked())
        self.assertIn("Qwen3-ASR-1.7B-Q8_0.gguf", window.asr_model.text())
        self.assertIn("HY-MT1.5-7B-Q4_K_M.gguf", window.mt_model.text())
        for value in (0, 20):
            window.context_lines.setValue(value)
            self.assertEqual(window.context_lines.value(), value)
            self.assertTrue(window.context_lines.isEnabled())

    def test_start_button_launches_selected_models_and_locks_settings(self):
        window = self.window
        (window.root / "logs").mkdir()
        window.url_input.setText("https://youtu.be/abcd")
        window.context_lines.setValue(7)
        window.safe_playback_checkbox.setChecked(True)
        proc = Mock()
        with patch("auto_translate.gui.prepare_project"), patch(
                "auto_translate.gui.build_backend_command", return_value=("pythonw.exe", [])) as command, patch(
                "auto_translate.gui.subprocess_environment", return_value={}), patch(
                "auto_translate.gui.bind_process_tree"), patch("auto_translate.gui.WindowsJob"), patch(
                "auto_translate.gui.session_launcher", side_effect=lambda command, args: (command, args)), patch.object(
                window, "_listen_for_control", return_value=[]), patch.object(window, "_new_process", return_value=proc):
            window.start_button.click()
        try:
            proc.start.assert_called_once()
            self.assertIs(window.process, proc)
            self.assertFalse(window.feature_settings.isEnabled())
            self.assertFalse(window.start_button.isEnabled())
            self.assertTrue(window.stop_button.isEnabled())
            self.assertEqual(command.call_args.kwargs["context_lines"], 7)
            self.assertTrue(command.call_args.kwargs["safe_playback"])
            self.assertEqual(command.call_args.kwargs["asr_model"], window.asr_model.text())
            self.assertEqual(command.call_args.kwargs["mt_model"], window.mt_model.text())
        finally:
            window.process = None

    def test_invalid_url_does_not_start_backend(self):
        window = self.window
        window.url_input.setText("not a YouTube URL")
        with patch("auto_translate.gui.QMessageBox.warning") as warning:
            window.start_button.click()
        warning.assert_called_once()
        self.assertIsNone(window.process)
        self.assertTrue(window.start_button.isEnabled())

    def test_diagnostics_events_preserve_player_state_and_clear_for_new_session(self):
        window = self.window
        window.seek_slider.setRange(0, 100)
        window.seek_slider.setValue(42)
        window._on_player_event({"event": "diagnostics", "gpu": {"used_gb": 4},
                                 "stages": {"asr_s": 1.2}})
        self.assertEqual(window.seek_slider.value(), 42)
        self.assertIn("used", window.diagnostic_labels["gpu"].text())
        self.assertIn("ASR 1.20 s", window.diagnostic_labels["stages"].text())
        window._update_diagnostics({})
        self.assertIn("unknown used", window.diagnostic_labels["gpu"].text())
        self.assertEqual(window.diagnostics_observation_status.text(), "Waiting for backend observations.")

    def test_configured_preferences_are_loaded(self):
        root = Path(self.folder.name)
        (root / "config.toml").write_text(
            '[asr]\nmodel_path = "models/custom-asr.gguf"\n'
            '[translate]\nmodel_path = "models/custom-mt.gguf"\ncontext_lines = 12\n'
            "[safe_playback]\nenabled = true\n"
            "[diagnostics]\nenabled = false\n", encoding="utf8")
        self.window._load_feature_settings()
        self.assertEqual(self.window.context_lines.value(), 12)
        self.assertTrue(self.window.context_lines.isEnabled())
        self.assertEqual(self.window.asr_model.text(), "models/custom-asr.gguf")
        self.assertEqual(self.window.mt_model.text(), "models/custom-mt.gguf")
        self.assertTrue(self.window.safe_playback_checkbox.isChecked())
        self.assertFalse(self.window.diagnostics_checkbox.isChecked())

    def test_export_uses_background_thread_and_opt_in_metadata(self):
        destination = Path(self.folder.name) / "diagnostics.zip"
        with patch("auto_translate.gui.QFileDialog.getSaveFileName", return_value=(str(destination), "")), patch(
                "auto_translate.runtime_diagnostics.export_diagnostics", return_value=destination) as export:
            self.window.export_diagnostics()
            self.assertFalse(self.window.export_diagnostics_button.isEnabled())
            self.window._export_thread.join(2)
            self.assertFalse(self.window._export_thread.is_alive())
            self.app.processEvents()
            self.assertTrue(self.window.export_diagnostics_button.isEnabled())
            export.assert_called_once_with(destination, log_dir=Path(self.folder.name) / "logs",
                                           include_windows_events=False)
            self.assertIn("saved to", self.window.diagnostics_export_status.text())

    def test_closure_during_export_does_not_update_closed_window(self):
        release = threading.Event()
        entered = threading.Event()

        def export(*args, **kwargs):
            entered.set()
            release.wait(2)
            return Path(self.folder.name) / "diagnostics.zip"

        with patch("auto_translate.gui.QFileDialog.getSaveFileName", return_value=("diagnostics.zip", "")), patch(
                "auto_translate.runtime_diagnostics.export_diagnostics", side_effect=export):
            self.window.export_diagnostics()
            self.assertTrue(entered.wait(1))
            self.window.close()
            self.assertTrue(self.window._closing)
            release.set()
            self.window._export_thread.join(2)
            self.app.processEvents()
            self.assertNotIn("saved to", self.window.diagnostics_export_status.text())


if __name__ == "__main__":
    unittest.main()
