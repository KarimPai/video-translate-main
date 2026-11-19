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

    def _cursor_tick(self, point, now, *, active=True, buttons=None):
        from PySide6.QtCore import Qt
        with patch("auto_translate.gui.QCursor.pos", return_value=point), patch(
                "auto_translate.gui.time.monotonic", return_value=now), patch.object(
                self.window, "isActiveWindow", return_value=active), patch(
                "auto_translate.gui.QApplication.mouseButtons",
                return_value=buttons or Qt.MouseButton.NoButton):
            self.window._player_activity_tick()

    def test_idle_video_cursor_hides_and_movement_restores_in_both_modes(self):
        from PySide6.QtCore import QPoint, Qt
        window = self.window
        window.show()
        window._session_ready()
        self.app.processEvents()
        self.assertTrue(window._player_activity_timer.isActive())
        for fullscreen in (False, True):
            with self.subTest(fullscreen=fullscreen):
                if fullscreen:
                    window.toggle_fullscreen()
                    self.app.processEvents()
                point = window.video_surface.mapToGlobal(window.video_surface.rect().center())
                self._cursor_tick(point, 100)
                self._cursor_tick(point, 102.9)
                self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
                self._cursor_tick(point, 103.1)
                self.assertEqual(window.video_surface.cursor().shape(), Qt.CursorShape.BlankCursor)
                self.assertEqual(window.video_controls.isHidden(), fullscreen)
                self._cursor_tick(point + QPoint(10, 0), 105)
                self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
                self.assertFalse(window.video_controls.isHidden())

    def test_cursor_restores_over_controls_and_outside_video(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        window._session_ready()
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        controls = window.pause_button.mapToGlobal(window.pause_button.rect().center())
        self._cursor_tick(video, 100)
        self._cursor_tick(video, 104)
        self._cursor_tick(controls, 105)
        self._cursor_tick(controls, 110)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        window.toggle_fullscreen()
        self.app.processEvents()
        controls = window.pause_button.mapToGlobal(window.pause_button.rect().center())
        self._cursor_tick(controls, 111)
        self._cursor_tick(controls, 115)
        self.assertFalse(window.video_controls.isHidden())
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))

    def test_cursor_restores_on_focus_loss_and_tab_switch(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        window._session_ready()
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 100)
        self._cursor_tick(video, 104)
        self._cursor_tick(video, 105, active=False)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        self._cursor_tick(video, 106)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        self._cursor_tick(video, 110)
        self.assertEqual(window.video_surface.cursor().shape(), Qt.CursorShape.BlankCursor)
        window.tabs.setCurrentWidget(window.console_tab)
        self._cursor_tick(video, 111)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))

    def test_cursor_stays_visible_while_dragging_or_holding_mouse(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        window._session_ready()
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 100)
        self._cursor_tick(video, 104, buttons=Qt.MouseButton.LeftButton)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        window._seeking = True
        self._cursor_tick(video, 108)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))

    def test_dialog_popup_and_occluding_window_restore_cursor(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        window._session_ready()
        window.toggle_fullscreen()
        self.app.processEvents()
        for index, (method, result) in enumerate((("activeModalWidget", window),
                                                  ("activePopupWidget", window),
                                                  ("widgetAt", None))):
            with self.subTest(method=method):
                video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
                now = 100 + index * 20
                self._cursor_tick(video, now)
                self._cursor_tick(video, now + 4)
                self.assertEqual(window.video_surface.cursor().shape(), Qt.CursorShape.BlankCursor)
                with patch(f"auto_translate.gui.QApplication.{method}", return_value=result):
                    self._cursor_tick(video, now + 5)
                    self._cursor_tick(video, now + 10)
                self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
                self.assertFalse(window.video_controls.isHidden())

    def test_cursor_remains_visible_before_playback_and_resets_with_player(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        self.app.processEvents()
        self.assertFalse(window._player_activity_timer.isActive())
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 100)
        self._cursor_tick(video, 104)
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        window._session_ready()
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 110)
        self._cursor_tick(video, 114)
        self.assertEqual(window.video_surface.cursor().shape(), Qt.CursorShape.BlankCursor)
        window._reset_player_controls()
        self.assertFalse(window._player_activity_timer.isActive())
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))

    def test_mode_change_stop_and_close_restore_video_cursor(self):
        from PySide6.QtCore import Qt
        window = self.window
        window.show()
        window._session_ready()
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 100)
        self._cursor_tick(video, 104)
        window.toggle_fullscreen()
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        self.app.processEvents()
        video = window.video_surface.mapToGlobal(window.video_surface.rect().center())
        self._cursor_tick(video, 110)
        self._cursor_tick(video, 114)
        window.exit_fullscreen()
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        self.assertTrue(window._player_activity_timer.isActive())
        window.stop_translation()
        self.assertFalse(window._player_activity_timer.isActive())
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))
        window._reset_player_activity()
        window._player_activity_timer.start()
        window.close()
        self.assertFalse(window._player_activity_timer.isActive())
        self.assertFalse(window.video_surface.testAttribute(Qt.WidgetAttribute.WA_SetCursor))

    def test_feature_defaults_and_context_controls(self):
        window = self.window
        self.assertEqual(window.context_lines.value(), 6)
        self.assertEqual(window.context_lines.minimum(), 0)
        self.assertEqual(window.context_lines.maximum(), 20)
        self.assertTrue(window.context_lines.isEnabled())
        self.assertTrue(window.safe_playback_checkbox.isChecked())
        self.assertTrue(window.diagnostics_checkbox.isChecked())
        self.assertFalse(window.include_windows_events.isChecked())
        self.assertIn("haas-ja-v1.0", window.asr_model.text())
        self.assertIn("google_gemma-4-E4B-it-Q4_K_M.gguf", window.mt_model.text())
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
