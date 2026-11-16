"""Offscreen transcript review flows; inference models and media are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QT_AVAILABLE = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(QT_AVAILABLE, "optional PySide6 frontend unavailable")
class GuiReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from auto_translate.gui import VideoTranslateWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        (self.root / "logs").mkdir()
        (self.root / "config.example.toml").write_text("[mpv]\n", encoding="utf8")
        self.window = VideoTranslateWindow(self.root)

    def tearDown(self):
        self.window._diagnostic_timer.stop()
        self.window._review_refresh_timer.stop()
        self.window.process = None
        self.window._cleanup_refinement()
        self.window.close()
        self.app.processEvents()
        self.folder.cleanup()

    def add(self, start=1.0, end=2.0, ja="こんにちは", en="Hello", **extra):
        self.window._on_player_event(dict(event="subtitle", start_s=start, end_s=end,
                                          ja=ja, en=en, generation=0, **extra))
        self.window._refresh_transcript()

    def select_first(self):
        self.window.transcript_table.selectRow(0)
        self.app.processEvents()

    def test_correction_survives_automatic_replay_and_retains_revision(self):
        self.add()
        self.select_first()
        self.window.en_editor.setPlainText("Hi there")
        self.window._save_review_correction()
        self.add(en="Overwritten by backend?")
        cue = next(iter(self.window.review.cues.values()))
        self.assertEqual(cue.en, "Hi there")
        self.assertEqual(cue.revisions, [("こんにちは", "Hello")])
        self.assertEqual(self.window.review.snapshot()[0].en, "Hi there")

    def test_search_review_filter_and_diagnostic_details(self):
        self.add()
        self.add(3, 4, "待って", "Wait")
        self.window.review.add_diagnostic(dict(start_s=3, end_s=4, flags=["disagreement"],
                                               haas="待って", whisper_turbo="まって", chosen_by="uncertain"))
        self.window.needs_review_filter.setChecked(True)
        self.window._refresh_transcript()
        self.assertEqual(self.window.transcript_model.rowCount(), 1)
        self.select_first()
        self.assertIn("Haas: 待って", self.window.review_info.text())
        self.assertIn("Whisper: まって", self.window.review_info.text())
        self.window.transcript_search.setText("Hello")
        self.window._refresh_transcript()
        self.assertEqual(self.window.transcript_model.rowCount(), 0)

    def test_previews_stay_out_of_review_and_clear_at_final_and_generation(self):
        w = self.window
        w._on_player_event(dict(event="preview", ja="仮", generation=0))
        self.assertEqual(len(w.review.cues), 0)
        self.assertIn("Draft Japanese", w.draft_label.text())
        self.add()
        self.assertEqual(w.draft_label.text(), "")
        w._on_player_event(dict(event="preview", ja="次", generation=1))
        w._on_player_event(dict(event="preview", ja="古い", generation=0))
        self.assertIn("次", w.draft_label.text())
        w._on_player_event(dict(event="preview", clear=True, generation=2))
        self.assertEqual(w.draft_label.text(), "")
        self.assertEqual(len(w.review.cues), 1)

    def test_export_uses_selected_language_and_format(self):
        self.add()
        self.window.file_language.setCurrentIndex(self.window.file_language.findData("japanese"))
        path = str(self.root / "reviewed")
        with patch("auto_translate.gui.QFileDialog.getSaveFileName", return_value=(path, "WebVTT (*.vtt)")), \
             patch("auto_translate.subtitle_files.export_subtitles", return_value=1) as export:
            self.window._export_review_subtitles()
        self.assertEqual(export.call_args.args[0], path + ".vtt")
        self.assertEqual(export.call_args.kwargs["language"], "japanese")
        self.assertEqual(export.call_args.args[1][0].start_s, 1)

    def test_import_corrections_records_edits_without_changing_timing(self):
        self.add()
        self.select_first()
        corrected = self.window.review.snapshot()[0]
        corrected.en = "Corrected English"
        revision = SimpleNamespace(revised=corrected)
        with patch("auto_translate.gui.QFileDialog.getOpenFileName", return_value=("edited.srt", "")), \
             patch("auto_translate.subtitle_files.import_corrections", return_value=[revision]):
            self.window._import_review_corrections()
        cue = next(iter(self.window.review.cues.values()))
        self.assertEqual((cue.start_s, cue.end_s), (1, 2))
        self.assertEqual(cue.en, "Corrected English")
        self.assertEqual(len(cue.revisions), 1)
        self.assertEqual(self.window.en_editor.toPlainText(), "Corrected English")

    def test_replay_seeks_selected_cue_and_requires_running_vod(self):
        self.add()
        self.select_first()
        self.window._request_seek = Mock()
        self.window._replay_review_cue()
        self.window._request_seek.assert_not_called()
        self.window._ready = self.window._control_ready = True
        self.window._replay_review_cue()
        self.window._request_seek.assert_called_once_with(1.0)

    def test_start_uses_only_new_diagnostics_and_preview_is_opt_in(self):
        path = self.root / "logs" / "asr-comparison.jsonl"
        path.write_text(json.dumps(dict(session_id="old", start_s=1, end_s=2, flags=["old"])) + "\n")
        w = self.window
        w.url_input.setText("https://youtu.be/example")
        self.assertFalse(w.preview_checkbox.isChecked())
        w.preview_checkbox.setChecked(True)
        args = ["-m", "auto_translate"]
        with patch("auto_translate.gui.build_backend_command", return_value=(sys.executable, args)), \
             patch.object(w, "_listen_for_control", return_value=[]), \
             patch.object(w, "_new_process", return_value=Mock()):
            w.start_translation()
        self.assertIn("--preview", args)
        with path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(session_id="new", start_s=3, end_s=4,
                                         stage="asr_fallback", reason="timeout")) + "\n")
        w._poll_diagnostics()
        self.assertEqual(set(w.review.diagnostics), {(3000, 4000)})
        self.assertEqual(w.review.diagnostics[(3000, 4000)]["fallback_reason"], "timeout")

    def test_report_runs_in_background_and_updates_gui(self):
        log_path = self.root / "logs" / "asr-comparison.jsonl"
        log_path.write_text("{}\n")
        result = dict(html_path=self.root / "report.html", json_path=self.root / "report.json")
        with patch("auto_translate.gui.QFileDialog.getExistingDirectory", return_value=str(self.root)), \
             patch("auto_translate.quality_report.generate_report", return_value=result) as report:
            self.window._generate_performance_report()
            deadline = time.monotonic() + 3
            while self.window._review_job is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
        self.assertIsNone(self.window._review_job)
        report.assert_called_once_with(log_path, self.root)
        self.assertIn("report.html", self.window.status.text())

    def test_refinement_uses_separate_backend_and_keeps_review_snapshot(self):
        self.add()
        destination = str(self.root / "aligned.srt")
        process = Mock()
        with patch("auto_translate.gui.QFileDialog.getOpenFileName", return_value=("matching.wav", "")), \
             patch("auto_translate.gui.QFileDialog.getSaveFileName", return_value=(destination, "")), \
             patch("auto_translate.subtitle_files.export_subtitles", return_value=1), \
             patch("auto_translate.gui.backend_python", return_value=Path(sys.executable)), \
             patch("auto_translate.gui.QProcess", return_value=process):
            self.window._refine_review_timings()
        arguments = process.setArguments.call_args.args[0]
        self.assertIn("--refine-subtitles", arguments)
        self.assertEqual(arguments[-1], "matching.wav")
        self.assertEqual(arguments[arguments.index("--export") + 1], destination)
        self.assertEqual(len(self.window.review.cues), 1)
        process.start.assert_called_once_with()

    def test_live_timing_refinement_is_rejected_before_file_dialog(self):
        self.add()
        self.window._review_live = True
        with patch("auto_translate.gui.QFileDialog.getOpenFileName") as dialog:
            self.window._refine_review_timings()
        dialog.assert_not_called()
