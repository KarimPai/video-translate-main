"""Windows PySide6 frontend for the existing multi-process translation pipeline.

The GUI never imports or initializes CUDA/ASR models. QProcess runs the same
Python backend used by Start translate.bat and native MPV renders into a HWND.
"""
import codecs
import json
import logging
import os
import uuid
import secrets
import select
import socket
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel, QEvent, QModelIndex, QProcess, QProcessEnvironment,
    QThread, Qt, QTimer, Signal,
)
from PySide6.QtGui import QCloseEvent, QCursor, QFont, QTextCursor, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
    QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
    QSlider, QStyle, QStyleOptionSlider, QTableView, QTabWidget, QVBoxLayout, QWidget,
)

from auto_translate.gui_control import parse_event, GUI_SHUTDOWN_MS
from auto_translate.windows_job import WindowsJob, bind_process_tree
from auto_translate.gui_runner import (
    QUALITY_OPTIONS, READY_MARKER, backend_python, build_backend_command, discover_project_root,
    prepare_project, session_launcher, subprocess_environment, valid_youtube_url, validate_gui_launch,
)
from auto_translate.review import DiagnosticTail, TranscriptReview, cue_key
from auto_translate.gui_build import build_identity, verify_build

log = logging.getLogger(__name__)

_STYLE = """
QMainWindow, QWidget { background: #141820; color: #e8ecf3; font-size: 13px; }
QLineEdit, QComboBox, QPlainTextEdit {
    background: #1e2530; color: #edf1f8; border: 1px solid #384251;
    border-radius: 7px; padding: 7px;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #79a9f5; }
QPushButton { background: #304056; color: white; border-radius: 7px; padding: 8px 14px; }
QPushButton:hover { background: #425977; }
QPushButton:disabled { background: #252b35; color: #6d7786; }
QPushButton#start { background: #376eb3; font-weight: bold; }
QPushButton#start:hover { background: #4784d1; }
QTabWidget::pane { border: 1px solid #303845; border-radius: 5px; }
QTabBar::tab { background: #1c232d; padding: 10px 24px; margin-right: 3px; }
QTabBar::tab:selected { background: #35475c; }
QLabel#muted { color: #a0aec0; }
QSlider::groove:horizontal { height: 5px; background: #384251; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #659be4; border-radius: 2px; }
QSlider::handle:horizontal { width: 13px; margin: -5px 0; background: #c7d9f5; border-radius: 6px; }
"""


class SeekSlider(QSlider):
    """VOD timeline: clicking anywhere jumps, dragging the handle seeks on release.

    QSlider's stock groove click only moves one page step and never emits
    sliderReleased for the handle. That made ordinary seekbar clicks a no-op.
    """
    seekRequested = Signal(float)

    def mousePressEvent(self, event):
        if (self.isEnabled() and
                event.button() == Qt.MouseButton.LeftButton):
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                option, QStyle.SubControl.SC_SliderHandle, self,
            )
            if not handle.contains(event.position().toPoint()):
                groove = self.style().subControlRect(
                    QStyle.ComplexControl.CC_Slider,
                    option, QStyle.SubControl.SC_SliderGroove, self,
                )
                travel = max(1, groove.width() - handle.width())
                x = int(event.position().x() - groove.left() - handle.width() / 2)
                target = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    max(0, min(travel, x)), travel, option.upsideDown,
                )
                self.setValue(target)
                self.seekRequested.emit(float(target))
                event.accept()
                return
        super().mousePressEvent(event)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        if (self.isEnabled() and not event.isAutoRepeat() and
                event.key() in (
                    Qt.Key.Key_Left, Qt.Key.Key_Right,
                    Qt.Key.Key_Up, Qt.Key.Key_Down,
                    Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
                    Qt.Key.Key_Home, Qt.Key.Key_End,
                )):
            self.seekRequested.emit(float(self.value()))


class TranscriptTableModel(QAbstractTableModel):
    HEADERS = ("Time", "Japanese", "English", "ASR / review", "Edits")

    def __init__(self, review, parent=None):
        super().__init__(parent)
        self.review = review
        self.rows = []

    def refresh(self, query="", needs_review=False):
        self.beginResetModel()
        self.rows = self.review.filtered(query, needs_review)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        _, cue = self.rows[index.row()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return json.dumps(cue.diagnostic, ensure_ascii=False, indent=2)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        diagnostic = cue.diagnostic
        reason = diagnostic.get("fallback_reason") or diagnostic.get("chosen_by", "")
        if cue.needs_review:
            reason = "Review · " + str(reason)
        values = (f"{cue.start_s:.2f}–{cue.end_s:.2f}", cue.ja, cue.en,
                  str(reason), str(len(cue.revisions)))
        return values[index.column()]


class ReviewJob(QThread):
    completed = Signal(object, str)

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.task = task

    def run(self):
        try:
            result, error = self.task(), ""
        except Exception as exc:
            result, error = None, str(exc)
        self.completed.emit(result, error)


class VideoTranslateWindow(QMainWindow):
    FULLSCREEN_CONTROLS_TIMEOUT_S = 3.0

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self._build_identity = build_identity(root)
        self._job = None
        self._session = None
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setSingleShot(True)
        self._shutdown_timer.timeout.connect(self._force_shutdown)
        self.process: QProcess | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_line = ""
        self._session_error = ""
        self._log_stream = None
        self._ready = False
        self._stopping = False
        self._closing = False
        self._duration_s = None
        self._live = False
        self._seeking = False
        self._pending_seek_until = 0.0
        self._pending_seek_target = None
        self._fullscreen = False
        self._was_maximized = False
        self._fullscreen_cursor_pos = None
        self._fullscreen_activity_at = 0.0
        self.review = TranscriptReview()
        self._editing_key = None
        self._diagnostic_tail = None
        self._diagnostic_session = None
        self._review_generation = None
        self._review_live = False
        self._review_job = None
        self._refine_process = None
        self._refine_temp = None
        self._refine_output = ""
        self.setWindowTitle("Video Translate - Dual ASR [" + self._build_identity["commit"][:8] + "]")
        self.setMinimumSize(750, 520)
        self.resize(1120, 790)
        self._control_server = None
        self._control_socket = None
        self._control_peers = {}
        self._control_ready = False
        self._control_lost = False
        self._control_timer = QTimer(self)
        self._control_timer.setInterval(50)
        self._control_timer.timeout.connect(self._poll_control)
        self._build_ui()
        self._review_refresh_timer = QTimer(self)
        self._review_refresh_timer.setSingleShot(True)
        self._review_refresh_timer.setInterval(200)
        self._review_refresh_timer.timeout.connect(self._refresh_transcript)
        self._diagnostic_timer = QTimer(self)
        self._diagnostic_timer.setInterval(1000)
        self._diagnostic_timer.timeout.connect(self._poll_diagnostics)
        # Qt mouse events never reach the host while embedded native MPV owns
        # its HWND. Poll the global pointer instead of relying on enter/move.
        self._fullscreen_controls_timer = QTimer(self)
        self._fullscreen_controls_timer.setInterval(150)
        self._fullscreen_controls_timer.timeout.connect(self._fullscreen_controls_tick)
        # Application-level shortcuts also work when focus is on the seekbar
        # or other Qt controls. Native MPV keys are forwarded through IPC.
        for key, callback in (("F11", self.toggle_fullscreen),
                              ("Escape", self.exit_fullscreen)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, widget, event):
        kind = event.type()
        if (kind in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and self._ready and not self._stopping
                and isinstance(widget, QWidget) and widget.window() is self
                and self.tabs.currentWidget() is self.player_tab
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
                and event.key() in (Qt.Key.Key_F, Qt.Key.Key_Space)):
            # Consume the release too: a focused QPushButton activates on Space.
            event.accept()
            if kind == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_F:
                    self.toggle_fullscreen()
                else:
                    self._send_player_command("pause")
            return True
        return super().eventFilter(widget, event)

    def _build_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        self.setCentralWidget(central)

        self.heading = QLabel("Video Translate")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)

        self.setup_panel = QWidget(central)
        setup = QVBoxLayout(self.setup_panel)
        setup.setContentsMargins(0, 0, 0, 0)
        setup.setSpacing(8)
        setup.addWidget(QLabel("YouTube live or VOD URL"))
        url_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.returnPressed.connect(self.start_translation)
        self.paste_button = QPushButton("Paste")
        self.paste_button.clicked.connect(self.paste_url)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.paste_button)
        setup.addLayout(url_row)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Quality"))
        self.quality = QComboBox()
        for value in QUALITY_OPTIONS:
            self.quality.addItem(f"{value}p", value)
        self.quality.setCurrentIndex(self.quality.findData(720))
        settings_row.addWidget(self.quality)
        self.vsr_checkbox = QCheckBox("RTX Video Super Resolution")
        self.vsr_checkbox.setChecked(False)
        self.vsr_checkbox.setToolTip(
            "Optional NVIDIA RTX Video Super Resolution. Requires an eligible "
            "source, compatible MPV/D3D11, and RTX VSR enabled in "
            "NVIDIA settings. Check activation in the NVIDIA app."
        )
        settings_row.addWidget(self.vsr_checkbox)
        self.preview_checkbox = QCheckBox("Live drafts")
        self.preview_checkbox.setToolTip("Best-effort Japanese drafts. Final subtitles take priority.")
        settings_row.addWidget(self.preview_checkbox)
        settings_row.addStretch(1)
        self.start_button = QPushButton("Start Translation")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self.start_translation)
        settings_row.addWidget(self.start_button)
        setup.addLayout(settings_row)
        layout.addWidget(self.setup_panel)

        self.tabs = QTabWidget()
        self.player_tab = QWidget()
        player_layout = QVBoxLayout(self.player_tab)
        player_layout.setContentsMargins(5, 5, 5, 5)
        self.video_surface = QFrame()
        self.video_surface.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.video_surface.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.video_surface.setStyleSheet("background-color: black; border: 0;")
        self.video_surface.setMinimumSize(320, 180)
        self.video_surface.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        player_layout.addWidget(self.video_surface, 1)

        self.video_controls = QWidget()
        playback_row = QHBoxLayout(self.video_controls)
        playback_row.setContentsMargins(3, 3, 3, 3)
        playback_row.setSpacing(8)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setToolTip("Play / pause (Space)")
        self.pause_button.clicked.connect(lambda: self._send_player_command("pause"))
        self.rewind_button = QPushButton("-10s")
        self.rewind_button.clicked.connect(lambda: self._send_player_command("seek_relative", -10))
        self.forward_button = QPushButton("+10s")
        self.forward_button.clicked.connect(lambda: self._send_player_command("seek_relative", 10))
        self.seek_slider = SeekSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._begin_seek)
        self.seek_slider.sliderReleased.connect(self._finish_seek)
        self.seek_slider.valueChanged.connect(self._seek_preview)
        self.seek_slider.seekRequested.connect(self._request_seek)
        self.time_label = QLabel("--:-- / --:--")
        self.time_label.setMinimumWidth(112)
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.setToolTip("Fullscreen player (F11, F or double-click MPV; Esc exits)")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        for widget in (self.pause_button, self.rewind_button, self.forward_button, self.seek_slider, self.fullscreen_button):
            widget.setEnabled(False)
        for widget in (self.pause_button, self.rewind_button, self.seek_slider, self.forward_button,
                       self.time_label, self.fullscreen_button):
            playback_row.addWidget(widget, 1 if widget is self.seek_slider else 0)
        player_layout.addWidget(self.video_controls)
        self.draft_label = QLabel("")
        self.draft_label.setTextFormat(Qt.TextFormat.PlainText)
        self.draft_label.setWordWrap(True)
        self.draft_label.setObjectName("muted")
        self.draft_label.setVisible(False)
        player_layout.addWidget(self.draft_label)
        self.tabs.addTab(self.player_tab, "Player")

        self._build_transcript_tab()

        self.console_tab = QWidget()
        console_layout = QVBoxLayout(self.console_tab)
        console_layout.setContentsMargins(5, 5, 5, 5)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(4000)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setPlaceholderText("Backend output, Japanese ASR and English translations appear here.")
        console_layout.addWidget(self.console)
        self.tabs.addTab(self.console_tab, "Console")
        layout.addWidget(self.tabs, 1)

        self.footer_panel = QWidget()
        footer = QHBoxLayout(self.footer_panel)
        footer.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("Ready. Paste a YouTube URL to begin.")
        self.status.setObjectName("muted")
        footer.addWidget(self.status, 1)
        self.stop_button = QPushButton("Stop Translation")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_translation)
        footer.addWidget(self.stop_button)
        layout.addWidget(self.footer_panel)

    def _build_transcript_tab(self):
        self.transcript_tab = QWidget()
        layout = QVBoxLayout(self.transcript_tab)
        filters = QHBoxLayout()
        self.transcript_search = QLineEdit()
        self.transcript_search.setPlaceholderText("Search Japanese, English or ASR diagnostics")
        self.transcript_search.textChanged.connect(self._schedule_transcript_refresh)
        self.needs_review_filter = QCheckBox("Needs review")
        self.needs_review_filter.toggled.connect(self._schedule_transcript_refresh)
        filters.addWidget(self.transcript_search, 1)
        filters.addWidget(self.needs_review_filter)
        layout.addLayout(filters)
        self.transcript_model = TranscriptTableModel(self.review, self)
        self.transcript_table = QTableView()
        self.transcript_table.setModel(self.transcript_model)
        self.transcript_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.transcript_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.transcript_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.transcript_table.setWordWrap(False)
        header = self.transcript_table.horizontalHeader()
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.transcript_table.selectionModel().selectionChanged.connect(self._select_review_cue)
        self.transcript_table.doubleClicked.connect(lambda _index: self._replay_review_cue())
        layout.addWidget(self.transcript_table, 1)
        self.review_info = QLabel("Select a cue to review. Corrections affect review and export only.")
        self.review_info.setWordWrap(True)
        self.review_info.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.review_info)
        editors = QHBoxLayout()
        self.ja_editor, self.en_editor = QPlainTextEdit(), QPlainTextEdit()
        self.ja_editor.setPlaceholderText("Japanese correction")
        self.en_editor.setPlaceholderText("English correction")
        for editor in (self.ja_editor, self.en_editor):
            editor.setMaximumHeight(90)
            editors.addWidget(editor)
        layout.addLayout(editors)
        actions = QHBoxLayout()
        self.save_correction_button = QPushButton("Save correction")
        self.save_correction_button.clicked.connect(self._save_review_correction)
        self.replay_cue_button = QPushButton("Replay selected cue")
        self.replay_cue_button.clicked.connect(self._replay_review_cue)
        self.file_language = QComboBox()
        for title, language in (("Bilingual", "bilingual"), ("English", "english"),
                                ("Japanese", "japanese")):
            self.file_language.addItem(title, language)
        self.file_language.setToolTip("Language layout used for subtitle import and export")
        self.import_corrections_button = QPushButton("Import corrections")
        self.import_corrections_button.clicked.connect(self._import_review_corrections)
        self.export_subtitles_button = QPushButton("Export subtitles")
        self.export_subtitles_button.clicked.connect(self._export_review_subtitles)
        for widget in (self.save_correction_button, self.replay_cue_button,
                       self.file_language, self.import_corrections_button, self.export_subtitles_button):
            actions.addWidget(widget)
        layout.addLayout(actions)
        tools = QHBoxLayout()
        self.performance_report_button = QPushButton("Performance report")
        self.performance_report_button.setToolTip("Generate HTML and JSON from recorded ASR comparison logs")
        self.performance_report_button.clicked.connect(self._generate_performance_report)
        self.refine_timings_button = QPushButton("Refine VOD timing…")
        self.refine_timings_button.setToolTip(
            "Stop playback, then select local audio/video matching this VOD. "
            "Runs optional WhisperX alignment in a separate backend process.")
        self.refine_timings_button.clicked.connect(self._refine_review_timings)
        tools.addWidget(self.performance_report_button)
        tools.addWidget(self.refine_timings_button)
        tools.addStretch(1)
        self.review_count = QLabel("0 cues")
        tools.addWidget(self.review_count)
        layout.addLayout(tools)
        self.tabs.addTab(self.transcript_tab, "Transcript")

    def _schedule_transcript_refresh(self, *_args):
        if not self._review_refresh_timer.isActive():
            self._review_refresh_timer.start()

    def _refresh_transcript(self):
        selected = self._editing_key
        self.transcript_model.refresh(self.transcript_search.text(), self.needs_review_filter.isChecked())
        self.review_count.setText(f"{len(self.transcript_model.rows)} / {len(self.review.cues)} cues")
        for row, (key, _cue) in enumerate(self.transcript_model.rows):
            if key == selected:
                self.transcript_table.selectRow(row)
                break
        if selected in self.review.cues:
            self._show_review_info(self.review.cues[selected])

    def _select_review_cue(self, *_args):
        rows = self.transcript_table.selectionModel().selectedRows()
        if not rows:
            return
        key, cue = self.transcript_model.rows[rows[0].row()]
        if key != self._editing_key:
            self._editing_key = key
            self.ja_editor.setPlainText(cue.ja)
            self.en_editor.setPlainText(cue.en)
        self._show_review_info(cue)

    def _show_review_info(self, cue):
        d = cue.diagnostic
        reason = d.get("fallback_reason") or d.get("chosen_by") or "No selection diagnostic"
        self.review_info.setText(
            f"{cue.start_s:.2f}–{cue.end_s:.2f}s · {len(cue.revisions)} revisions · {reason}\n"
            f"Haas: {str(d.get('haas', '—'))[:1000]}\n"
            f"Whisper: {str(d.get('whisper_turbo', '—'))[:1000]}")

    def _save_review_correction(self):
        if self._editing_key not in self.review.cues:
            self.status.setText("Select a transcript cue first.")
            return
        try:
            cue = self.review.correct(self._editing_key, self.ja_editor.toPlainText(),
                                      self.en_editor.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "Correction not saved", str(exc))
            return
        self._show_review_info(cue)
        self._schedule_transcript_refresh()
        self.status.setText("Correction saved for review and export.")

    def _replay_review_cue(self):
        cue = self.review.cues.get(self._editing_key)
        if cue is None:
            self.status.setText("Select a transcript cue first.")
        elif self._ready and self._control_ready and not self._live:
            self._request_seek(cue.start_s)
            self.tabs.setCurrentWidget(self.player_tab)
        else:
            self.status.setText("Replay requires the same VOD to be running.")

    def _export_review_subtitles(self):
        if not self.review.cues:
            self.status.setText("There are no finalized subtitles to export.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export reviewed subtitles", str(self.root / "subtitles.srt"),
            "SubRip (*.srt);;WebVTT (*.vtt);;ASS (*.ass)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".vtt" if "vtt" in selected else ".ass" if "ass" in selected else ".srt"
        try:
            from auto_translate.subtitle_files import export_subtitles
            count = export_subtitles(path, self.review.snapshot(), language=self.file_language.currentData())
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.status.setText(f"Exported {count} cues to {path}")

    def _import_review_corrections(self):
        if not self.review.cues:
            self.status.setText("Load finalized subtitles before importing corrections.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import corrections with matching timestamps", str(self.root),
            "Subtitle files (*.srt *.vtt *.ass)")
        if not path:
            return
        try:
            from auto_translate.subtitle_files import import_corrections
            revisions = import_corrections(path, self.review.snapshot(), language=self.file_language.currentData())
            for revision in revisions:
                cue = revision.revised
                self.review.correct(cue_key(cue.start_s, cue.end_s), cue.ja, cue.en)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Corrections not imported", str(exc))
            return
        if self._editing_key in self.review.cues:
            cue = self.review.cues[self._editing_key]
            self.ja_editor.setPlainText(cue.ja)
            self.en_editor.setPlainText(cue.en)
        self._schedule_transcript_refresh()
        self.status.setText(f"Imported {len(revisions)} corrections; original cue timing retained.")

    def _poll_diagnostics(self):
        if self._diagnostic_tail is None:
            return
        changed = False
        for record in self._diagnostic_tail.read():
            session = record.get("session_id")
            if self._diagnostic_session is not None and session != self._diagnostic_session:
                continue
            if self._diagnostic_session is None and session:
                self._diagnostic_session = session
            if record.get("stage") == "asr_fallback":
                record = {**record, "fallback_reason": record.get("fallback_reason") or record.get("reason", "Secondary ASR unavailable")}
            self.review.add_diagnostic(record)
            changed = True
        if changed:
            self._schedule_transcript_refresh()

    def _generate_performance_report(self):
        if self._review_job is not None:
            return
        log_path = self.root / "logs" / "asr-comparison.jsonl"
        if not log_path.is_file():
            self.status.setText("No ASR diagnostic log is available yet.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save performance report", str(self.root / "logs"))
        if not folder:
            return
        from auto_translate.quality_report import generate_report
        self.performance_report_button.setEnabled(False)
        self.status.setText("Generating performance report from comparison logs…")
        job = ReviewJob(lambda: generate_report(log_path, Path(folder)), self)
        self._review_job = job
        job.completed.connect(self._report_generated)
        job.finished.connect(self._review_job_finished)
        job.start()

    def _report_generated(self, result, error):
        self.performance_report_button.setEnabled(True)
        if error:
            self.status.setText(f"Report failed: {error}")
        else:
            self.status.setText(f"Report: {result['html_path']}")
            self._write_console(f"Performance report: {result['html_path']}\nJSON: {result['json_path']}\n")

    def _review_job_finished(self):
        job, self._review_job = self._review_job, None
        if job is not None:
            job.deleteLater()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _refine_review_timings(self):
        if self.process is not None or self._refine_process is not None:
            self.status.setText("Stop translation before refining VOD timing.")
            return
        if self._review_live or not self.review.cues:
            self.status.setText("Timing refinement requires finalized VOD subtitles.")
            return
        media, _ = QFileDialog.getOpenFileName(
            self, "Select matching local VOD audio/video (original full timeline)", str(self.root),
            "Media files (*.wav *.flac *.mp3 *.m4a *.mp4 *.mkv *.webm);;All files (*)")
        if not media:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save refined subtitles", str(self.root / "subtitles-aligned.srt"),
            "SubRip (*.srt);;WebVTT (*.vtt);;ASS (*.ass)")
        if not destination:
            return
        if not Path(destination).suffix:
            destination += ".srt"
        try:
            from auto_translate.subtitle_files import export_subtitles
            temporary = tempfile.TemporaryDirectory(prefix="video-translate-refine-")
            self._refine_temp = temporary
            source = Path(temporary.name) / "reviewed.srt"
            export_subtitles(source, self.review.snapshot(), language="bilingual")
            python = backend_python(self.root, frozen=getattr(sys, "frozen", False))
            if os.name == "nt" and python.with_name("pythonw.exe").is_file():
                python = python.with_name("pythonw.exe")
        except (OSError, ValueError, RuntimeError) as exc:
            self._cleanup_refinement()
            QMessageBox.warning(self, "Cannot refine timing", str(exc))
            return
        process = QProcess(self)
        self._refine_process = process
        self._refine_output = ""
        process.setWorkingDirectory(str(self.root))
        environment = QProcessEnvironment()
        for key, value in subprocess_environment(self.root).items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setProgram(str(python))
        process.setArguments(["-u", "-m", "auto_translate", "--refine-subtitles", str(source),
                              "--export", destination, "--export-language", "bilingual", media])
        process.readyReadStandardOutput.connect(self._read_refinement_output)
        process.finished.connect(lambda code, status: self._refinement_finished(code, status, destination))
        process.errorOccurred.connect(self._refinement_error)
        self.start_button.setEnabled(False)
        self.refine_timings_button.setEnabled(False)
        self.status.setText("Refining timing in the backend… Optional WhisperX must be installed.")
        process.start()

    def _read_refinement_output(self):
        if self._refine_process is not None:
            text = bytes(self._refine_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._refine_output = (self._refine_output + text)[-12000:]
            self._write_console(text)

    def _refinement_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self._refine_process is not None:
            self.status.setText(f"Timing refinement could not start: {self._refine_process.errorString()}")
            self._cleanup_refinement()

    def _refinement_finished(self, code, status, destination):
        if self._refine_process is None:
            return
        self._read_refinement_output()
        success = code == 0 and status == QProcess.ExitStatus.NormalExit
        self.status.setText(f"Refined subtitles: {destination}" if success else
                            f"Timing refinement failed. See Console. {self._refine_output[-500:]}")
        self._cleanup_refinement()

    def _cleanup_refinement(self):
        process, self._refine_process = self._refine_process, None
        if process is not None:
            process.deleteLater()
        if self._refine_temp is not None:
            self._refine_temp.cleanup()
            self._refine_temp = None
        self.start_button.setEnabled(self.process is None)
        self.refine_timings_button.setEnabled(True)
        if self._closing:
            QTimer.singleShot(0, self.close)

    def paste_url(self):
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_input.setText(text)

    def _lifecycle(self, message):
        line = f"session={self._session} {message}"
        self._write_console(line + "\n")
        try:
            with (self.root / "logs" / "gui-lifecycle.log").open("a", encoding="utf8") as stream:
                print(line, file=stream)
        except OSError:
            pass

    def _write_console(self, text: str):
        if not text:
            return
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.ensureCursorVisible()
        if self._log_stream:
            try:
                self._log_stream.write(text)
                self._log_stream.flush()
            except OSError:
                pass

    def _listen_for_control(self):
        self._close_control()
        self._control_lost = False
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(4)
            server.setblocking(False)
        except OSError:
            server.close()
            raise
        self._control_server = server
        self._control_token = secrets.token_hex(32)
        self._control_timer.start()
        return ["--gui-control-port", str(server.getsockname()[1]),
                "--gui-control-token", self._control_token]

    def _close_control(self):
        self._control_timer.stop()
        for peer in self._control_peers:
            peer.close()
        self._control_peers.clear()
        for sock in (self._control_socket, self._control_server):
            if sock is not None:
                sock.close()
        self._control_socket = self._control_server = None
        self._control_ready = False

    def _control_disconnected(self):
        self._close_control()
        self._pending_seek_target = None
        self._update_seek_buttons()
        if not self._control_lost and not self._stopping:
            self._control_lost = True
            self.status.setText("Seek channel lost. See Console.")
            self._write_console("GUI: control channel lost. Stopping translation.\n")

            self.stop_translation()

    def _poll_control(self):
        # Nonblocking sockets keep Qt responsive and reuse the packaged Qt runtime.
        try:
            if self._control_socket is not None:
                if select.select([self._control_socket], [], [], 0)[0]:
                    # Backend-to-GUI events use stdout. Readability here means
                    # EOF or unexpected protocol data.
                    self._control_disconnected()
                return
            server = self._control_server
            if server is None:
                return
            if select.select([server], [], [], 0)[0]:
                peer, _ = server.accept()
                peer.setblocking(False)
                if len(self._control_peers) >= 8:
                    peer.close()
                else:
                    self._control_peers[peer] = (b"", time.monotonic() + 5)
            for peer, (data, deadline) in list(self._control_peers.items()):
                if time.monotonic() < deadline:
                    try:
                        chunk = peer.recv(128)
                    except BlockingIOError:
                        continue
                    except OSError:
                        chunk = b""
                    data += chunk
                    if chunk and b"\n" not in data and len(data) <= 64:
                        self._control_peers[peer] = (data, deadline)
                        continue
                    if secrets.compare_digest(data, self._control_token.encode() + b"\n"):
                        try:
                            peer.sendall(b"OK\n")
                        except OSError:
                            pass
                        else:
                            del self._control_peers[peer]
                            self._control_socket = peer
                            server.close()
                            self._control_server = None
                            for other in self._control_peers:
                                other.close()
                            self._control_peers.clear()
                            return
                del self._control_peers[peer]
                peer.close()
        except OSError as exc:
            self._write_console(f"GUI control socket error: {exc}\n")
            self._control_disconnected()

    def _new_process(self) -> QProcess:
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        return proc

    def start_translation(self):
        if self.process is not None or self._refine_process is not None:
            return
        url = self.url_input.text().strip()
        if not valid_youtube_url(url):
            QMessageBox.warning(self, "Invalid URL", "Paste a valid YouTube live or VOD video URL.")
            return
        try:
            prepare_project(self.root)
            self.tabs.setCurrentWidget(self.player_tab)
            # The native HWND must exist before MPV --wid is launched.
            hwnd = int(self.video_surface.winId())
            command, args = build_backend_command(
                self.root, url, int(self.quality.currentData()), hwnd,
                frozen=getattr(sys, "frozen", False),
                rtx_vsr=self.vsr_checkbox.isChecked(),
            )
            if self.preview_checkbox.isChecked():
                args.append("--preview")
            env = subprocess_environment(self.root)
            self._session = uuid.uuid4().hex
            env["VIDEO_TRANSLATE_SESSION"] = self._session
            if os.name == "nt":
                # The GUI job covers even a crash before the gate joins the
                # per-translation job. Existing children are never adopted.
                bind_process_tree()
                self._job = WindowsJob(self._session)
            qenv = QProcessEnvironment()
            for key, value in env.items():
                qenv.insert(key, value)
            args.extend(self._listen_for_control())
            validate_gui_launch(args)
            if os.name == "nt":
                command, args = session_launcher(command, args)
        except (OSError, ValueError) as exc:
            self._close_control()
            if self._job:
                self._job.close()
                self._job = None
            logging.exception("session=%s launch/containment failed", self._session)
            if (self.root / "logs").is_dir():
                with (self.root / "logs" / "run.log").open("a", encoding="utf8") as stream:
                    print(f"session={self._session} launch/containment failed: {exc}", file=stream)
            QMessageBox.critical(self, "Cannot start translation", str(exc))
            return

        self.console.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_line = ""
        self._session_error = ""
        self._ready = False
        self._stopping = False
        self.review = TranscriptReview()
        self.transcript_model.review = self.review
        self._editing_key = None
        self._review_generation = None
        self._review_live = False
        self._diagnostic_session = None
        self._diagnostic_tail = DiagnosticTail(self.root / "logs" / "asr-comparison.jsonl", from_end=True)
        self._diagnostic_timer.start()
        self.ja_editor.clear()
        self.en_editor.clear()
        self._clear_draft()
        self._refresh_transcript()
        self._reset_player_controls()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Initializing models and loading video... See Console for progress.")
        try:
            self._log_stream = (self.root / "logs" / "run.log").open("a", encoding="utf-8")
        except OSError:
            self._log_stream = None
        proc = self._new_process()
        self.process = proc
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessEnvironment(qenv)
        proc.setProgram(command)
        proc.setArguments(args)
        self._lifecycle(f"Starting translation backend gui_pid={os.getpid()}")
        self._lifecycle(f"frontend={sys.executable} gui_source={__file__} backend_root={self.root} control_protocol=1")
        self._lifecycle(f"build={self._build_identity['commit']} gui_sha256={self._build_identity['gui_sha256']}")
        if "--rtx-vsr" in args:
            self._write_console(
                "RTX VSR requested: NVIDIA D3D11 scaling. "
                "Actual activation requires MPV filter support and RTX VSR "
                "enabled in NVIDIA settings.\n"
            )
        # Keep the native video host visible while MPV attaches to its HWND.
        # The user can open Console during startup to inspect model logs.
        proc.start()

    def _on_output(self):
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput())
        decoded = self._decoder.decode(raw)
        if not decoded:
            return
        self._pending_line += decoded
        while "\n" in self._pending_line:
            line, self._pending_line = self._pending_line.split("\n", 1)
            if line.strip() == READY_MARKER:
                self._session_ready()
            elif (event := parse_event(line.strip())) is not None:
                self._on_player_event(event)
            else:
                self._write_console(line + "\n")
        # Preserve partial multibyte chunks across Qt readyRead signals.

    def _session_ready(self):
        if self._ready or self._stopping:
            return
        self._ready = True
        self.setup_panel.setVisible(False)
        self.tabs.setCurrentWidget(self.player_tab)
        self.fullscreen_button.setEnabled(True)
        self._update_seek_buttons()
        if not self._control_lost:
            self.status.setText("Running." if self._control_ready else "Waiting for player controls...")

    @staticmethod
    def _format_time(value: float | None) -> str:
        if value is None or value < 0:
            return "--:--"
        seconds = int(value)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return (f"{hours}:{minutes:02}:{seconds:02}" if hours else
                f"{minutes:02}:{seconds:02}")

    def _reset_player_controls(self):
        self._duration_s = None
        self._live = False
        self._seeking = False
        self._pending_seek_target = None
        self._pending_seek_until = 0.0
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.time_label.setText("--:-- / --:--")
        for widget in (self.rewind_button, self.forward_button, self.pause_button, self.fullscreen_button):
            widget.setEnabled(False)
        self.pause_button.setText("Pause")

    def _update_seek_buttons(self):
        self.pause_button.setEnabled(self._ready and self._control_ready)
        seekable = bool(self._ready and self._control_ready and not self._live and
                        self._duration_s is not None and self._duration_s > 0)
        for widget in (self.rewind_button, self.forward_button, self.seek_slider):
            widget.setEnabled(seekable)
        if self._live:
            self.time_label.setText("LIVE · no seeking")

    def _on_player_event(self, event: dict):
        action = event.get("event")
        if action in ("subtitle", "preview"):
            generation = event.get("generation", 0)
            if type(generation) is not int or generation < 0:
                return
            if self._review_generation is not None and generation < self._review_generation:
                return
            if generation != self._review_generation:
                self._clear_draft()
                self._review_generation = generation
            if action == "preview":
                if event.get("clear"):
                    self._clear_draft()
                elif isinstance(event.get("ja"), str) and len(event["ja"]) <= 100000:
                    self.draft_label.setText("Draft Japanese: " + event["ja"])
                    self.draft_label.setVisible(bool(event["ja"].strip()))
                return
            self._clear_draft()
            session = event.get("diagnostic_session")
            if session:
                self._diagnostic_session = session
            try:
                self.review.add(event)
            except (KeyError, TypeError, ValueError) as exc:
                self._write_console(f"Ignored invalid subtitle event: {exc}\n")
                return
            self._schedule_transcript_refresh()
            return
        if action == "session_error":
            self._session_error = str(event.get("message", ""))
            self.status.setText(self._session_error)
            return
        if action == "control_ready":
            if self._control_socket is not None:
                self._control_ready = True
                if self._stopping:
                    self._send_player_command("quit")
                self._update_seek_buttons()
                if self._ready:
                    self.status.setText("Running.")
            return
        if action == "control_closed":
            self._control_disconnected()
            return
        if action == "toggle_fullscreen":
            self.toggle_fullscreen()
            return
        if action == "exit_fullscreen":
            self.exit_fullscreen()
            return
        if action == "seek_ack":
            target = event.get("target")
            if event.get("accepted"):
                self._clear_draft()
                if isinstance(target, (int, float)):
                    # The backend may clamp seeks near a VOD's end.
                    self._pending_seek_target = float(target)
                    self._pending_seek_until = time.monotonic() + 15.0
                if not self._control_lost and not self._stopping:
                    self.status.clear()
            else:
                self._pending_seek_target = None
                self.status.setText(
                    "Seek could not be completed; check the Console tab."
                )
            return
        if action not in ("ready", "status"):
            return
        self._live = bool(event.get("live", self._live))
        self._review_live = self._live
        raw_duration = event.get("duration")
        if isinstance(raw_duration, (int, float)) and 0 < raw_duration < 1e8:
            self._duration_s = float(raw_duration)
        self._update_seek_buttons()
        if self._duration_s is not None:
            maximum = max(1, int(self._duration_s))
            if self.seek_slider.maximum() != maximum:
                self.seek_slider.setMaximum(maximum)

        if action == "status":
            self.pause_button.setText("Play" if event.get("paused") else "Pause")
            raw_position = event.get("position")
            if isinstance(raw_position, (int, float)) and raw_position >= 0:
                position = float(raw_position)
                if self._pending_seek_target is not None:
                    if abs(position - self._pending_seek_target) < 2 or time.monotonic() > self._pending_seek_until:
                        self._pending_seek_target = None
                    else:
                        return
                if not self._seeking:
                    self.seek_slider.blockSignals(True)
                    self.seek_slider.setValue(min(int(position), self.seek_slider.maximum()))
                    self.seek_slider.blockSignals(False)
                    if not self._live:
                        self.time_label.setText(
                            f"{self._format_time(position)} / {self._format_time(self._duration_s)}"
                        )

    def _clear_draft(self):
        self.draft_label.clear()
        self.draft_label.setVisible(False)

    def _send_player_command(self, action: str, value: float | None = None) -> bool:
        if not ((self._ready or action == "quit") and self._control_ready and self._control_socket is not None):
            return False
        if self._stopping and action != "quit":
            return False
        message = {"action": action}
        if value is not None:
            message["value"] = float(value)
        try:
            self._control_socket.sendall((json.dumps(message) + "\n").encode("utf-8"))
        except OSError:
            self._control_disconnected()
            return False
        return True

    def _begin_seek(self):
        self._seeking = True
        if self._fullscreen:
            self._fullscreen_activity_at = time.monotonic()

    def _seek_preview(self, value: int):
        if self._seeking and self._duration_s is not None:
            self.time_label.setText(
                f"{self._format_time(value)} / {self._format_time(self._duration_s)}"
            )

    def _request_seek(self, value: float):
        """Submit a completed timeline action; never seek live streams."""
        if self._fullscreen:
            self.video_controls.show()
            self._fullscreen_activity_at = time.monotonic()
        if self._live or self._duration_s is None or not self._ready:
            self.status.setText("Seeking unavailable for live or uninitialized playback.")
            return
        target = min(max(float(value), 0.0), self._duration_s)
        if not self._send_player_command("seek", target):
            return
        self._pending_seek_target = target
        self._pending_seek_until = time.monotonic() + 15.0
        self.time_label.setText(
            f"{self._format_time(target)} / {self._format_time(self._duration_s)}"
        )
        self._write_console(f"GUI: seek requested to {self._format_time(target)}\n")

    def _finish_seek(self):
        self._seeking = False
        if self._fullscreen:
            self._fullscreen_activity_at = time.monotonic()
        self._request_seek(float(self.seek_slider.value()))

    def _fullscreen_controls_tick(self):
        """Reveal on pointer movement; hide after 3s of fullscreen inactivity."""
        if not self._fullscreen:
            self._fullscreen_controls_timer.stop()
            return
        now = time.monotonic()
        cursor = QCursor.pos()
        if cursor != self._fullscreen_cursor_pos:
            self._fullscreen_cursor_pos = cursor
            self._fullscreen_activity_at = now
            self.video_controls.show()
        elif self.seek_slider.isSliderDown() or self._seeking:
            # Never dismiss controls while dragging the VOD timeline.
            self._fullscreen_activity_at = now
            self.video_controls.show()
        elif now - self._fullscreen_activity_at >= self.FULLSCREEN_CONTROLS_TIMEOUT_S:
            self.video_controls.hide()

    def toggle_fullscreen(self):
        if getattr(self, "_fullscreen_transition", False):
            return
        self._fullscreen_transition = True
        try:
            self._toggle_fullscreen()
        finally:
            self._fullscreen_transition = False

    def _toggle_fullscreen(self):
        if not self._ready:
            return
        if self._fullscreen:
            self.exit_fullscreen()
            return
        self._fullscreen = True
        self._was_maximized = self.isMaximized()
        self.tabs.setCurrentWidget(self.player_tab)
        self.heading.hide()
        self.setup_panel.hide()
        self.tabs.tabBar().hide()
        self.footer_panel.hide()
        # Leave playback controls visible and clickable in fullscreen.
        # Hiding this bar previously removed both the VOD seekbar and the
        # only mouse-accessible way to exit fullscreen.
        self.video_controls.show()
        self.fullscreen_button.setText("Exit Fullscreen")
        self.fullscreen_button.setToolTip("Exit fullscreen (F, F11 or Esc). Move mouse to reveal controls.")
        self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
        self.centralWidget().layout().setSpacing(0)
        self.player_tab.layout().setContentsMargins(0, 0, 0, 0)
        self.tabs.setStyleSheet("QTabWidget::pane { border: none; }")
        # Keep MPV's native child HWND in place. Fullscreening the *host*
        # (rather than the --wid child window) preserves its render surface.
        self.showFullScreen()
        self._fullscreen_cursor_pos = QCursor.pos()
        self._fullscreen_activity_at = time.monotonic()
        self._fullscreen_controls_timer.start()

    def exit_fullscreen(self):
        if not self._fullscreen:
            return
        self._fullscreen = False
        self._fullscreen_controls_timer.stop()
        self.showNormal()
        if self._was_maximized:
            self.showMaximized()
        self.heading.show()
        self.setup_panel.setVisible(not self._ready)
        self.tabs.tabBar().show()
        self.footer_panel.show()
        self.video_controls.show()
        self.fullscreen_button.setText("Fullscreen")
        self.fullscreen_button.setToolTip("Fullscreen player (F11, F or double-click MPV; Esc exits)")
        self.centralWidget().layout().setContentsMargins(14, 12, 14, 12)
        self.centralWidget().layout().setSpacing(10)
        self.player_tab.layout().setContentsMargins(5, 5, 5, 5)
        self.tabs.setStyleSheet("")

    def stop_translation(self):
        self.exit_fullscreen()
        if self.process is None or self._stopping:
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping translation and player...")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._lifecycle("graceful shutdown requested")
            self._send_player_command("quit")
            self._shutdown_timer.start(GUI_SHUTDOWN_MS)

    def _force_shutdown(self):
        if self.process is None:
            return
        self._lifecycle("timeout; forcing session shutdown")
        self._terminate_job()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _terminate_job(self):
        if self._job:
            try:
                members = self._job.pids()
                if members:
                    self._lifecycle(f"terminating owned PIDs={members}")
                # Terminate even an empty job: a late launcher cannot add work.
                self._job.terminate()
            except OSError as exc:
                self._lifecycle(f"job termination/query error: {exc}")
                self._job.close()
                self._job = None
                self._lifecycle("kill-on-close fallback; remaining membership unverified")

    def _on_error(self, error):
        if self.process is None:
            return
        self._write_console(f"Backend process error: {self.process.errorString()}\n")
        if error == QProcess.ProcessError.FailedToStart:
            self._session_finished("Backend could not start; verify your .venv environment.")

    def _on_finished(self, exit_code, exit_status):
        if self.process is None:
            return
        self._on_output()
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._pending_line += tail
        if self._pending_line:
            self._write_console(self._pending_line + "\n")
        self._pending_line = ""
        normal = (exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit)
        reason = "Stopped." if self._stopping else (
            "Video finished." if normal else
            self._session_error or f"Backend stopped with exit code {exit_code}; check Console / logs/run.log.")
        self._session_finished(reason)

    def _session_finished(self, reason: str):
        self._shutdown_timer.stop()
        self._close_control()
        if self._job:
            self._terminate_job()
            self._verify_job(reason, time.monotonic() + 3)
            return
        self._finish_ui(reason)

    def _verify_job(self, reason, deadline):
        if self._job:
            try:
                remaining = self._job.pids()
            except OSError as exc:
                remaining = f"unknown ({exc})"
            if remaining and time.monotonic() < deadline:
                QTimer.singleShot(50, lambda: self._verify_job(reason, deadline))
                return
            self._lifecycle(f"remaining owned PIDs={remaining}")
            self._job.close()
            self._job = None
        self._finish_ui(reason)

    def _finish_ui(self, reason):
        self._poll_diagnostics()
        self._diagnostic_timer.stop()
        self._clear_draft()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        if self._log_stream:
            self._log_stream.close()
            self._log_stream = None
        self.exit_fullscreen()
        self._reset_player_controls()
        self.setup_panel.setVisible(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._ready = False
        self._stopping = False
        self.status.setText(reason)
        if self._closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent):
        if self.process is not None:
            self._closing = True
            self.stop_translation()
            event.ignore()
            return
        if self._refine_process is not None:
            self._closing = True
            self._refine_process.kill()
            event.ignore()
            return
        if self._review_job is not None and self._review_job.isRunning():
            self._closing = True
            self.status.setText("Finishing report before closing…")
            event.ignore()
            return
        self._close_control()
        if self._log_stream:
            self._log_stream.close()
            self._log_stream = None
        event.accept()


def _smoke_review(window, root):
    """Exercise lazy file/report imports in both script and frozen entry modes."""
    from auto_translate.subtitle_files import export_subtitles, import_subtitles
    from auto_translate.quality_report import generate_report
    window._on_player_event(dict(event="subtitle", start_s=1, end_s=2,
                                 ja="こんにちは", en="Hello", generation=0))
    window._refresh_transcript()
    if window.transcript_model.rowCount() != 1:
        raise RuntimeError("Transcript model failed its packaged smoke check")
    with tempfile.TemporaryDirectory(prefix="video-translate-smoke-") as folder:
        folder = Path(folder)
        subs = folder / "review.srt"
        export_subtitles(subs, window.review.snapshot(), language="bilingual")
        if import_subtitles(subs, language="bilingual")[0].en != "Hello":
            raise RuntimeError("Packaged subtitle round trip failed")
        events = folder / "events.jsonl"
        events.write_text(json.dumps(dict(session_id="smoke", stage="translation",
                                          start_s=1, end_s=2, final_ja="こんにちは", final_en="Hello")) + "\n",
                          encoding="utf-8")
        if not generate_report(events)["html_path"].is_file():
            raise RuntimeError("Packaged report generation failed")
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "gui-packaged-smoke.log").write_text("Transcript/files/report smoke passed\n", encoding="utf-8")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Translate")
    app.setStyleSheet(_STYLE)
    try:
        root = discover_project_root()
    except FileNotFoundError as exc:
        QMessageBox.critical(None, "Project not found", str(exc))
        return 1
    window = VideoTranslateWindow(root)
    window.show()
    if "--smoke-test" in sys.argv:
        try:
            if "--verify-build" in sys.argv or "--verify-source-sha" in sys.argv:
                commit = sys.argv[sys.argv.index("--verify-build") + 1]
                digest = sys.argv[sys.argv.index("--verify-source-sha") + 1]
                verify_build(window._build_identity, commit, digest)
            _smoke_review(window, root)
        except Exception:
            import traceback
            (root / "logs").mkdir(exist_ok=True)
            (root / "logs" / "gui-packaged-smoke.log").write_text(traceback.format_exc(), encoding="utf-8")
            window.close()
            return 1
        QTimer.singleShot(250, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
