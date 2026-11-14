"""Windows PySide6 frontend for the existing multi-process translation pipeline.

The GUI never imports or initializes CUDA/ASR models. QProcess runs the same
Python backend used by Start translate.bat and native MPV renders into a HWND.
"""
import codecs
import json
import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont, QTextCursor, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
    QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from auto_translate.gui_control import parse_event
from auto_translate.gui_runner import (
    QUALITY_OPTIONS, READY_MARKER, build_backend_command, discover_project_root,
    prepare_project, subprocess_environment, valid_youtube_url,
)

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


class VideoTranslateWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.process: QProcess | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_line = ""
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
        self.setWindowTitle("Video Translate - Dual ASR")
        self.setMinimumSize(750, 520)
        self.resize(1120, 790)
        self._build_ui()
        QShortcut(QKeySequence("F11"), self).activated.connect(self.toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.exit_fullscreen)

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
        self.pause_button.clicked.connect(lambda: self._send_player_command("pause"))
        self.rewind_button = QPushButton("-10s")
        self.rewind_button.clicked.connect(lambda: self._send_player_command("seek_relative", -10))
        self.forward_button = QPushButton("+10s")
        self.forward_button.clicked.connect(lambda: self._send_player_command("seek_relative", 10))
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._begin_seek)
        self.seek_slider.sliderReleased.connect(self._finish_seek)
        self.seek_slider.valueChanged.connect(self._seek_preview)
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
        self.tabs.addTab(self.player_tab, "Player")

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

    def paste_url(self):
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_input.setText(text)

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

    def _new_process(self) -> QProcess:
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        return proc

    def start_translation(self):
        if self.process is not None:
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
            )
            env = subprocess_environment(self.root)
            qenv = QProcessEnvironment()
            for key, value in env.items():
                qenv.insert(key, value)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Cannot start translation", str(exc))
            return

        self.console.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_line = ""
        self._ready = False
        self._stopping = False
        self._reset_player_controls()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Initializing models and loading video... See Console for progress.")
        try:
            self._log_stream = (self.root / "logs" / "run.log").open("w", encoding="utf-8")
        except OSError:
            self._log_stream = None
        proc = self._new_process()
        self.process = proc
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessEnvironment(qenv)
        proc.setProgram(command)
        proc.setArguments(args)
        self._write_console("Starting translation backend...\n")
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
        self.pause_button.setEnabled(True)
        self.fullscreen_button.setEnabled(True)
        self._update_seek_buttons()
        self.status.setText("Running. MPV is embedded; translations appear in Console.")

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
        seekable = bool(self._ready and not self._live and
                        self._duration_s is not None and self._duration_s > 0)
        for widget in (self.rewind_button, self.forward_button, self.seek_slider):
            widget.setEnabled(seekable)
        if self._live:
            self.time_label.setText("LIVE · no seeking")

    def _on_player_event(self, event: dict):
        action = event.get("event")
        if action == "toggle_fullscreen":
            self.toggle_fullscreen()
            return
        if action == "exit_fullscreen":
            self.exit_fullscreen()
            return
        if action not in ("ready", "status"):
            return
        self._live = bool(event.get("live", self._live))
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

    def _send_player_command(self, action: str, value: float | None = None):
        if (not self._ready or self.process is None or
                self.process.state() != QProcess.ProcessState.Running):
            return
        message = {"action": action}
        if value is not None:
            message["value"] = float(value)
        self.process.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))

    def _begin_seek(self):
        self._seeking = True

    def _seek_preview(self, value: int):
        if self._seeking and self._duration_s is not None:
            self.time_label.setText(
                f"{self._format_time(value)} / {self._format_time(self._duration_s)}"
            )

    def _finish_seek(self):
        self._seeking = False
        if self._live or self._duration_s is None:
            return
        target = float(self.seek_slider.value())
        self._pending_seek_target = target
        self._pending_seek_until = time.monotonic() + 8.0
        self._send_player_command("seek", target)

    def toggle_fullscreen(self):
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
        self.video_controls.hide()
        self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
        self.centralWidget().layout().setSpacing(0)
        self.player_tab.layout().setContentsMargins(0, 0, 0, 0)
        self.tabs.setStyleSheet("QTabWidget::pane { border: none; }")
        # Keep MPV's native child HWND in place. Fullscreening the *host*
        # (rather than the --wid child window) preserves its render surface.
        self.showFullScreen()

    def exit_fullscreen(self):
        if not self._fullscreen:
            return
        self._fullscreen = False
        self.showNormal()
        if self._was_maximized:
            self.showMaximized()
        self.heading.show()
        self.setup_panel.setVisible(not self._ready)
        self.tabs.tabBar().show()
        self.footer_panel.show()
        self.video_controls.show()
        self.centralWidget().layout().setContentsMargins(14, 12, 14, 12)
        self.centralWidget().layout().setSpacing(10)
        self.player_tab.layout().setContentsMargins(5, 5, 5, 5)
        self.tabs.setStyleSheet("")
        self.fullscreen_button.setText("Fullscreen")

    def stop_translation(self):
        self.exit_fullscreen()
        if self.process is None or self._stopping:
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping translation and player...")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            QTimer.singleShot(4000, self._kill_if_still_running)

    def _kill_if_still_running(self):
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self._write_console("Graceful shutdown timed out; terminating backend.\n")
            self.process.kill()

    def _on_error(self, error):
        if self.process is None or self._closing:
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
            f"Backend stopped with exit code {exit_code}; check Console / logs/run.log.")
        self._session_finished(reason)

    def _session_finished(self, reason: str):
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

    def closeEvent(self, event: QCloseEvent):
        self._closing = True
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(3000)
        if self._log_stream:
            self._log_stream.close()
            self._log_stream = None
        event.accept()


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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
