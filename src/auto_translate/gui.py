"""Windows PySide6 frontend for the existing multi-process translation pipeline.

The GUI never imports or initializes CUDA/ASR models. QProcess runs the same
Python backend used by Start translate.bat and native MPV renders into a HWND.
"""
import codecs
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
    QTabWidget, QVBoxLayout, QWidget,
)

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
        self.setWindowTitle("Video Translate - Dual ASR")
        self.setMinimumSize(750, 520)
        self.resize(1120, 790)
        self._build_ui()

    def _build_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        self.setCentralWidget(central)

        heading = QLabel("Video Translate")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

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

        footer = QHBoxLayout()
        self.status = QLabel("Ready. Paste a YouTube URL to begin.")
        self.status.setObjectName("muted")
        footer.addWidget(self.status, 1)
        self.stop_button = QPushButton("Stop Translation")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_translation)
        footer.addWidget(self.stop_button)
        layout.addLayout(footer)

    def paste_url(self):
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_input.setText(text)

    def _write_console(self, text: str):
        if not text:
            return
        self.console.moveCursor(self.console.textCursor().MoveOperation.End)
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
        self.tabs.setCurrentWidget(self.console_tab)
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
            else:
                self._write_console(line + "\n")
        # Preserve partial multibyte chunks across Qt readyRead signals.

    def _session_ready(self):
        if self._ready or self._stopping:
            return
        self._ready = True
        self.setup_panel.setVisible(False)
        self.tabs.setCurrentWidget(self.player_tab)
        self.status.setText("Running. MPV is embedded; translations appear in Console.")

    def stop_translation(self):
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
