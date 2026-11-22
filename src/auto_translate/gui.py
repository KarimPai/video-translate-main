"""Windows PySide6 frontend for the existing multi-process translation pipeline.

The GUI never imports or initializes CUDA/ASR models. QProcess runs the same
Python backend and native MPV renders into a HWND.
"""
import codecs
import hashlib
import json
import logging
import os
import secrets
import select
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QFont,
    QKeySequence,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from auto_translate.config import load_config, resolve_model_identity
from auto_translate.gui_control import GUI_SHUTDOWN_MS, parse_event
from auto_translate.gui_runner import (
    QUALITY_OPTIONS,
    READY_MARKER,
    build_backend_command,
    diagnostic_rows,
    discover_project_root,
    prepare_project,
    session_launcher,
    subprocess_environment,
    valid_youtube_url,
)
from auto_translate.runtime_diagnostics import open_rotating_log
from auto_translate.windows_job import WindowsJob, bind_process_tree

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


class VideoTranslateWindow(QMainWindow):
    PLAYER_IDLE_TIMEOUT_S = 3.0
    diagnosticsExported = Signal(str, str)

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        # Keep GUI-only choices per project and Windows account, not in tracked
        # config.toml. This also works when the application is packaged as an exe.
        self._preferences = QSettings("KarimPai", "VideoTranslate")
        project_id = hashlib.sha256(str(root.resolve()).casefold().encode("utf-8")).hexdigest()[:24]
        self._preferences_prefix = f"projects/{project_id}"
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
        self._player_cursor_pos = None
        self._player_activity_at = 0.0
        self._export_thread = None
        self._asr_model_overrides: dict[str, str] = {}
        self._mt_model_overrides: dict[str, str] = {}
        self._last_asr_backend: str | None = None
        self._last_mt_backend: str | None = None
        self.setWindowTitle("Video Translate")
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
        self.diagnosticsExported.connect(self._on_diagnostics_exported)
        # Embedded MPV cannot autohide the host cursor; Qt owns its idle state.
        self._player_activity_timer = QTimer(self)
        self._player_activity_timer.setInterval(150)
        self._player_activity_timer.timeout.connect(self._player_activity_tick)
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
        self._build_diagnostics_tab()
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

    def _build_diagnostics_tab(self):
        self.diagnostics_tab = QScrollArea()
        self.diagnostics_tab.setWidgetResizable(True)
        self.diagnostics_tab.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.diagnostics_tab.setWidget(content)
        layout = QVBoxLayout(content)
        self.feature_settings = QGroupBox("Accuracy and playback settings (apply when starting)")
        settings = QFormLayout(self.feature_settings)
        self.asr_backend = QComboBox()
        self.asr_backend.setToolTip("Speech recognition backend. Select from available models.")
        settings.addRow("ASR backend", self.asr_backend)
        self.mt_backend = QComboBox()
        self.mt_backend.setToolTip("Japanese to English translation backend. Select from available models.")
        settings.addRow("Translation backend", self.mt_backend)
        self.asr_model = QLineEdit()
        self.asr_model.setPlaceholderText("Optional: override ASR model path")
        self.asr_model.setToolTip("Override the model path for the selected ASR backend.")
        settings.addRow("ASR model path", self.asr_model)
        self.mt_model = QLineEdit()
        self.mt_model.setPlaceholderText("Optional: override translation model path")
        self.mt_model.setToolTip("Override the model path for the selected translation backend.")
        settings.addRow("Translation model path", self.mt_model)
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 20)
        self.context_lines.setValue(6)
        self.context_lines.setSuffix(" lines")
        self.context_lines.setToolTip(
            "Previous Japanese dialogue lines, without earlier English translations. "
            "Context clears on seeking, changing videos, or the configured audio-silence gap."
        )
        settings.addRow("Translation context", self.context_lines)
        self.safe_playback_checkbox = QCheckBox("Safe Playback")
        self.safe_playback_checkbox.setToolTip(
            "Coordinate inference and bound queued audio while preserving hardware video decoding."
        )
        settings.addRow(self.safe_playback_checkbox)
        self.diagnostics_checkbox = QCheckBox("Diagnostic logging and resource dashboard")
        self.diagnostics_checkbox.setChecked(True)
        settings.addRow(self.diagnostics_checkbox)
        layout.addWidget(self.feature_settings)
        self._load_feature_settings()

        dashboard = QGroupBox("Resource dashboard")
        rows = QFormLayout(dashboard)
        self.diagnostics_observation_status = QLabel("Waiting for backend observations.")
        rows.addRow(self.diagnostics_observation_status)
        self.diagnostic_labels = {}
        for key, label in (("gpu", "GPU memory"), ("activity", "GPU activity"),
                           ("stages", "Latest processing"), ("subtitles", "Subtitles"),
                           ("queues", "Queues"), ("playback", "Playback"),
                           ("workers", "Owned workers")):
            value = QLabel("unknown")
            value.setWordWrap(True)
            value.setTextFormat(Qt.TextFormat.PlainText)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.diagnostic_labels[key] = value
            rows.addRow(label, value)
        layout.addWidget(dashboard)
        self._update_diagnostics({})
        export_row = QHBoxLayout()
        self.include_windows_events = QCheckBox("Include relevant Windows event metadata in export")
        self.include_windows_events.setChecked(False)
        self.include_windows_events.setToolTip(
            "Opt in to display, watchdog and application error metadata. Unrelated event records are excluded."
        )
        export_row.addWidget(self.include_windows_events, 1)
        self.export_diagnostics_button = QPushButton("Export Diagnostics…")
        self.export_diagnostics_button.clicked.connect(self.export_diagnostics)
        export_row.addWidget(self.export_diagnostics_button)
        layout.addLayout(export_row)
        self.diagnostics_export_status = QLabel("")
        self.diagnostics_export_status.setWordWrap(True)
        layout.addWidget(self.diagnostics_export_status)
        note = QLabel(
            "Unsupported metrics show unknown. An application cannot reliably record events after Windows freezes."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        self.tabs.addTab(self.diagnostics_tab, "Settings / Diagnostics")

    def _preference_key(self, name: str) -> str:
        return f"{self._preferences_prefix}/{name}"

    def _initial_backend(self, name: str, configured: str, combo: QComboBox) -> str:
        """Use a saved installed backend, otherwise retain config.toml's default."""
        saved = self._preferences.value(self._preference_key(name), "", type=str)
        if saved and combo.findData(saved) >= 0:
            return saved
        if combo.findData(configured) >= 0:
            return configured
        return str(combo.itemData(0) or "")

    def _remember_backend(self, name: str, backend: str | None) -> None:
        """Persist a new GUI selection immediately, including before Start."""
        if not backend:
            return
        self._preferences.setValue(self._preference_key(name), str(backend))
        self._preferences.sync()
        if self._preferences.status() != QSettings.Status.NoError:
            log.warning("Could not save GUI backend preference %s", name)

    def _load_feature_settings(self):
        # Ensure the package source directory is on sys.path so registry can be imported.
        # When installed as an editable package from a different project location, the
        # registry module may not be available. Add the directory containing this module
        # to sys.path so the import succeeds.
        _pkg_dir = Path(__file__).resolve().parent
        _pkg_src = _pkg_dir.parent
        if str(_pkg_src) not in sys.path:
            sys.path.insert(0, str(_pkg_src))
        try:
            cfg = load_config(self.root / "config.toml")
            # Populate backend dropdowns from registry
            from auto_translate.registry import get_default_registry
            registry = get_default_registry()
            backends = registry.list_backends()
            # ASR backends
            self.asr_backend.clear()
            for name in backends.get("asr", []):
                caps = registry.get_capabilities("asr", name)
                label = f"{name} — {caps.description}" if caps else name
                self.asr_backend.addItem(label, name)
            # Translation backends
            self.mt_backend.clear()
            for name in backends.get("translate", []):
                caps = registry.get_capabilities("translate", name)
                label = f"{name} — {caps.description}" if caps else name
                self.mt_backend.addItem(label, name)
            # Restore last selected backends only if they still exist in the
            # registry; otherwise fall back to the configured defaults.
            asr_backend_name = self._initial_backend(
                "asr_backend", cfg.models.asr_backend, self.asr_backend,
            )
            self.asr_backend.setCurrentIndex(self.asr_backend.findData(asr_backend_name))
            mt_backend_name = self._initial_backend(
                "translate_backend", cfg.models.translate_backend, self.mt_backend,
            )
            self.mt_backend.setCurrentIndex(self.mt_backend.findData(mt_backend_name))
            # Model paths: use resolve_model_identity for accurate per-backend resolution.
            # Never fall back to another backend's default (e.g., Haas path for Whisper).
            asr_identity = resolve_model_identity(cfg, "asr", asr_backend_name)
            mt_identity = resolve_model_identity(cfg, "translate", mt_backend_name)
            self.asr_model.setText(asr_identity.get("model_path", ""))
            self.mt_model.setText(mt_identity.get("model_path", ""))
            self.context_lines.setValue(cfg.translate.context_lines)
            self.safe_playback_checkbox.setChecked(cfg.safe_playback.enabled)
            self.diagnostics_checkbox.setChecked(cfg.diagnostics.enabled)
            # Store config reference for backend switch handler
            self._cfg = cfg
            # Initialize tracked backends so the first switch saves the previous path
            self._last_asr_backend = asr_backend_name
            self._last_mt_backend = mt_backend_name
            # Connect backend dropdown changes to update model paths
            self.asr_backend.currentIndexChanged.connect(self._on_asr_backend_changed)
            self.mt_backend.currentIndexChanged.connect(self._on_mt_backend_changed)
        except Exception:
            log.exception("Could not load GUI feature settings; using defaults")
            # Ensure dropdowns are populated even if config loading fails
            try:
                from auto_translate.registry import get_default_registry
                registry = get_default_registry()
                backends = registry.list_backends()
                self.asr_backend.clear()
                for name in backends.get("asr", []):
                    caps = registry.get_capabilities("asr", name)
                    label = f"{name} — {caps.description}" if caps else name
                    self.asr_backend.addItem(label, name)
                self.mt_backend.clear()
                for name in backends.get("translate", []):
                    caps = registry.get_capabilities("translate", name)
                    label = f"{name} — {caps.description}" if caps else name
                    self.mt_backend.addItem(label, name)
            except Exception:
                log.exception("Failed to populate backend dropdowns")

    def _on_asr_backend_changed(self, index: int) -> None:
        """Update ASR model path when backend selection changes.

        Stores the current backend's model path before switching, then loads
        the new backend's resolved model identity. Uses resolve_model_identity
        so unconfigured backends show empty path, not another backend's model.
        """
        new_backend = self.asr_backend.itemData(index)

        # Save the PREVIOUS backend's model path before switching
        old_backend = self._last_asr_backend
        if old_backend is not None:
            current_text = self.asr_model.text().strip()
            old_identity = resolve_model_identity(self._cfg, "asr", old_backend)
            old_default = old_identity.get("model_path", "")
            if current_text and current_text != old_default:
                self._asr_model_overrides[str(old_backend)] = current_text
            else:
                # User cleared the field or restored the default — remove stale override
                self._asr_model_overrides.pop(str(old_backend), None)

        # Load the new backend's model path
        if new_backend is not None and hasattr(self, "_cfg"):
            identity = resolve_model_identity(self._cfg, "asr", new_backend)
            model_path = identity.get("model_path", "")
            # Use stored override if available, otherwise use resolved identity
            stored = self._asr_model_overrides.get(str(new_backend))
            if stored:
                model_path = stored
            self.asr_model.setText(model_path)

        # Update tracked backend for next switch and persist the user's choice.
        self._last_asr_backend = new_backend
        self._remember_backend("asr_backend", new_backend)

    def _on_mt_backend_changed(self, index: int) -> None:
        """Update translation model path when backend selection changes.

        Stores the current backend's model path before switching, then loads
        the new backend's resolved model identity. Uses resolve_model_identity
        so unconfigured backends show empty path, not another backend's model.
        """
        new_backend = self.mt_backend.itemData(index)

        # Save the PREVIOUS backend's model path before switching
        old_backend = self._last_mt_backend
        if old_backend is not None:
            current_text = self.mt_model.text().strip()
            old_identity = resolve_model_identity(self._cfg, "translate", old_backend)
            old_default = old_identity.get("model_path", "")
            if current_text and current_text != old_default:
                self._mt_model_overrides[str(old_backend)] = current_text
            else:
                # User cleared the field or restored the default — remove stale override
                self._mt_model_overrides.pop(str(old_backend), None)

        # Load the new backend's model path
        if new_backend is not None and hasattr(self, "_cfg"):
            identity = resolve_model_identity(self._cfg, "translate", new_backend)
            model_path = identity.get("model_path", "")
            # Use stored override if available, otherwise use resolved identity
            stored = self._mt_model_overrides.get(str(new_backend))
            if stored:
                model_path = stored
            self.mt_model.setText(model_path)

        # Update tracked backend for next switch and persist the user's choice.
        self._last_mt_backend = new_backend
        self._remember_backend("translate_backend", new_backend)

    def _get_asr_model_override(self) -> str | None:
        """Return the ASR model path only if it differs from the backend resolved identity."""
        backend_name = self.asr_backend.currentData()
        if not backend_name or not hasattr(self, "_cfg"):
            return None
        current_text = self.asr_model.text().strip()
        if not current_text:
            return None
        identity = resolve_model_identity(self._cfg, "asr", backend_name)
        default_path = identity.get("model_path", "")
        if current_text == default_path:
            return None  # using resolved default, no override needed
        return current_text

    def _get_mt_model_override(self) -> str | None:
        """Return the MT model path only if it differs from the backend resolved identity."""
        backend_name = self.mt_backend.currentData()
        if not backend_name or not hasattr(self, "_cfg"):
            return None
        current_text = self.mt_model.text().strip()
        if not current_text:
            return None
        identity = resolve_model_identity(self._cfg, "translate", backend_name)
        default_path = identity.get("model_path", "")
        if current_text == default_path:
            return None  # using resolved default, no override needed
        return current_text


    def _update_diagnostics(self, event):
        for name, value in diagnostic_rows(event).items():
            self.diagnostic_labels[name].setText(value)
        self.diagnostics_observation_status.setText(
            "Latest backend observations." if event.get("event") == "diagnostics"
            else "Waiting for backend observations."
        )

    def export_diagnostics(self):
        if self._export_thread is not None and self._export_thread.is_alive():
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export application diagnostics", str(self.root / "diagnostics.zip"), "ZIP archive (*.zip)"
        )
        if not destination:
            return
        include_events = self.include_windows_events.isChecked()
        self.export_diagnostics_button.setEnabled(False)
        self.diagnostics_export_status.setText("Collecting application diagnostics…")

        def collect():
            try:
                from auto_translate.runtime_diagnostics import export_diagnostics
                result = export_diagnostics(Path(destination), log_dir=self.root / "logs",
                                            include_windows_events=include_events)
            except Exception as exc:  # noqa: BLE001
                if not self._closing:
                    self.diagnosticsExported.emit("", str(exc))
            else:
                if not self._closing:
                    self.diagnosticsExported.emit(str(result), "")

        self._export_thread = threading.Thread(target=collect, name="diagnostics-export", daemon=True)
        self._export_thread.start()

    def _on_diagnostics_exported(self, path, error):
        self.export_diagnostics_button.setEnabled(True)
        self.diagnostics_export_status.setText(
            f"Diagnostics export failed: {error}" if error else f"Diagnostics saved to {path}"
        )

    def paste_url(self):
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_input.setText(text)

    def _lifecycle(self, message):
        line = f"session={self._session} {message}"
        self._write_console(line + "\n")
        try:
            with open_rotating_log(self.root / "logs" / "gui-lifecycle.log") as stream:
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
        self._backend_finishing = False
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
            # The socket also closes during normal VOD completion. Do not mark
            # this as a user stop; let QProcess.finished determine success/failure.
            self._control_lost = True
            self.status.setText("Player controls disconnected; waiting for backend exit.")
            self._write_console("GUI: control channel closed; awaiting backend exit status.\n")
            session = self._session
            QTimer.singleShot(3000, lambda: self._resolve_control_disconnect(session))

    def _resolve_control_disconnect(self, session):
        # A real channel failure must still stop an orphaned backend. A normal
        # video end emits session_ending before closing its command socket.
        if (self._session != session or self.process is None or self._stopping
                or self._backend_finishing or not self._control_lost):
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._write_console("GUI: unexpected control loss; stopping backend.\n")
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
                asr_model=self._get_asr_model_override(),
                mt_model=self._get_mt_model_override(),
                asr_backend=self.asr_backend.currentData(),
                translate_backend=self.mt_backend.currentData(),
                context_lines=self.context_lines.value(),
                safe_playback=self.safe_playback_checkbox.isChecked(),
                diagnostics=self.diagnostics_checkbox.isChecked(),
            )
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
            if os.name == "nt":
                command, args = session_launcher(command, args)
        except (OSError, ValueError) as exc:
            self._close_control()
            if self._job:
                self._job.close()
                self._job = None
            logging.exception("session=%s launch/containment failed", self._session)  # noqa: LOG015
            if (self.root / "logs").is_dir():
                with open_rotating_log(self.root / "logs" / "run.log") as stream:
                    print(f"session={self._session} launch/containment failed: {exc}", file=stream)
            QMessageBox.critical(self, "Cannot start translation", str(exc))
            return

        self.console.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_line = ""
        self._session_error = ""
        self._ready = False
        self._stopping = False
        self._reset_player_controls()
        self._update_diagnostics({})
        self.feature_settings.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Initializing models and loading video... See Console for progress.")
        try:
            self._log_stream = open_rotating_log(self.root / "logs" / "run.log")
        except OSError:
            self._log_stream = None
        proc = self._new_process()
        self.process = proc
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessEnvironment(qenv)
        proc.setProgram(command)
        proc.setArguments(args)
        self._lifecycle(f"Starting translation backend gui_pid={os.getpid()}")
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
        self._reset_player_activity()
        self._player_activity_timer.start()
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
        self._player_activity_timer.stop()
        self._reset_player_activity()
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
        if action == "diagnostics":
            self._update_diagnostics(event)
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
        if action == "session_ending":
            self._backend_finishing = True
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
            self._player_activity_at = time.monotonic()

    def _seek_preview(self, value: int):
        if self._seeking and self._duration_s is not None:
            self.time_label.setText(
                f"{self._format_time(value)} / {self._format_time(self._duration_s)}"
            )

    def _request_seek(self, value: float):
        """Submit a completed timeline action; never seek live streams."""
        if self._fullscreen:
            self.video_controls.show()
            self._player_activity_at = time.monotonic()
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
            self._player_activity_at = time.monotonic()
        self._request_seek(float(self.seek_slider.value()))

    def _reset_player_activity(self):
        self._player_cursor_pos = QCursor.pos()
        self._player_activity_at = time.monotonic()
        self.video_surface.unsetCursor()

    def _player_activity_tick(self):
        now = time.monotonic()
        cursor = QCursor.pos()
        over_video = (
            self._ready and not self._stopping and not self._closing
            and self.isActiveWindow() and self.video_surface.isVisible()
            and QApplication.activeModalWidget() is None
            and QApplication.activePopupWidget() is None
            and QApplication.widgetAt(cursor) is self.video_surface
        )
        interacting = (QApplication.mouseButtons() != Qt.MouseButton.NoButton
                       or self.seek_slider.isSliderDown() or self._seeking)
        if cursor != self._player_cursor_pos or not over_video or interacting:
            self._player_activity_at = now
            self.video_surface.unsetCursor()
            if self._fullscreen:
                self.video_controls.show()
        elif now - self._player_activity_at >= self.PLAYER_IDLE_TIMEOUT_S:
            if self.video_surface.cursor().shape() != Qt.CursorShape.BlankCursor:
                self.video_surface.setCursor(Qt.CursorShape.BlankCursor)
            if self._fullscreen:
                self.video_controls.hide()
        self._player_cursor_pos = cursor

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
        self._reset_player_activity()

    def exit_fullscreen(self):
        if not self._fullscreen:
            return
        self._fullscreen = False
        self._reset_player_activity()
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
        self._player_activity_timer.stop()
        self._reset_player_activity()
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
        self.feature_settings.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._ready = False
        self._stopping = False
        self.status.setText(reason)
        self.diagnostics_observation_status.setText("Session ended; showing last observations.")
        if self._closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent):
        self._closing = True
        self._player_activity_timer.stop()
        self._reset_player_activity()
        if self.process is not None:
            self.stop_translation()
            event.ignore()
            return
        self._close_control()
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
