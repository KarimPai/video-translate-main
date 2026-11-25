"""GUI remembers last selected ASR and MT, independently per project."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from auto_translate.gui import VideoTranslateWindow

_CONFIG = """[models]
asr_backend = "haas"
translate_backend = "gemma4"

[asr]
model_path = "C:/models/haas"

[asr.whisper]
model_path = "C:/models/whisper"

[translate]
model_path = "C:/models/gemma4"

[translate.plamo]
model_path = "C:/models/plamo"

[translate.hymt3]
model_path = "models/hy-mt/Hy-MT3-1.8B-Q8_0.gguf"
"""


def _make_window(tmp_path):
    QApplication.instance() or QApplication([])
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config.toml"
    if not config_file.exists():
        config_file.write_text(_CONFIG, encoding="utf-8")
    window = VideoTranslateWindow(tmp_path)
    return window


def test_last_picked_asr_and_translator_survive_gui_reopen(tmp_path):
    root = tmp_path / "universal"
    first = _make_window(root)
    prefs = first._preferences
    prefix = first._preferences_prefix
    original = (root / "config.toml").read_bytes()
    try:
        assert first.asr_backend.currentData() == "haas"
        assert first.mt_backend.currentData() == "gemma4"

        first.asr_backend.setCurrentIndex(first.asr_backend.findData("whisper"))
        first.mt_backend.setCurrentIndex(first.mt_backend.findData("hymt3"))
        assert first.asr_model.text() == "C:/models/whisper"
        assert first.mt_model.text() == "models/hy-mt/Hy-MT3-1.8B-Q8_0.gguf"
        assert first.mt_backend.currentText().startswith("Hy-MT3 1.8B")
        # Saves on change; no translation needs to start.
        assert prefs.value(first._preference_key("asr_backend")) == "whisper"
        assert prefs.value(first._preference_key("translate_backend")) == "hymt3"
        assert (root / "config.toml").read_bytes() == original

        first.close()
        second = _make_window(root)
        try:
            assert second.asr_backend.currentData() == "whisper"
            assert second.mt_backend.currentData() == "hymt3"
            assert second._last_asr_backend == "whisper"
            assert second._last_mt_backend == "hymt3"
            assert second.asr_model.text() == "C:/models/whisper"
            assert second.mt_model.text() == "models/hy-mt/Hy-MT3-1.8B-Q8_0.gguf"
            # Last picked (even before pressing Start) should be remembered.
            second.asr_backend.setCurrentIndex(second.asr_backend.findData("haas"))
            assert prefs.value(second._preference_key("asr_backend")) == "haas"
            assert prefs.value(second._preference_key("translate_backend")) == "hymt3"
        finally:
            second.close()
        third = _make_window(root)
        try:
            assert third.asr_backend.currentData() == "haas"
            assert third.mt_backend.currentData() == "hymt3"
        finally:
            third.close()
    finally:
        first.close()
        prefs.remove(prefix)
        prefs.sync()


def test_invalid_saved_backends_fall_back_to_config_without_crashing(tmp_path):
    root = tmp_path / "universal"
    first = _make_window(root)
    prefs = first._preferences
    prefix = first._preferences_prefix
    try:
        prefs.setValue(first._preference_key("asr_backend"), "deleted-asr-model")
        prefs.setValue(first._preference_key("translate_backend"), "deleted-translator")
        prefs.sync()
        first.close()
        reopened = _make_window(root)
        try:
            assert reopened.asr_backend.currentData() == "haas"
            assert reopened.mt_backend.currentData() == "gemma4"
            assert reopened.asr_model.text() == "C:/models/haas"
            assert reopened.mt_model.text() == "C:/models/gemma4"
        finally:
            reopened.close()
    finally:
        first.close()
        prefs.remove(prefix)
        prefs.sync()


def test_gui_selections_are_scoped_to_project_folder(tmp_path):
    a = _make_window(tmp_path / "project-a")
    b = None
    try:
        a.asr_backend.setCurrentIndex(a.asr_backend.findData("whisper"))
        a.mt_backend.setCurrentIndex(a.mt_backend.findData("plamo"))
        b = _make_window(tmp_path / "project-b")
        assert a._preferences_prefix != b._preferences_prefix
        assert b.asr_backend.currentData() == "haas"
        assert b.mt_backend.currentData() == "gemma4"
    finally:
        for window in (a, b):
            if window is not None:
                window.close()
                window._preferences.remove(window._preferences_prefix)
                window._preferences.sync()
