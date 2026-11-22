"""Regression coverage for the November 22 repository audit.

All tests are CPU-only and must not start real inference servers.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auto_translate.asr.haas import HaasASR
from auto_translate.asr.qwen3_asr import Qwen3ASR, Qwen3ASRError
from auto_translate.config import Config, get_backend_config, load_config
from auto_translate.glossary import Glossary
from auto_translate.translate.gemma4 import Gemma4Translator
from auto_translate.translate.hymt import HymtTranslator
from auto_translate.translate.plamo import PlamoTranslator


def test_miko_and_mikochi_resolve_to_sakura_miko():
    glossary = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")
    canonical = glossary._identity("さくらみこ", "Sakura Miko")
    assert canonical == "さくらみこ"
    assert glossary._identity("みこ", "Miko") == canonical
    assert glossary._identity("みこち", "Mikochi") == canonical
    assert ("みこち", "Mikochi") in glossary.name_terms("みこちだよ", context="さくらみこ")
    assert not any(en == "Sora" for _, en in glossary.name_terms("みこちだよ", context="さくらみこ"))


@pytest.mark.parametrize("backend", [PlamoTranslator, Gemma4Translator, HymtTranslator])
def test_spoken_glossary_term_survives_sixteen_contextual_names(backend):
    class DenseContext:
        def name_terms(self, text, *, context=""):
            if text == "先ほどの会話":
                return [(f"人物{i}", f"Person{i}") for i in range(16)]
            return []

        def find(self, text, *, context=""):
            return [("用語", "special term")]

    translator = backend.__new__(backend)  # No model or server startup
    translator.glossary = DenseContext()
    terms, all_names, utterance_names = translator._terms_for("重要な用語", "先ほどの会話")
    assert len(terms) <= 16
    assert len(all_names) == 16
    assert utterance_names == []
    assert ("用語", "special term") in terms


def test_haas_does_not_claim_to_apply_unavailable_context(caplog):
    asr = HaasASR.__new__(HaasASR)  # No speech.cpp process
    asr._state_lock = threading.Lock()
    asr._context_warning_logged = False
    asr.context = ""
    asr.supports_context_biasing = False
    with caplog.at_level(logging.WARNING):
        asr.set_context("みこち")
    assert not asr.context
    assert "unsupported" in caplog.text


def test_unknown_common_config_setting_emits_warning(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[asr]\nresponse_timeot_s = 7\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        config = load_config(config_path)
    assert config.asr.response_timeout_s != 7
    assert "response_timeot_s" in caplog.text


def test_mmproj_override_is_scoped_to_selected_asr():
    cfg = Config()
    overrides = {"mmproj_path": "models/selected-mmproj.gguf"}
    selected = get_backend_config(
        cfg, "asr", "qwen3-asr", explicit_overrides=overrides,
        selected_backend="qwen3-asr")
    other = get_backend_config(
        cfg, "asr", "haas", explicit_overrides=overrides,
        selected_backend="qwen3-asr")
    assert selected["mmproj_path"] == "models/selected-mmproj.gguf"
    assert other.get("mmproj_path") != overrides["mmproj_path"]


def test_qwen3_truncation_is_retried_before_acceptance():
    responses = [
        {"choices": [{"message": {"content": "language Japanese<asr_text>途中"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "language Japanese<asr_text>ありがとう"}, "finish_reason": "stop"}]},
    ]
    def post(url, *, json, timeout):
        calls.append(json["max_tokens"])
        response = responses.pop(0)
        return SimpleNamespace(
            json=lambda: response,
            raise_for_status=lambda: None,
        )

    calls = []
    asr = Qwen3ASR.__new__(Qwen3ASR)  # No model loading
    asr.server = SimpleNamespace(base_url="http://127.0.0.1:0",
                                  client=SimpleNamespace(post=post))
    result = asr._transcribe_impl(np.zeros(16000, dtype=np.float32), timeout=10)
    assert result == "ありがとう"
    assert calls == [256, 512]


def test_qwen3_rejects_second_truncated_response():
    asr = Qwen3ASR.__new__(Qwen3ASR)
    asr.server = SimpleNamespace(
        base_url="http://127.0.0.1:0",
        client=SimpleNamespace(post=lambda *a, **kw: SimpleNamespace(
            json=lambda: {"choices": [{"message": {"content": "途中"}, "finish_reason": "length"}]},
            raise_for_status=lambda: None,
        )),
    )
    with pytest.raises(Qwen3ASRError, match="truncated"):
        asr._transcribe_impl(np.zeros(16000, dtype=np.float32), timeout=10)


def test_gui_control_close_does_not_mark_natural_end_as_stopped():
    pytest.importorskip("PySide6")
    from auto_translate.gui import VideoTranslateWindow

    status = Mock()
    fake = SimpleNamespace(
        _close_control=Mock(),
        _pending_seek_target=12.0,
        _update_seek_buttons=Mock(),
        _control_lost=False,
        _stopping=False,
        _session="session-a",
        _backend_finishing=True,
        status=status,
        _write_console=Mock(),
        _resolve_control_disconnect=Mock(),
        stop_translation=Mock(),
    )
    with patch("auto_translate.gui.QTimer.singleShot") as timer:
        VideoTranslateWindow._control_disconnected(fake)
    assert fake._control_lost is True
    fake.stop_translation.assert_not_called()
    timer.assert_called_once()
    assert "waiting for backend exit" in status.setText.call_args.args[0]


def test_unexpected_gui_control_loss_stops_backend():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QProcess
    from auto_translate.gui import VideoTranslateWindow

    fake = SimpleNamespace(
        _session="session-a",
        process=SimpleNamespace(state=lambda: QProcess.ProcessState.Running),
        _stopping=False,
        _backend_finishing=False,
        _control_lost=True,
        _write_console=Mock(),
        stop_translation=Mock(),
    )
    VideoTranslateWindow._resolve_control_disconnect(fake, "session-a")
    fake.stop_translation.assert_called_once()

    fake.stop_translation.reset_mock()
    fake._backend_finishing = True
    VideoTranslateWindow._resolve_control_disconnect(fake, "session-a")
    fake.stop_translation.assert_not_called()


def test_expired_cached_live_subtitle_releases_pending_coverage():
    import queue

    from auto_translate.pipeline import run_translate
    from auto_translate.types import Subtitle

    incoming, outgoing = queue.Queue(), queue.Queue()
    completed = []
    incoming.put(Subtitle(0.0, 1.0, "こんにちは", "Hello.",
                          generation=0, speech_id=42))
    incoming.put(None)
    run_translate(
        None, 0, incoming, outgoing, live=True,
        playback_position=lambda: 20.0,
        complete_speech_cb=lambda start, end, generation, *, speech_id:
            completed.append((start, end, generation, speech_id)),
    )
    assert completed == [(0.0, 1.0, 0, 42)]


@pytest.mark.parametrize("backend", [Gemma4Translator, HymtTranslator])
def test_oversized_translation_prompt_uses_simplified_request(backend):
    from auto_translate.config import TranslateConfig

    translator = backend.__new__(backend)
    translator._relief = threading.Event()
    translator._lock = threading.Lock()
    translator._closed = False
    translator.server = SimpleNamespace(alive=lambda: True)
    translator.cfg = TranslateConfig(context_size=512, max_tokens=128)
    translator._context = lambda history: "文脈" * 150
    translator._terms_for = lambda ja, context: ([], [], [])
    prompts = []
    def chat(prompt, *, timeout, **kwargs):
        prompts.append(prompt)
        return ("Hello.", "stop")
    translator._chat = chat

    output = translator.translate("こんにちは", [])
    assert output == "Hello."
    assert len(prompts) == 1
    assert "文脈" not in str(prompts[0])
