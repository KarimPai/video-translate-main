"""CPU-only checks for absolute translation timeout propagation."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from auto_translate import llama_server as llama_server_module
from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate import gemma4 as gemma4_module
from auto_translate.translate import hymt as hymt_module
from auto_translate.translate import plamo as plamo_module
from auto_translate.translate.base import (
    absolute_translation_deadline,
    remaining_translation_timeout,
    start_translation_server,
    with_translation_deadline,
)
from auto_translate.translate.gemma4 import Gemma4Translator
from auto_translate.translate.hymt import HymtTranslator
from auto_translate.translate.plamo import PlamoTranslator


@pytest.mark.parametrize(
    ("translator_type", "prompt"),
    [
        (Gemma4Translator, {"messages": []}),
        (PlamoTranslator, {"messages": []}),
        (HymtTranslator, "こんにちは"),
    ],
)
def test_chat_uses_remaining_end_to_end_timeout(translator_type, prompt):
    backend = object.__new__(translator_type)
    backend._closed = False
    backend.cfg = SimpleNamespace(
        max_tokens=128, temperature=0.0, top_k=20, top_p=0.6,
        repeat_penalty=1.05,
    )
    timeouts = []

    def chat(_body, *, timeout):
        timeouts.append(timeout)
        return {"choices": [{"message": {"content": "Hello."}, "finish_reason": "stop"}]}

    backend.server = SimpleNamespace(chat=chat)
    with absolute_translation_deadline(0.5):
        content, finish = backend._chat(prompt, timeout=30.0)

    assert content == "Hello."
    assert finish == "stop"
    assert 0 < timeouts[0] <= 0.5


def test_translation_retries_share_one_deadline():
    class Translator:
        cfg = SimpleNamespace(timeout_s=0.5)

        @with_translation_deadline
        def translate(self):
            first = remaining_translation_timeout(30.0)
            time.sleep(0.02)
            second = remaining_translation_timeout(30.0)
            return first, second

    first, second = Translator().translate()
    assert 0 < second < first <= 0.5


def test_expired_deadline_rejects_another_attempt():
    with absolute_translation_deadline(0.01):
        time.sleep(0.02)
        with pytest.raises(TimeoutError, match="translation deadline"):
            remaining_translation_timeout(30.0)


def test_server_recovery_receives_active_translation_deadline():
    captured = []
    server = SimpleNamespace(start=lambda **kwargs: captured.append(kwargs["deadline"]))
    with absolute_translation_deadline(1.0):
        start_translation_server(server)

    assert len(captured) == 1
    assert time.monotonic() < captured[0] <= time.monotonic() + 1.0


@pytest.mark.parametrize(
    ("translator_type", "module"),
    [
        (Gemma4Translator, gemma4_module),
        (HymtTranslator, hymt_module),
        (PlamoTranslator, plamo_module),
    ],
)
def test_llama_translators_enable_strict_http_deadlines(
    monkeypatch, tmp_path, translator_type, module,
):
    created = []

    class FakeServer:
        def __init__(self, *_args, **kwargs):
            self.strict_http_deadlines = kwargs.get("strict_http_deadlines", False)
            self.args = []
            created.append(self)

        def start(self, **_kwargs):
            return None

        def chat(self, _body, timeout):
            assert timeout > 0
            return {"choices": [{"message": {"content": "Hello."}, "finish_reason": "stop"}]}

        def close(self):
            return None

        def log_tail(self):
            return ""

    monkeypatch.setattr(module, "LlamaServer", FakeServer)
    monkeypatch.setattr(module, "require_gguf", lambda *_args: tmp_path / "model.gguf")
    monkeypatch.setattr(module, "gguf_block_count", lambda _path: 32)
    monkeypatch.setattr(module, "resolve_executable", lambda _path: tmp_path / "llama-server.exe")
    monkeypatch.setattr(module, "detect_device", lambda *_args: "none")
    monkeypatch.setattr(module, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "update_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "increment_counter", lambda *_args, **_kwargs: None)

    cfg = TranslateConfig(
        model_path="model.gguf", server_path="llama-server", context_size=1024,
        max_tokens=128, load_timeout_s=1.0, cuda_device=-1,
    )
    translator = translator_type(cfg, Glossary({}))
    try:
        assert created[-1].strict_http_deadlines is True
    finally:
        translator.close()


def test_deadline_http_client_enforces_total_wall_clock_timeout(monkeypatch):
    class SlowAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, *_args, **_kwargs):
            await asyncio.sleep(0.2)

    monkeypatch.setattr(llama_server_module.httpx, "AsyncClient", SlowAsyncClient)
    client = llama_server_module.DeadlineHttpClient()
    started = time.monotonic()
    with pytest.raises(httpx.TimeoutException, match="wall-clock deadline"):
        client.get("http://127.0.0.1:1/", timeout=0.02)
    assert time.monotonic() - started < 0.15


def test_server_restart_lock_acquisition_obeys_operation_deadline(tmp_path):
    server = llama_server_module.LlamaServer(
        "test", tmp_path / "llama-server.exe", [], startup_timeout_s=5.0,
    )
    server._lock.acquire()
    deadline = time.monotonic() + 0.03
    started = time.monotonic()
    try:
        with pytest.raises(llama_server_module.LlamaServerError, match="deadline"):
            server.reset_for_restart(deadline=deadline)
        assert time.monotonic() - started < 0.2
    finally:
        server._lock.release()
        server.close()