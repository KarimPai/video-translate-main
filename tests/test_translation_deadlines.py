"""CPU-only checks for absolute translation timeout propagation."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

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