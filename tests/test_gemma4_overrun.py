"""Regression tests for extreme Gemma contextual over-translation.

All tests are CPU-only and mock the llama-server; no model is loaded.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from auto_translate.config import TranslateConfig
from auto_translate.translate.gemma4 import Gemma4Translator, translation_is_overlong


def test_translation_overrun_threshold_targets_extreme_context_echo():
    ja = "30万円も出してくれるの? 嬉しい じゃあこれ"
    leaked = "You'll pay a lot of money? " + "Prior conversation unrelated to the current speech. " * 9
    assert translation_is_overlong(ja, leaked)
    assert not translation_is_overlong(ja, "You'll pay 300,000 yen? I'm happy! Well, then...")
    assert not translation_is_overlong("これ", "A very long but possible explanation. " * 12)


def test_gemma_retries_extreme_overrun_without_context_or_glossary():
    ja = "30万円も出してくれるの? 嬉しい じゃあこれ"
    leaked = "You'll pay a lot of money? " + "Prior conversation unrelated to the current speech. " * 9
    corrected = "You'll pay 300,000 yen? I'm happy! Well, then..."

    translator = Gemma4Translator.__new__(Gemma4Translator)
    translator._closed = False
    translator._lock = threading.RLock()
    translator._relief = threading.Event()
    translator.cfg = TranslateConfig()
    translator.server = SimpleNamespace(alive=lambda: True)
    translator._context = Mock(return_value="previous Japanese dialogue")
    translator._terms_for = Mock(return_value=([], [], []))
    translator._chat = Mock(side_effect=[(leaked, "stop"), (corrected, "stop")])

    actual = translator.translate(ja, [("前の会話", "previous dialogue")])

    assert actual == corrected
    assert translator._chat.call_count == 2
    retry_prompt = translator._chat.call_args_list[1].args[0]
    assert retry_prompt["messages"][-1]["content"] == ja


def test_gemma_never_publishes_overrun_if_retry_is_also_bad():
    ja = "30万円も出してくれるの? 嬉しい じゃあこれ"
    leaked = "Prior conversation unrelated to current speech. " * 12
    translator = Gemma4Translator.__new__(Gemma4Translator)
    translator._closed = False
    translator._lock = threading.RLock()
    translator._relief = threading.Event()
    translator.cfg = TranslateConfig()
    translator.server = SimpleNamespace(alive=lambda: True)
    translator._context = Mock(return_value="previous Japanese dialogue")
    translator._terms_for = Mock(return_value=([], [], []))
    translator._chat = Mock(side_effect=[(leaked, "stop"), (leaked, "stop")])

    assert translator.translate(ja, []) == ""
    assert translator._chat.call_count == 2
