"""PLaMo backend defaults keep context without optional wording constraints."""
from types import SimpleNamespace

import pytest

from auto_translate.config import Config, TranslateConfig, get_backend_config
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import PlamoTranslator


@pytest.mark.parametrize("backend", ["plamo", "plamo8b"])
def test_plamo_defaults_override_shared_tuning_but_keep_context(backend):
    cfg = Config()
    cfg.translate.temperature = 0.7
    cfg.translate.repeat_penalty = 1.2
    cfg.translate_profiles[backend] = {
        "context_lines": 9, "max_context_chars": 1200,
        "max_tokens": 256, "top_p": 0.5, "remove_fillers": True,
    }
    effective = get_backend_config(cfg, "translate", backend)
    assert effective["plamo_context_only"] is True
    assert effective["temperature"] == 0.0
    assert effective["top_k"] == 0
    assert effective["top_p"] == 1.0
    assert effective["repeat_penalty"] == 1.0
    assert effective["style"] == ""
    assert not effective["remove_fillers"]
    assert not effective["collapse_repetitions"]
    assert effective["context_lines"] == 9
    assert effective["max_context_chars"] == 1200
    assert effective["max_tokens"] == 256
    TranslateConfig(**effective)


@pytest.mark.parametrize("backend", ["gemma4", "hymt"])
def test_other_backends_keep_shared_tuning(backend):
    cfg = Config()
    cfg.translate.temperature = 0.7
    effective = get_backend_config(cfg, "translate", backend)
    assert effective["temperature"] == 0.7
    assert effective["remove_fillers"]
    assert not effective["plamo_context_only"]


def test_plamo_legacy_profile_can_be_explicitly_enabled():
    cfg = Config()
    cfg.translate_profiles["plamo"] = {"plamo_context_only": False, "temperature": 0.4}
    effective = get_backend_config(cfg, "translate", "plamo")
    assert effective["temperature"] == 0.4
    assert not effective["plamo_context_only"]


def test_plamo_native_prompt_keeps_context_without_glossary():
    translator = object.__new__(PlamoTranslator)
    translator.cfg = SimpleNamespace(plamo_context_only=True)
    translator.glossary = Glossary({"ミク": "Miku"})
    translator._context = lambda history: history[0][0]

    prompt = translator.prompt_for("ミクはどこ？", [("前の会話", "")])
    assert prompt["messages"] == [{"role": "user", "content": "ミクはどこ？"}]
    assert prompt["chat_template_kwargs"] == {
        "source_lang": "ja", "target_lang": "en", "context": "前の会話",
    }
