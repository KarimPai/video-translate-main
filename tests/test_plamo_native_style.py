"""PLaMo uses its native style without changing other translation settings."""
import pytest

from auto_translate.config import Config, TranslateConfig, get_backend_config


@pytest.mark.parametrize("backend", ["plamo", "plamo8b"])
def test_plamo_native_style_preserves_translation_settings(backend):
    cfg = Config()
    cfg.translate_profiles[backend] = {
        "context_lines": 9,
        "temperature": 0.3,
        "style": "Custom style",
    }
    effective = get_backend_config(cfg, "translate", backend)
    assert effective["style"] == ""
    assert effective["context_lines"] == 9
    assert effective["temperature"] == 0.3
    assert effective["top_k"] == cfg.translate.top_k
    assert effective["top_p"] == cfg.translate.top_p
    assert effective["repeat_penalty"] == cfg.translate.repeat_penalty
    assert effective["glossary"] == cfg.translate.glossary
    assert effective["remove_fillers"] == cfg.translate.remove_fillers
    assert effective["collapse_repetitions"] == cfg.translate.collapse_repetitions
    TranslateConfig(**effective)


@pytest.mark.parametrize("backend", ["gemma4", "hymt"])
def test_other_backends_keep_style(backend):
    cfg = Config()
    assert get_backend_config(cfg, "translate", backend)["style"] == cfg.translate.style
