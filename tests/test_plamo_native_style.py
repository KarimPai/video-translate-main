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


def test_hymt_keeps_common_style():
    cfg = Config()
    assert get_backend_config(cfg, "translate", "hymt")["style"] == cfg.translate.style


@pytest.mark.parametrize(
    ("profile", "overrides"),
    [
        ({}, None),
        ({"style": "Profile style"}, None),
        ({}, {"style": "Explicit style"}),
    ],
)
def test_gemma4_uses_native_style_without_inherited_guidance(profile, overrides):
    from auto_translate.translate.gemma4 import build_prompt

    cfg = Config()
    cfg.translate_profiles["gemma4"] = profile
    effective = get_backend_config(
        cfg, "translate", "gemma4", explicit_overrides=overrides,
        selected_backend="gemma4",
    )

    assert effective["style"] == ""
    prompt = build_prompt("こんにちは", [("ゆき", "Yuki")], "前の会話", effective["style"])
    user_content = prompt["messages"][1]["content"]
    assert "Translation style:" not in user_content
    assert "前の会話" in user_content
    assert "ゆき → Yuki" in user_content
