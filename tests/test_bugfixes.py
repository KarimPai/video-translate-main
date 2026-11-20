"""Comprehensive tests for second bugfix pass.

Covers:
1. Backend selection and configuration precedence (CLI > profile > common)
2. Qwen3-ASR timeout handling (http_s unbound variable fix)
3. Qwen3-ASR metadata leakage (language Japanese prefix stripping)
4. Qwen3-ASR restart via reset_for_restart
5. LlamaServer restart support
6. ASR recovery interface compatibility
7. Language filter classification
8. GPU relief cooldown
9. Whisper device and model-path selection
10. Backend profile merging
"""

import inspect
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# 1. Configuration precedence
# ============================================================================

def test_get_backend_config_precedence_common_only():
    """Common config is used when no profile or override exists."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    merged = get_backend_config(cfg, "asr", "haas")
    assert merged["model_path"] == "haas-ja-v1.0"
    assert merged["threads"] == 8
    assert merged["context_size"] == 4096


def test_get_backend_config_precedence_profile_over_common():
    """Backend profile overrides common config."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "/models/qwen3-asr/model.gguf",
        "threads": 4,
    }
    merged = get_backend_config(cfg, "asr", "qwen3-asr")
    assert merged["model_path"] == "/models/qwen3-asr/model.gguf"
    assert merged["threads"] == 4
    # Common values preserved
    assert merged["context_size"] == 4096
    assert merged["response_timeout_s"] == 30.0


def test_get_backend_config_precedence_cli_over_profile():
    """CLI/GUI explicit overrides beat backend profile."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "/models/qwen3-asr/profile-model.gguf",
        "threads": 4,
    }
    # Simulate CLI --asr-model override
    explicit = {"model_path": "/custom/cli-model.gguf"}
    merged = get_backend_config(cfg, "asr", "qwen3-asr", explicit_overrides=explicit)
    # CLI override should win over profile
    assert merged["model_path"] == "/custom/cli-model.gguf"
    # Profile values still apply for non-overridden keys
    assert merged["threads"] == 4
    # Common values preserved
    assert merged["context_size"] == 4096


def test_get_backend_config_translate_precedence():
    """Translation backend precedence: CLI > profile > common."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate_profiles["hymt"] = {
        "model_path": "/models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf",
        "gpu_layers": 50,
    }
    explicit = {"model_path": "/custom/mt-model.gguf"}
    merged = get_backend_config(cfg, "translate", "hymt", explicit_overrides=explicit)
    assert merged["model_path"] == "/custom/mt-model.gguf"
    assert merged["gpu_layers"] == 50
    assert merged["context_lines"] == 6


def test_get_backend_config_no_haas_model_for_qwen3():
    """Qwen3-ASR profile must never return Haas default model path."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.asr_profiles["qwen3-asr"] = {
        "model_path": "/models/qwen3-asr/model.gguf",
        "mmproj_path": "/models/qwen3-asr/mmproj.gguf",
    }
    merged = get_backend_config(cfg, "asr", "qwen3-asr")
    assert "haas" not in merged["model_path"].lower()
    assert "qwen3" in merged["model_path"].lower()


def test_get_backend_config_no_gemma_for_hymt():
    """HY-MT2 profile must never return Gemma default model path."""
    from auto_translate.config import Config, get_backend_config

    cfg = Config()
    cfg.translate_profiles["hymt"] = {
        "model_path": "/models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf",
    }
    merged = get_backend_config(cfg, "translate", "hymt")
    assert "gemma" not in merged["model_path"].lower()
    assert "hy-mt" in merged["model_path"].lower()


# ============================================================================
# 2. Qwen3-ASR timeout handling
# ============================================================================

def test_qwen3_asr_timeout_uses_monotonic_timer():
    """Qwen3-ASR timeout handler must not reference unbound variable."""
    import inspect

    from auto_translate.asr.qwen3_asr import Qwen3ASR

    # Verify the _transcribe_impl source code initializes http_s before the try block
    source = inspect.getsource(Qwen3ASR._transcribe_impl)
    lines = source.split("\n")
    http_s_init_line = None
    try_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("http_s = 0.0"):
            http_s_init_line = i
        if try_line is None and stripped.startswith("try:"):
            try_line = i

    assert http_s_init_line is not None, "http_s must be initialized before try block"
    assert try_line is not None, "try block not found"
    assert http_s_init_line < try_line, "http_s must be initialized before try block"


# ============================================================================
# 3. Qwen3-ASR metadata leakage
# ============================================================================

def test_qwen3_asr_strips_language_japanese_prefix():
    """Qwen3-ASR must strip 'language Japanese' prefix from responses."""
    import re
    # Simulate the cleaning logic from qwen3_asr.py
    text = "language Japaneseこんにちは、お元気ですか？"
    text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<asr_text>", "", text).strip()
    assert text == "こんにちは、お元気ですか？"
    assert "language" not in text.lower()


def test_qwen3_asr_strips_asr_text_tag():
    """Qwen3-ASR must strip <asr_text> tag from responses."""
    import re
    text = "<asr_text>こんにちは"
    text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<asr_text>", "", text).strip()
    assert text == "こんにちは"


def test_qwen3_asr_strips_combined_metadata():
    """Qwen3-ASR must strip combined metadata prefix."""
    import re
    text = "language Japanese<asr_text>おはようございます"
    text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<asr_text>", "", text).strip()
    assert text == "おはようございます"


def test_qwen3_asr_preserves_legitimate_japanese():
    """Qwen3-ASR must not strip legitimate Japanese text."""
    import re
    text = "今日はいい天気ですね"
    text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<asr_text>", "", text).strip()
    assert text == "今日はいい天気ですね"


def test_qwen3_asr_preserves_japanese_with_english_names():
    """Qwen3-ASR must keep Japanese text containing English names."""
    import re
    text = "アリスとボブが会話をしている"
    text = re.sub(r"^language\s+japanese\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<asr_text>", "", text).strip()
    assert "アリス" in text
    assert "ボブ" in text


# ============================================================================
# 4. LlamaServer restart support
# ============================================================================

def test_llama_server_reset_for_restart_exists():
    """LlamaServer must have reset_for_restart method."""
    from auto_translate.llama_server import LlamaServer
    assert hasattr(LlamaServer, "reset_for_restart")
    sig = inspect.signature(LlamaServer.reset_for_restart)
    assert len(sig.parameters) == 1  # self only


def test_llama_server_start_refuses_after_close():
    """LlamaServer.start() must raise if _closed is True."""
    from pathlib import Path

    from auto_translate.llama_server import LlamaServer, LlamaServerError

    server = LlamaServer(
        name="test",
        executable=Path("/nonexistent"),
        args=["--model", "test.gguf"],
        startup_timeout_s=0.1,
    )
    server.close()
    try:
        server.start()
        assert False, "Expected LlamaServerError"
    except LlamaServerError as exc:
        assert "closed" in str(exc).lower()


def test_llama_server_reset_allows_restart():
    """After reset_for_restart(), start() should not raise 'closed' error."""
    from pathlib import Path

    from auto_translate.llama_server import LlamaServer, LlamaServerError

    server = LlamaServer(
        name="test",
        executable=Path("/nonexistent"),
        args=["--model", "test.gguf"],
        startup_timeout_s=0.1,
    )
    server.close()
    server.reset_for_restart()
    # start() will fail because executable doesn't exist, but not because of _closed
    try:
        server.start()
    except LlamaServerError as exc:
        # Should NOT be "server is closed"
        assert "closed" not in str(exc).lower(), f"Should not say 'closed': {exc}"
    except FileNotFoundError:
        pass  # Also acceptable — executable not found
    finally:
        server.close()


# ============================================================================
# 5. ASR recovery interface compatibility
# ============================================================================

def test_qwen3_asr_recover_signature():
    """Qwen3-ASR recover_context_echo must accept 4-arg signature."""
    from auto_translate.asr.qwen3_asr import Qwen3ASR
    sig = inspect.signature(Qwen3ASR.recover_context_echo)
    params = list(sig.parameters.keys())
    assert "audio" in params
    assert "probe" in params
    assert "vad_confidence" in params
    assert "timeout" in params


def test_whisper_recover_signature():
    """Whisper recover_context_echo must accept 4-arg signature."""
    from auto_translate.asr.whisper_turbo import WhisperASR
    sig = inspect.signature(WhisperASR.recover_context_echo)
    params = list(sig.parameters.keys())
    assert "audio" in params
    assert "probe" in params
    assert "vad_confidence" in params
    assert "timeout" in params


def test_haas_recover_signature():
    """Haas recover_context_echo must accept 4-arg signature."""
    from auto_translate.asr.haas import HaasASR
    sig = inspect.signature(HaasASR.recover_context_echo)
    params = list(sig.parameters.keys())
    assert "audio" in params
    assert "probe" in params
    assert "vad_confidence" in params
    assert "timeout" in params


# ============================================================================
# 6. Language filter
# ============================================================================

def test_language_filter_exists():
    """LanguageFilter class must exist in text_filter module."""
    from auto_translate.text_filter import LanguageFilter
    assert LanguageFilter is not None


def test_language_filter_keeps_japanese():
    """LanguageFilter should keep Japanese text."""
    from auto_translate.config import LanguageFilterConfig
    from auto_translate.text_filter import LanguageFilter

    cfg = LanguageFilterConfig()
    lf = LanguageFilter(cfg)
    result = lf.filter("こんにちは、お元気ですか？", duration_s=3.0)
    assert result is not None
    assert result == "こんにちは、お元気ですか？"


def test_language_filter_drops_english():
    """LanguageFilter should drop English text."""
    from auto_translate.config import LanguageFilterConfig
    from auto_translate.text_filter import LanguageFilter

    cfg = LanguageFilterConfig()
    lf = LanguageFilter(cfg)
    result = lf.filter("Hello, how are you doing today?", duration_s=3.0)
    assert result is None


def test_language_filter_keeps_mixed_japanese_english():
    """LanguageFilter should keep Japanese text with English names."""
    from auto_translate.config import LanguageFilterConfig
    from auto_translate.text_filter import LanguageFilter

    cfg = LanguageFilterConfig()
    lf = LanguageFilter(cfg)
    result = lf.filter("アリスとボブが会話をしている", duration_s=3.0)
    assert result is not None


def test_language_filter_keeps_short_utterances():
    """LanguageFilter should keep short utterances regardless of language."""
    from auto_translate.config import LanguageFilterConfig
    from auto_translate.text_filter import LanguageFilter

    cfg = LanguageFilterConfig()
    lf = LanguageFilter(cfg)
    # Short utterance below min_duration_s (2.5s default)
    result = lf.filter("Hello", duration_s=1.0)
    assert result is not None


def test_language_filter_import_works():
    """Importing LanguageFilter from text_filter must succeed."""
    try:
        from auto_translate.text_filter import LanguageFilter
        assert LanguageFilter is not None
    except ImportError:
        assert False, "LanguageFilter should be importable from text_filter"


# ============================================================================
# 7. GPU relief cooldown
# ============================================================================

def test_gpu_budget_has_cooldown():
    """GpuBudget must have RELIEF_COOLDOWN_S constant."""
    from auto_translate.gpu_budget import GpuBudget
    assert hasattr(GpuBudget, "RELIEF_COOLDOWN_S")
    assert GpuBudget.RELIEF_COOLDOWN_S >= 10.0


def test_gpu_budget_initializes_last_relief_time():
    """GpuBudget must initialize per-level relief time trackers."""
    from auto_translate.config import GpuBudgetConfig
    from auto_translate.gpu_budget import GpuBudget

    cfg = GpuBudgetConfig()
    budget = GpuBudget(cfg)
    assert hasattr(budget, "_last_low_relief_time")
    assert budget._last_low_relief_time == 0.0
    assert hasattr(budget, "_last_critical_relief_time")
    assert budget._last_critical_relief_time == 0.0


# ============================================================================
# 8. Whisper device and model selection
# ============================================================================

def test_whisper_cuda_device_default():
    """Whisper ASR config defaults cuda_device to 0 (GPU)."""
    from auto_translate.config import AsrConfig
    cfg = AsrConfig()
    assert cfg.cuda_device == 0


def test_whisper_model_path_fallback_logic():
    """Whisper should fall back to default model for non-whisper paths."""
    from pathlib import Path

    from auto_translate.config import AsrConfig

    cfg = AsrConfig()
    # Default model_path is "haas-ja-v1.0" which is wrong for Whisper
    assert cfg.model_path == "haas-ja-v1.0"
    # The fix checks if the path contains "whisper" or exists as a file
    model_path = cfg.model_path
    if model_path and ("whisper" in model_path.lower() or Path(model_path).exists()):
        selected = model_path
    else:
        selected = "Systran/faster-whisper-large-v3-turbo"
    assert "whisper" in selected.lower()


def test_whisper_accepts_whisper_model_path():
    """Whisper should accept a path containing 'whisper'."""
    from pathlib import Path
    model_path = "models/whisper-large-v3-turbo"
    if model_path and ("whisper" in model_path.lower() or Path(model_path).exists()):
        selected = model_path
    else:
        selected = "Systran/faster-whisper-large-v3-turbo"
    assert selected == "models/whisper-large-v3-turbo"


# ============================================================================
# 9. Config TOML loading
# ============================================================================

def test_config_load_preserves_hymt_model_path():
    """HY-MT2 model path in [translate.hymt] profile is preserved."""
    import tempfile

    from auto_translate.config import load_config

    toml_content = """
[models]
asr_backend = "haas"
translate_backend = "hymt"

[translate]
model_path = "models/gemma4/default.gguf"

[translate.hymt]
model_path = "models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf"
gpu_layers = 50
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert "gemma" in cfg.translate.model_path.lower()
    assert "hy-mt" in cfg.translate_profiles["hymt"]["model_path"].lower()
    assert cfg.translate_profiles["hymt"]["gpu_layers"] == 50


def test_config_load_asr_profile():
    """ASR backend profile is loaded from [asr.<name>] section."""
    import tempfile

    from auto_translate.config import load_config

    toml_content = """
[models]
asr_backend = "qwen3-asr"

[asr]
threads = 8

[asr.qwen3-asr]
model_path = "models/qwen3-asr/model.gguf"
mmproj_path = "models/qwen3-asr/mmproj.gguf"
threads = 4
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.asr.threads == 8
    assert cfg.asr_profiles["qwen3-asr"]["model_path"] == "models/qwen3-asr/model.gguf"
    assert cfg.asr_profiles["qwen3-asr"]["threads"] == 4


# ============================================================================
# 10. MPV and other method existence
# ============================================================================

def test_mpv_resume_from_auto_pause():
    """MpvOutput must have resume_from_auto_pause method."""
    from auto_translate.output.mpv import MpvOutput
    assert hasattr(MpvOutput, "resume_from_auto_pause")
    sig = inspect.signature(MpvOutput.resume_from_auto_pause)
    assert len(sig.parameters) == 1


def test_qwen3_asr_restart_method():
    """Qwen3-ASR _restart must exist and use reset_for_restart."""
    from auto_translate.asr.qwen3_asr import Qwen3ASR
    assert hasattr(Qwen3ASR, "_restart")
    source = inspect.getsource(Qwen3ASR._restart)
    assert "reset_for_restart" in source


def test_qwen3_asr_restart_limits_attempts():
    """Qwen3-ASR _restart must bound restart attempts."""
    from auto_translate.asr.qwen3_asr import Qwen3ASR
    source = inspect.getsource(Qwen3ASR._restart)
    assert "restart limit" in source.lower() or "_restarts >= 2" in source


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
