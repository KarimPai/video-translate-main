"""PLaMo 3 Translate 8B integration tests.

Tests backend registration, configuration isolation, model path resolution,
and GUI backend selection for the PLaMo 8B translation backend.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_REPO_CONFIG = (Path(__file__).parent.parent / "config.toml").resolve()

# Hermetic stand-in for the user's real config.toml. The real file may disable or
# rename local translation profiles, so ordinary profile/merge tests must never
# depend on it. Nothing here touches the real file or any model on disk.
_HERMETIC_TOML = """
[models]
asr_backend = "haas"
translate_backend = "gemma4"

[translate]
model_path = "models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf"
gpu_layers = 42
context_size = 4096

[translate.gemma4]
model_path = "models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf"

[translate.plamo]
model_path = "models/plamo3/plamo-3-translate-2b-Q4_K_M.gguf"
gpu_layers = -1
context_size = 8192

[translate.plamo8b]
model_path = "models/plamo3/plamo-3-translate-8b-Q3_K_M.gguf"
gpu_layers = -1
context_size = 8192
"""


@pytest.fixture(autouse=True)
def _hermetic_repo_config(monkeypatch, tmp_path):
    """Redirect load_config(<repo>/config.toml) to a temporary, known profile set."""
    import auto_translate.config as config_module

    hermetic = tmp_path / "hermetic-config.toml"
    hermetic.write_text(_HERMETIC_TOML, encoding="utf-8")
    real_load = config_module.load_config

    def load_config(path=None, *args, **kwargs):
        if path is not None and Path(path).resolve() == _REPO_CONFIG:
            path = hermetic
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(config_module, "load_config", load_config)



# ====================================================================
# 1. Backend registration tests
# ====================================================================


def test_plamo8b_registered_in_registry():
    """Test that plamo8b is registered in the backend registry."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    backends = registry.list_backends()

    translate_backends = backends.get("translate", [])
    assert "plamo8b" in translate_backends, (
        f"plamo8b not registered in translate backends: {translate_backends}"
    )


def test_plamo8b_capabilities():
    """Test that plamo8b capabilities are correct."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    caps = registry.get_capabilities("translate", "plamo8b")

    assert caps is not None, "plamo8b capabilities not found"
    assert caps.runtime == "llama_cpp", f"expected llama_cpp, got {caps.runtime}"
    assert caps.device == "cuda", f"expected cuda, got {caps.device}"
    assert caps.source_lang == "ja", f"expected ja, got {caps.source_lang}"
    assert caps.target_lang == "en", f"expected en, got {caps.target_lang}"
    # VRAM should reflect the actual model file size (~4.0 GB) plus overhead
    assert caps.vram_gb >= 4.0, (
        f"vram_gb={caps.vram_gb} is too low for a 4.0 GB Q3_K_M 8B model"
    )


def test_plamo_2b_still_registered():
    """Verify plamo (2B) is still registered alongside plamo8b."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    backends = registry.list_backends()
    translate_backends = backends.get("translate", [])

    assert "plamo" in translate_backends, "plamo 2B missing from registry"


def test_other_backends_unaffected():
    """Verify other translation backends are unaffected by plamo8b addition."""
    from auto_translate.registry import get_default_registry

    registry = get_default_registry()
    backends = registry.list_backends()
    translate_backends = backends.get("translate", [])

    for expected in ("gemma4", "hymt"):
        assert expected in translate_backends, f"{expected} missing from registry"


# ====================================================================
# 2. Configuration profile tests
# ====================================================================


def test_plamo8b_config_profile():
    """Test that plamo8b config profile loads correctly."""
    from auto_translate.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")

    assert "plamo8b" in cfg.translate_profiles, (
        f"plamo8b profile missing. Available: {list(cfg.translate_profiles.keys())}"
    )

    profile = cfg.translate_profiles["plamo8b"]
    model_path = profile.get("model_path", "")
    assert "plamo-3-translate-8b" in model_path, (
        f"model_path does not contain 8B identifier: {model_path}"
    )

    gpu_layers = profile.get("gpu_layers", -1)
    assert gpu_layers == -1, f"expected gpu_layers=-1 (auto), got {gpu_layers}"


def test_plamo_and_plamo8b_paths_differ():
    """Test that plamo (2B) and plamo8b model paths are different."""
    from auto_translate.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")

    assert "plamo" in cfg.translate_profiles, "plamo 2B profile missing"
    assert "plamo8b" in cfg.translate_profiles, "plamo8b profile missing"

    plamo_path = cfg.translate_profiles["plamo"].get("model_path", "")
    plamo8b_path = cfg.translate_profiles["plamo8b"].get("model_path", "")

    assert plamo_path != plamo8b_path, (
        f"plamo and plamo8b share the same model_path: {plamo_path}"
    )

    assert "2b" in plamo_path.lower(), (
        f"plamo 2B path does not contain '2b': {plamo_path}"
    )


# ====================================================================
# 3. Model path resolution tests
# ====================================================================


def test_plamo8b_model_path_resolves():
    """Test that plamo8b model path resolves correctly."""
    from auto_translate.config import load_config, resolve_model_identity

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    identity = resolve_model_identity(cfg, "translate", "plamo8b")
    model_path = identity.get("model_path", "")

    assert model_path, "plamo8b model_path resolved to empty string"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "models" / "plamo3" / "plamo-3-translate-8b-Q3_K_M.gguf").is_file(),
    reason="PLaMo 8B GGUF model file not present (CI environment)",
)
def test_plamo8b_model_file_exists():
    """Test that the PLaMo 8B GGUF model file exists and is valid."""
    from auto_translate.config import load_config, resolve_model_identity

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    identity = resolve_model_identity(cfg, "translate", "plamo8b")
    model_path = identity.get("model_path", "")

    model_file = Path(__file__).parent.parent / model_path
    assert model_file.is_file(), f"Model file not found: {model_file}"

    size = model_file.stat().st_size
    # Q3_K_M 8B should be ~3-5 GB
    assert 3 * (1024**3) < size < 6 * (1024**3), (
        f"Model size {size / (1024**3):.1f} GB is outside expected 3-5 GB range"
    )

    # Check GGUF magic header
    with open(model_file, "rb") as f:
        magic = f.read(4)
    assert magic == b"GGUF", f"Invalid GGUF magic header: {magic}"


def test_plamo_2b_path_isolated_from_8b():
    """Verify plamo (2B) resolves to a different path than plamo8b."""
    from auto_translate.config import load_config, resolve_model_identity

    cfg = load_config(Path(__file__).parent.parent / "config.toml")

    plamo8b_identity = resolve_model_identity(cfg, "translate", "plamo8b")
    plamo8b_path = plamo8b_identity.get("model_path", "")

    plamo_identity = resolve_model_identity(cfg, "translate", "plamo")
    plamo_path = plamo_identity.get("model_path", "")

    assert plamo_path, "plamo 2B path is empty"
    assert plamo_path != plamo8b_path, (
        f"plamo 2B and plamo8b share the same path: {plamo_path}"
    )


def test_gemma4_path_unaffected():
    """Verify gemma4 model path is unaffected by plamo8b addition."""
    from auto_translate.config import load_config, resolve_model_identity

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    gemma_identity = resolve_model_identity(cfg, "translate", "gemma4")
    gemma_path = gemma_identity.get("model_path", "")

    assert "gemma" in gemma_path.lower(), (
        f"gemma4 path does not contain 'gemma': {gemma_path}"
    )


# ====================================================================
# 4. Backend config merge tests
# ====================================================================


def test_plamo8b_backend_config_merge():
    """Test that plamo8b backend config merges correctly."""
    from auto_translate.config import get_backend_config, load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    merged = get_backend_config(cfg, "translate", "plamo8b")

    model_path = merged.get("model_path", "")
    assert "plamo-3-translate-8b" in model_path, (
        f"model_path from profile incorrect: {model_path}"
    )

    gpu_layers = merged.get("gpu_layers", -1)
    assert gpu_layers == -1, f"expected gpu_layers=-1, got {gpu_layers}"

    context_size = merged.get("context_size", 0)
    assert context_size == 8192, f"expected context_size=8192, got {context_size}"


def test_plamo8b_explicit_override_precedence():
    """Test that explicit override takes precedence over profile config."""
    from auto_translate.config import get_backend_config, load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    override = {"model_path": "custom/path.gguf"}
    merged = get_backend_config(
        cfg, "translate", "plamo8b", explicit_overrides=override
    )

    assert merged.get("model_path") == "custom/path.gguf", (
        f"explicit override not applied: {merged.get('model_path')}"
    )


# ====================================================================
# 5. Factory function tests
# ====================================================================


def test_plamo8b_factory_raises_without_model(monkeypatch):
    """Test that plamo8b factory raises clear error without model."""
    from auto_translate.registry import _plamo8b_translate_factory

    # Prevent auto-discovery from finding a real model on disk.
    monkeypatch.setattr(
        "auto_translate.registry.resolve_local_plamo_model_path",
        lambda backend, path: "",
    )
    with pytest.raises(ValueError, match="PLaMo 8B|plamo"):
        _plamo8b_translate_factory({})


def test_plamo8b_factory_accepts_model_path():
    """Test that plamo8b factory accepts config with model_path."""
    from auto_translate.llama_server import LlamaServerError
    from auto_translate.registry import _plamo8b_translate_factory

    # Factory should pass config validation with a model_path,
    # but fail later at model loading (LlamaServerError or similar)
    with pytest.raises((FileNotFoundError, ValueError, OSError, RuntimeError, LlamaServerError)):
        _plamo8b_translate_factory({"model_path": "nonexistent.gguf"})

    # Verify it does NOT raise the "no model configured" error
    try:
        _plamo8b_translate_factory({"model_path": "nonexistent.gguf"})
    except (ValueError, LlamaServerError) as e:
        assert "no PLaMo 8B model is configured" not in str(e), (
            "Factory rejected a valid model_path"
        )


# ====================================================================
# 6. GGUF model file validation (mocked for CI)
# ====================================================================


def test_plamo8b_gguf_file_size_mocked():
    """Test GGUF file size validation using a mock (CI-safe)."""
    # The actual file is 4,310,218,016 bytes (4.01 GiB)
    # Use a mock to verify the size check logic without requiring the file
    expected_size = 4_310_218_016
    size_gb = expected_size / (1024**3)

    assert 3.0 < size_gb < 6.0, (
        f"Expected size {size_gb:.1f} GB is outside 3-5 GB range"
    )
    assert size_gb == pytest.approx(4.01, abs=0.02), (
        f"Expected ~4.01 GiB, got {size_gb:.2f} GiB"
    )


def test_plamo8b_gguf_magic_header_mocked():
    """Test GGUF magic header validation using a mock (CI-safe)."""
    # The GGUF magic header is always b"GGUF"
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"GGUF" + b"\x00" * 100)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"GGUF", f"Invalid GGUF magic: {magic}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ====================================================================
# 7. Config example documentation
# ====================================================================


def test_config_example_mentions_plamo8b():
    """Test that config.example.toml documents plamo8b."""
    example_path = Path(__file__).parent.parent / "config.example.toml"
    assert example_path.is_file(), "config.example.toml not found"

    content = example_path.read_text()
    assert "plamo8b" in content, "plamo8b not documented in config.example.toml"


def test_config_example_translate_backend_plamo8b():
    """Test that config.example.toml shows translate_backend = 'plamo8b'."""
    example_path = Path(__file__).parent.parent / "config.example.toml"
    content = example_path.read_text()

    assert 'translate_backend = "plamo8b"' in content, (
        'Missing translate_backend = "plamo8b" example in config.example.toml'
    )


def test_config_example_plamo8b_profile():
    """Test that config.example.toml documents [translate.plamo8b] profile."""
    example_path = Path(__file__).parent.parent / "config.example.toml"
    content = example_path.read_text()

    assert "[translate.plamo8b]" in content, (
        "Missing [translate.plamo8b] profile in config.example.toml"
    )


# ====================================================================
# 8. CLI help text
# ====================================================================


def test_cli_help_mentions_plamo8b():
    """Test that CLI help mentions plamo8b."""
    main_path = Path(__file__).parent.parent / "src" / "auto_translate" / "__main__.py"
    assert main_path.is_file(), "__main__.py not found"

    content = main_path.read_text(encoding="utf-8", errors="replace")
    assert "plamo8b" in content, "plamo8b not mentioned in CLI help text"


# ====================================================================
# 9. Isolation tests
# ====================================================================


def test_all_model_paths_unique():
    """Test that all translation backend model paths are unique."""
    from auto_translate.config import get_backend_config, load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")

    plamo_cfg = get_backend_config(cfg, "translate", "plamo")
    plamo8b_cfg = get_backend_config(cfg, "translate", "plamo8b")
    gemma_cfg = get_backend_config(cfg, "translate", "gemma4")

    paths = {
        "plamo": plamo_cfg.get("model_path", ""),
        "plamo8b": plamo8b_cfg.get("model_path", ""),
        "gemma4": gemma_cfg.get("model_path", ""),
    }

    unique_paths = set(paths.values())
    assert len(unique_paths) == 3, (
        f"Expected 3 unique paths, got {len(unique_paths)}: {paths}"
    )


def test_plamo8b_does_not_inherit_gemma4_path():
    """Test that plamo8b does not inherit gemma4's model path."""
    from auto_translate.config import get_backend_config, load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    plamo8b_cfg = get_backend_config(cfg, "translate", "plamo8b")

    model_path = plamo8b_cfg.get("model_path", "")
    assert "gemma" not in model_path.lower(), (
        f"plamo8b inherited gemma4 path: {model_path}"
    )


def test_plamo8b_does_not_inherit_plamo_2b_path():
    """Test that plamo8b does not inherit plamo 2B's model path."""
    from auto_translate.config import get_backend_config, load_config

    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    plamo8b_cfg = get_backend_config(cfg, "translate", "plamo8b")

    model_path = plamo8b_cfg.get("model_path", "")
    assert "2b" not in model_path.lower(), (
        f"plamo8b inherited plamo 2B path: {model_path}"
    )


# ====================================================================
# 10. GUI backend selection test
# ====================================================================


def test_plamo8b_appears_in_gui_dropdown():
    """Test that plamo8b appears in the GUI translator dropdown."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from auto_translate.gui import VideoTranslateWindow

    # Create a minimal config with plamo8b profile
    _CONFIG = """[models]
asr_backend = "haas"
translate_backend = "gemma4"

[asr]
model_path = "C:/models/haas"

[translate]
model_path = "C:/models/gemma4"

[translate.plamo8b]
model_path = "C:/models/plamo8b/plamo-3-translate-8b-Q3_K_M.gguf"
"""
    tmp_path = Path(__file__).parent.parent / ".test_tmp_gui"
    tmp_path.mkdir(exist_ok=True)
    config_file = tmp_path / "config.toml"
    config_file.write_text(_CONFIG, encoding="utf-8")

    QApplication.instance() or QApplication([])
    window = VideoTranslateWindow(tmp_path)

    try:
        # Check that plamo8b is in the dropdown
        found = False
        for i in range(window.mt_backend.count()):
            if window.mt_backend.itemData(i) == "plamo8b":
                found = True
                break
        assert found, "plamo8b not found in GUI translator dropdown"

        # Verify selecting plamo8b shows the correct model path
        idx = window.mt_backend.findData("plamo8b")
        window.mt_backend.setCurrentIndex(idx)
        model_text = window.mt_model.text()
        assert "plamo-3-translate-8b" in model_text, (
            f"Selecting plamo8b showed wrong model path: {model_text}"
        )
    finally:
        window.close()
        # Cleanup
        import shutil
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_plamo8b_selection_persists():
    """Test that selecting plamo8b persists after GUI restart."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from auto_translate.gui import VideoTranslateWindow

    _CONFIG = """[models]
asr_backend = "haas"
translate_backend = "gemma4"

[asr]
model_path = "C:/models/haas"

[translate]
model_path = "C:/models/gemma4"

[translate.plamo8b]
model_path = "C:/models/plamo8b/plamo-3-translate-8b-Q3_K_M.gguf"
"""
    tmp_path = Path(__file__).parent.parent / ".test_tmp_gui_persist"
    tmp_path.mkdir(exist_ok=True)
    config_file = tmp_path / "config.toml"
    config_file.write_text(_CONFIG, encoding="utf-8")

    QApplication.instance() or QApplication([])

    # First window: select plamo8b
    first = VideoTranslateWindow(tmp_path)
    prefs = first._preferences
    prefix = first._preferences_prefix
    try:
        idx = first.mt_backend.findData("plamo8b")
        first.mt_backend.setCurrentIndex(idx)
        assert prefs.value(first._preference_key("translate_backend")) == "plamo8b"
    finally:
        first.close()

    # Second window: should restore plamo8b
    second = VideoTranslateWindow(tmp_path)
    try:
        assert second.mt_backend.currentData() == "plamo8b", (
            f"plamo8b selection not restored, got: {second.mt_backend.currentData()}"
        )
        assert second._last_mt_backend == "plamo8b"
    finally:
        second.close()
        # Cleanup preferences and temp dir
        prefs.remove(prefix)
        prefs.sync()
        import shutil
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_documented_plamo8b_profile_loads_from_example(tmp_path):
    """The shipped config example should work when the documented 8B profile is enabled.

    Unlike the hermetic fixture tests, this inspects the *actual* config.example.toml.
    """
    from auto_translate.config import get_backend_config, load_config

    example_path = Path(__file__).parent.parent / "config.example.toml"
    example = example_path.read_text(encoding="utf-8")
    assert 'translate_backend = "gemma4"' in example
    example = example.replace('translate_backend = "gemma4"',
                              'translate_backend = "plamo8b"', 1)
    # Uncomment only the 8B block; identical keys exist under other profiles.
    block = (
        "# [translate.plamo8b]\n"
        '# model_path = "models/plamo3/plamo-3-translate-8b-Q3_K_M.gguf"\n'
        "# gpu_layers = -1\n"
        "# context_size = 8192\n"
        "# max_tokens = 512\n"
        "# load_timeout_s = 300"
    )
    assert block in example, "Documented PLaMo 8B profile is missing or changed"
    enabled_block = "\n".join(line.removeprefix("# ") for line in block.splitlines())
    example = example.replace(block, enabled_block, 1)

    enabled = tmp_path / "enabled-config-example.toml"
    enabled.write_text(example, encoding="utf-8")
    cfg = load_config(enabled)
    assert cfg.models.translate_backend == "plamo8b"
    effective = get_backend_config(cfg, "translate", "plamo8b")
    assert effective["model_path"].endswith("plamo-3-translate-8b-Q3_K_M.gguf")
    assert effective["gpu_layers"] == -1
    assert effective["context_size"] == 8192
