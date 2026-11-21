"""PLaMo 3 Translate 8B integration tests.

Tests backend registration, configuration isolation, model path resolution,
and GUI backend selection for the PLaMo 8B translation backend.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

results = []


def record(test_name: str, passed: bool, detail: str = ""):
    results.append({"test": test_name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ====================================================================
# 1. Backend registration tests
# ====================================================================


def test_plamo8b_registration():
    """Test that plamo8b is registered in the backend registry."""
    section("1. PLaMo 8B Backend Registration")

    try:
        from auto_translate.registry import get_default_registry

        registry = get_default_registry()
        backends = registry.list_backends()

        translate_backends = backends.get("translate", [])
        if "plamo8b" in translate_backends:
            record("plamo8b registered in registry", True,
                   f"translate backends: {translate_backends}")
        else:
            record("plamo8b registered in registry", False,
                   f"translate backends: {translate_backends}")

        # Check capabilities
        caps = registry.get_capabilities("translate", "plamo8b")
        if caps:
            record("plamo8b capabilities", True,
                   f"runtime={caps.runtime}, device={caps.device}, vram={caps.vram_gb}GB")
            if caps.runtime == "llama_cpp":
                record("plamo8b runtime is llama_cpp", True)
            else:
                record("plamo8b runtime is llama_cpp", False, f"got {caps.runtime}")
            if caps.device == "cuda":
                record("plamo8b device is cuda", True)
            else:
                record("plamo8b device is cuda", False, f"got {caps.device}")
            if caps.source_lang == "ja" and caps.target_lang == "en":
                record("plamo8b language pair ja->en", True)
            else:
                record("plamo8b language pair ja->en", False,
                       f"source={caps.source_lang}, target={caps.target_lang}")
        else:
            record("plamo8b capabilities", False, "no capabilities found")

        # Verify plamo (2B) is still registered
        if "plamo" in translate_backends:
            record("plamo 2B still registered", True)
        else:
            record("plamo 2B still registered", False, "plamo 2B missing from registry")

        # Verify other backends are unaffected
        for expected in ("gemma4", "hymt"):
            if expected in translate_backends:
                record(f"{expected} still registered", True)
            else:
                record(f"{expected} still registered", False, f"{expected} missing")

    except Exception as e:
        record("plamo8b registration", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 2. Configuration profile tests
# ====================================================================


def test_plamo8b_config_profile():
    """Test that plamo8b config profile loads correctly."""
    section("2. PLaMo 8B Configuration Profile")

    try:
        from auto_translate.config import (
            load_config,
        )

        cfg = load_config(Path(__file__).parent.parent / "config.toml")

        # Check translate_profiles contains plamo8b
        if "plamo8b" in cfg.translate_profiles:
            profile = cfg.translate_profiles["plamo8b"]
            record("plamo8b profile in config", True,
                   f"keys: {list(profile.keys())}")

            # Check model_path
            model_path = profile.get("model_path", "")
            if "plamo-3-translate-8b" in model_path:
                record("plamo8b model_path contains 8B", True, model_path)
            else:
                record("plamo8b model_path contains 8B", False, model_path)

            # Check gpu_layers
            gpu_layers = profile.get("gpu_layers", -1)
            if gpu_layers == -1:
                record("plamo8b gpu_layers auto", True)
            else:
                record("plamo8b gpu_layers auto", False, f"got {gpu_layers}")

        else:
            record("plamo8b profile in config", False,
                   f"available profiles: {list(cfg.translate_profiles.keys())}")

        # Check plamo (2B) profile is separate
        if "plamo" in cfg.translate_profiles:
            plamo_profile = cfg.translate_profiles["plamo"]
            plamo8b_profile = cfg.translate_profiles.get("plamo8b", {})
            plamo_path = plamo_profile.get("model_path", "")
            plamo8b_path = plamo8b_profile.get("model_path", "")

            if plamo_path != plamo8b_path:
                record("plamo and plamo8b model paths differ", True)
            else:
                record("plamo and plamo8b model paths differ", False,
                       f"both: {plamo_path}")

            if "2b" in plamo_path.lower() or "2B" in plamo_path:
                record("plamo 2B path contains 2B", True)
            else:
                record("plamo 2B path contains 2B", False, plamo_path)

        else:
            record("plamo 2B profile exists", False, "plamo profile missing")

    except Exception as e:
        record("plamo8b config profile", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 3. Model path resolution tests
# ====================================================================


def test_plamo8b_model_path_resolution():
    """Test that plamo8b model path resolves correctly."""
    section("3. PLaMo 8B Model Path Resolution")

    try:
        from auto_translate.config import load_config, resolve_model_identity

        cfg = load_config(Path(__file__).parent.parent / "config.toml")

        # Resolve plamo8b model identity
        identity = resolve_model_identity(cfg, "translate", "plamo8b")
        model_path = identity.get("model_path", "")

        if model_path:
            record("plamo8b model_path resolved", True, model_path)

            # Check the file exists
            model_file = Path(__file__).parent.parent / model_path
            if model_file.is_file():
                size = model_file.stat().st_size
                record("plamo8b model file exists", True,
                       f"size: {size:,} bytes ({size / (1024**3):.1f} GB)")
            else:
                record("plamo8b model file exists", False,
                       f"not found: {model_file}")
        else:
            record("plamo8b model_path resolved", False, "empty path")

        # Verify plamo (2B) resolves to a different path
        plamo_identity = resolve_model_identity(cfg, "translate", "plamo")
        plamo_path = plamo_identity.get("model_path", "")

        if plamo_path and plamo_path != model_path:
            record("plamo 2B path isolated from 8B", True,
                   f"2B: {plamo_path}")
        else:
            record("plamo 2B path isolated from 8B", False,
                   f"2B: {plamo_path}, 8B: {model_path}")

        # Verify gemma4 is unaffected
        gemma_identity = resolve_model_identity(cfg, "translate", "gemma4")
        gemma_path = gemma_identity.get("model_path", "")
        if "gemma" in gemma_path.lower():
            record("gemma4 path unaffected", True)
        else:
            record("gemma4 path unaffected", False, gemma_path)

    except Exception as e:
        record("plamo8b model path resolution", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 4. Backend config merge tests
# ====================================================================


def test_plamo8b_backend_config_merge():
    """Test that plamo8b backend config merges correctly."""
    section("4. PLaMo 8B Backend Config Merge")

    try:
        from auto_translate.config import get_backend_config, load_config

        cfg = load_config(Path(__file__).parent.parent / "config.toml")

        # Get merged config for plamo8b
        merged = get_backend_config(cfg, "translate", "plamo8b")

        # Model path should come from the profile
        model_path = merged.get("model_path", "")
        if "plamo-3-translate-8b" in model_path:
            record("plamo8b model_path from profile", True)
        else:
            record("plamo8b model_path from profile", False, model_path)

        # gpu_layers should be from profile (-1 for auto)
        gpu_layers = merged.get("gpu_layers", -1)
        if gpu_layers == -1:
            record("plamo8b gpu_layers from profile", True)
        else:
            record("plamo8b gpu_layers from profile", False, f"got {gpu_layers}")

        # context_size should be from profile
        context_size = merged.get("context_size", 0)
        if context_size == 8192:
            record("plamo8b context_size from profile", True)
        else:
            record("plamo8b context_size from profile", False, f"got {context_size}")

        # Explicit override should take precedence
        override = {"model_path": "custom/path.gguf"}
        merged_override = get_backend_config(
            cfg, "translate", "plamo8b", explicit_overrides=override
        )
        if merged_override.get("model_path") == "custom/path.gguf":
            record("explicit override takes precedence", True)
        else:
            record("explicit override takes precedence", False,
                   merged_override.get("model_path"))

    except Exception as e:
        record("plamo8b backend config merge", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 5. Factory function tests
# ====================================================================


def test_plamo8b_factory():
    """Test that plamo8b factory raises clear error without model."""
    section("5. PLaMo 8B Factory Function")

    try:
        from auto_translate.registry import _plamo8b_translate_factory

        # Factory should raise ValueError when model_path is empty
        try:
            _plamo8b_translate_factory({})
            record("factory raises without model", False, "no error raised")
        except ValueError as e:
            if "PLaMo 8B" in str(e) or "plamo" in str(e).lower():
                record("factory raises without model", True,
                       "error mentions PLaMo 8B")
            else:
                record("factory raises without model", False,
                       f"error: {e}")
        except Exception as e:
            record("factory raises without model", False,
                   f"unexpected error: {type(e).__name__}: {e}")

        # Factory should accept config dict with model_path
        # (will fail at model loading, but should pass config validation)
        try:
            _plamo8b_translate_factory({"model_path": "nonexistent.gguf"})
            record("factory accepts model_path dict", False, "no error raised")
        except ValueError as e:
            # If it's the "no model configured" error, something went wrong
            if "no PLaMo 8B model is configured" in str(e):
                record("factory accepts model_path dict", False,
                       "rejected valid model_path")
            else:
                # Some other validation error — might be OK
                record("factory accepts model_path dict", True,
                       f"validation passed, failed at model load: {e}")
        except FileNotFoundError:
            # Model file not found — config validation passed
            record("factory accepts model_path dict", True,
                   "config validation passed, model file not found")
        except Exception as e:
            # Other errors (e.g., llama-server not found) mean config passed
            record("factory accepts model_path dict", True,
                   f"config validation passed, failed at: {type(e).__name__}")

    except Exception as e:
        record("plamo8b factory", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 6. GGUF model file validation
# ====================================================================


def test_plamo8b_gguf_file():
    """Test that the PLaMo 8B GGUF file is valid."""
    section("6. PLaMo 8B GGUF File Validation")

    try:
        model_path = Path(__file__).parent.parent / "models" / "plamo3" / "plamo-3-translate-8b-Q3_K_M.gguf"

        if model_path.is_file():
            size = model_path.stat().st_size
            record("GGUF file exists", True,
                   f"size: {size:,} bytes ({size / (1024**3):.1f} GB)")

            # Check size is reasonable (Q3_K_M 8B should be ~3-5 GB)
            if 3 * (1024**3) < size < 6 * (1024**3):
                record("GGUF file size reasonable", True,
                       f"{size / (1024**3):.1f} GB")
            else:
                record("GGUF file size reasonable", False,
                       f"{size / (1024**3):.1f} GB — expected 3-5 GB")

            # Check GGUF magic header
            with open(model_path, "rb") as f:
                magic = f.read(4)
            if magic == b"GGUF":
                record("GGUF magic header valid", True)
            else:
                record("GGUF magic header valid", False,
                       f"got: {magic}")
        else:
            record("GGUF file exists", False, f"not found: {model_path}")

    except Exception as e:
        record("plamo8b GGUF file", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 7. Config example documentation
# ====================================================================


def test_config_example():
    """Test that config.example.toml documents plamo8b."""
    section("7. Config Example Documentation")

    try:
        example_path = Path(__file__).parent.parent / "config.example.toml"
        if example_path.is_file():
            content = example_path.read_text()

            if "plamo8b" in content:
                record("config.example.toml mentions plamo8b", True)
            else:
                record("config.example.toml mentions plamo8b", False,
                       "plamo8b not documented")

            if "translate_backend = \"plamo8b\"" in content:
                record("translate_backend plamo8b example", True)
            else:
                record("translate_backend plamo8b example", False,
                       "no translate_backend = \"plamo8b\" example")

            if "translate.plamo8b" in content:
                record("[translate.plamo8b] profile documented", True)
            else:
                record("[translate.plamo8b] profile documented", False)
        else:
            record("config.example.toml exists", False)

    except Exception as e:
        record("config example", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 8. CLI help text
# ====================================================================


def test_cli_help():
    """Test that CLI help mentions plamo8b."""
    section("8. CLI Help Text")

    try:
        # Check __main__.py for plamo8b mention
        main_path = Path(__file__).parent.parent / "src" / "auto_translate" / "__main__.py"
        if main_path.is_file():
            content = main_path.read_text(encoding="utf-8", errors="replace")

            if "plamo8b" in content:
                record("CLI help mentions plamo8b", True)
            else:
                record("CLI help mentions plamo8b", False,
                       "plamo8b not in CLI help text")
        else:
            record("__main__.py exists", False)

    except Exception as e:
        record("CLI help", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 9. Isolation tests
# ====================================================================


def test_backend_isolation():
    """Test that plamo8b config is isolated from other backends."""
    section("9. Backend Configuration Isolation")

    try:
        from auto_translate.config import get_backend_config, load_config

        cfg = load_config(Path(__file__).parent.parent / "config.toml")

        # Get configs for all translation backends
        plamo_cfg = get_backend_config(cfg, "translate", "plamo")
        plamo8b_cfg = get_backend_config(cfg, "translate", "plamo8b")
        gemma_cfg = get_backend_config(cfg, "translate", "gemma4")

        # Model paths should all be different
        paths = {
            "plamo": plamo_cfg.get("model_path", ""),
            "plamo8b": plamo8b_cfg.get("model_path", ""),
            "gemma4": gemma_cfg.get("model_path", ""),
        }

        unique_paths = set(paths.values())
        if len(unique_paths) == 3:
            record("all model paths unique", True,
                   f"{len(unique_paths)} unique paths")
        else:
            record("all model paths unique", False,
                   f"only {len(unique_paths)} unique paths: {paths}")

        # plamo8b should not inherit gemma4's path
        if "gemma" not in plamo8b_cfg.get("model_path", "").lower():
            record("plamo8b does not inherit gemma4 path", True)
        else:
            record("plamo8b does not inherit gemma4 path", False,
                   plamo8b_cfg.get("model_path"))

        # plamo8b should not inherit plamo 2B's path
        if "2b" not in plamo8b_cfg.get("model_path", "").lower():
            record("plamo8b does not inherit plamo 2B path", True)
        else:
            record("plamo8b does not inherit plamo 2B path", False,
                   plamo8b_cfg.get("model_path"))

    except Exception as e:
        record("backend isolation", False, f"{type(e).__name__}: {e}")


# ====================================================================
# Main
# ====================================================================


def main():
    print("PLaMo 3 Translate 8B Integration Tests")
    print("=" * 60)

    test_plamo8b_registration()
    test_plamo8b_config_profile()
    test_plamo8b_model_path_resolution()
    test_plamo8b_backend_config_merge()
    test_plamo8b_factory()
    test_plamo8b_gguf_file()
    test_config_example()
    test_cli_help()
    test_backend_isolation()

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  Summary: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['test']}: {r['detail']}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
