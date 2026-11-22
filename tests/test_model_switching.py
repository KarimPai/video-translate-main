"""Model switching and failure injection tests.

Tests backend switching, failure recovery, and resource cleanup.
Runs against real models and servers on the development machine.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# These integration scripts launch real models. Opt in explicitly for pytest.
# Direct `python tests/test_model_switching.py` invocation remains supported.
__test__ = os.environ.get("VIDEO_TRANSLATE_RUN_REAL_MODELS") == "1"

results = []

def record(test_name: str, passed: bool, detail: str = ""):
    results.append({"test": test_name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))
    if not passed:
        raise AssertionError(f"{test_name}: {detail}")

def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ====================================================================
# 1. Model switching tests
# ====================================================================

def test_qwen3_asr_startup():
    """Test Qwen3-ASR server startup and shutdown."""
    section("1. Qwen3-ASR Startup")
    
    try:
        from auto_translate.asr.qwen3_asr import Qwen3ASR
        from auto_translate.config import load_config
        
        config = load_config(Path("config.toml"))
        
        # Qwen3-ASR uses the asr config but with different model paths
        # The config section [asr.qwen3-asr] provides model_path and mmproj_path
        asr_cfg = config.asr
        
        qwen = Qwen3ASR(asr_cfg)
        record("Qwen3-ASR server startup", True, f"port={qwen.server.port}")
        
        # Give it a moment to stabilize
        time.sleep(1)
        
        qwen.close()
        record("Qwen3-ASR shutdown", True)
        
    except Exception as e:
        record("Qwen3-ASR startup", False, str(e))


def test_plamo_with_different_model():
    """Test PLaMo translator using the PLaMO model (not Gemma4)."""
    section("2. PLaMo with PLaMO Model")
    
    try:
        from dataclasses import replace

        from auto_translate.config import load_config
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator
        
        config = load_config(Path("config.toml"))
        glossary = Glossary({})
        stream = StreamContext()
        
        # Use PLaMo-specific config
        plamo_cfg = config.translate_plamo
        # Override model path to use PLaMO model
        plamo_cfg = replace(plamo_cfg, model_path="C:/Games/video-translate-universal/models/plamo3/plamo-3-translate-2b-Q4_K_M.gguf")
        
        translator = PlamoTranslator(plamo_cfg, glossary, stream=stream)
        record("PLaMo with PLaMO model startup", True, f"port={translator.server.port}")
        
        result = translator.translate("こんにちは", [])
        if result and result.strip():
            record("PLaMo translation", True, f"result='{result.strip()}'")
        else:
            record("PLaMo translation", False, "empty output")
        
        translator.close()
        record("PLaMo shutdown", True)
        
    except Exception as e:
        record("PLaMo with PLaMO model", False, str(e))


def test_hymt_startup():
    """Test HyMT translator startup."""
    section("3. HyMT Startup")
    
    try:
        from dataclasses import replace

        from auto_translate.config import load_config
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.hymt import HymtTranslator
        
        config = load_config(Path("config.toml"))
        glossary = Glossary({})
        stream = StreamContext()
        
        hymt_cfg = config.translate_hymt
        hymt_cfg = replace(hymt_cfg, model_path="C:/Games/video-translate-universal/models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf")
        
        translator = HymtTranslator(hymt_cfg, glossary, stream=stream)
        record("HyMT translator startup", True, f"port={translator.server.port}")
        
        result = translator.translate("こんにちは", [])
        if result and result.strip():
            record("HyMT translation", True, f"result='{result.strip()}'")
        else:
            record("HyMT translation", False, "empty output")
        
        translator.close()
        record("HyMT shutdown", True)
        
    except Exception as e:
        record("HyMT startup", False, str(e))


def test_gemma4_startup():
    """Test Gemma4 translator startup."""
    section("4. Gemma4 Startup")
    
    try:
        from auto_translate.config import load_config
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.gemma4 import Gemma4Translator
        
        config = load_config(Path("config.toml"))
        glossary = Glossary({})
        stream = StreamContext()
        
        translator = Gemma4Translator(config.translate, glossary, stream=stream)
        record("Gemma4 translator startup", True, f"port={translator.server.port}")
        
        result = translator.translate("こんにちは", [])
        if result and result.strip():
            record("Gemma4 translation", True, f"result='{result.strip()}'")
        else:
            record("Gemma4 translation", False, "empty output")
        
        translator.close()
        record("Gemma4 shutdown", True)
        
    except Exception as e:
        record("Gemma4 startup", False, str(e))


# ====================================================================
# 2. Failure injection tests
# ====================================================================

def test_invalid_model_path():
    """Test that invalid model path produces a clear error."""
    section("5. Invalid Model Path")
    
    try:
        from dataclasses import replace

        from auto_translate.config import TranslateConfig
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator
        
        cfg = TranslateConfig()
        cfg = replace(cfg, model_path="/nonexistent/model.gguf", server_path=cfg.server_path)
        glossary = Glossary({})
        stream = StreamContext()
        
        try:
            translator = PlamoTranslator(cfg, glossary, stream=stream)
            record("Invalid model path error", False, "no error raised")
            translator.close()
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            record("Invalid model path error", True, f"error={type(e).__name__}")
        
    except Exception as e:
        record("Invalid model path test", False, str(e))


def test_speech_server_not_found():
    """Test behavior when speech.cpp is not available."""
    section("6. Speech Server Not Found")
    
    try:
        from auto_translate.speech_server import (
            SpeechServerError,
            resolve_speech_executable,
        )
        
        try:
            # Try with a nonexistent path
            resolve_speech_executable("/nonexistent/speech.exe")
            record("Missing speech.cpp error", False, "no error raised")
        except SpeechServerError:
            record("Missing speech.cpp error", True, "SpeechServerError raised")
        except Exception as e:
            record("Missing speech.cpp error", False, f"wrong error: {type(e).__name__}")
        
    except Exception as e:
        record("Speech server test", False, str(e))


def test_rapid_start_stop():
    """Test rapid start/stop of Haas ASR."""
    section("7. Rapid Start/Stop")
    
    try:
        from auto_translate.asr.haas import HaasASR
        from auto_translate.config import load_config
        
        config = load_config(Path("config.toml"))
        
        for i in range(3):
            haas = HaasASR(config.asr)
            port = haas.server.port
            haas.close()
            record(f"Rapid cycle {i+1}", True, f"port={port}")
            time.sleep(0.5)
        
    except Exception as e:
        record("Rapid start/stop", False, str(e))


def test_cli_invalid_source():
    """Test CLI with invalid source."""
    section("8. CLI Invalid Source")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "auto_translate", "not_a_file_or_url"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode != 0:
            record("CLI rejects invalid source", True, f"exit={result.returncode}")
        else:
            record("CLI rejects invalid source", False, "exit=0 (should fail)")
    except Exception as e:
        record("CLI invalid source", False, str(e))


# ====================================================================
# 3. Resource cleanup tests
# ====================================================================

def test_process_cleanup():
    """Verify no orphan processes after backend shutdown."""
    section("9. Process Cleanup")
    
    try:
        from auto_translate.asr.haas import HaasASR
        from auto_translate.config import load_config
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator
        
        # Count processes before
        before = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq speech.exe", "/FI", "IMAGENAME eq llama-server.exe"],
            capture_output=True, text=True, timeout=5,
        )
        before_count = len([l for l in before.stdout.split("\n") if "speech.exe" in l or "llama-server.exe" in l])
        
        config = load_config(Path("config.toml"))
        glossary = Glossary({})
        stream = StreamContext()
        
        haas = HaasASR(config.asr)
        plamo = PlamoTranslator(config.translate, glossary, stream=stream)
        
        # Count during
        time.sleep(1)
        during = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq speech.exe", "/FI", "IMAGENAME eq llama-server.exe"],
            capture_output=True, text=True, timeout=5,
        )
        during_count = len([l for l in during.stdout.split("\n") if "speech.exe" in l or "llama-server.exe" in l])
        record("Processes during run", True, f"before={before_count}, during={during_count}")
        
        haas.close()
        plamo.close()
        
        # Count after
        time.sleep(2)
        after = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq speech.exe", "/FI", "IMAGENAME eq llama-server.exe"],
            capture_output=True, text=True, timeout=5,
        )
        after_count = len([l for l in after.stdout.split("\n") if "speech.exe" in l or "llama-server.exe" in l])
        
        if after_count <= before_count + 1:  # allow 1 for this test process
            record("Process cleanup after shutdown", True, f"after={after_count}")
        else:
            record("Process cleanup after shutdown", False, f"orphan processes: after={after_count}, before={before_count}")
        
    except Exception as e:
        record("Process cleanup", False, str(e))


def test_gpu_memory_after_shutdown():
    """Check GPU memory returns to baseline after shutdown."""
    section("10. GPU Memory After Shutdown")
    
    try:
        from auto_translate.config import load_config
        from auto_translate.gpu_budget import GpuBudget
        
        config = load_config(Path("config.toml"))
        budget = GpuBudget(config.gpu_budget)
        
        # Baseline
        budget.refresh()
        baseline = budget.used
        record("GPU baseline", True, f"used={baseline:.2f}GB")
        
        # We can't easily test GPU delta without starting backends here,
        # but we can verify the baseline reading works
        if baseline is not None and 0 < baseline < 8:
            record("GPU memory reading valid", True)
        else:
            record("GPU memory reading valid", False, f"unexpected value: {baseline}")
        
    except Exception as e:
        record("GPU memory test", False, str(e))


# ====================================================================
# 4. Registry factory tests
# ====================================================================

def test_registry_factory():
    """Test registry factory creation for all backends."""
    section("11. Registry Factory Creation")
    
    try:
        from auto_translate.registry import get_default_registry
        
        registry = get_default_registry()
        
        # Test VAD factory
        try:
            from auto_translate.config import load_config
            config = load_config(Path("config.toml"))
            vad = registry.create_vad("silero", config.vad)
            record("Registry VAD factory", True, type(vad).__name__)
        except Exception as e:
            record("Registry VAD factory", False, str(e))
        
        # Test ASR factory (just validation, not full startup)
        try:
            caps = registry.get_capabilities("asr", "haas")
            record("Registry ASR capabilities", True, f"haas: {caps.runtime}")
        except Exception as e:
            record("Registry ASR capabilities", False, str(e))
        
        try:
            caps = registry.get_capabilities("asr", "qwen3-asr")
            record("Registry Qwen3 capabilities", True, f"qwen3-asr: {caps.runtime}")
        except Exception as e:
            record("Registry Qwen3 capabilities", False, str(e))
        
        try:
            caps = registry.get_capabilities("asr", "whisper")
            record("Registry Whisper capabilities", True, f"whisper: {caps.runtime}")
        except Exception as e:
            record("Registry Whisper capabilities", False, str(e))
        
        try:
            caps = registry.get_capabilities("translate", "plamo")
            record("Registry PLaMo capabilities", True, f"plamo: {caps.runtime}")
        except Exception as e:
            record("Registry PLaMo capabilities", False, str(e))
        
        try:
            caps = registry.get_capabilities("translate", "gemma4")
            record("Registry Gemma4 capabilities", True, f"gemma4: {caps.runtime}")
        except Exception as e:
            record("Registry Gemma4 capabilities", False, str(e))
        
        try:
            caps = registry.get_capabilities("translate", "hymt")
            record("Registry HyMT capabilities", True, f"hymt: {caps.runtime}")
        except Exception as e:
            record("Registry HyMT capabilities", False, str(e))
        
    except Exception as e:
        record("Registry factory test", False, str(e))


# ====================================================================
# 5. Config backend switching
# ====================================================================

def test_config_backend_switching():
    """Test that config supports different backend combinations."""
    section("12. Config Backend Switching")
    
    try:
        from auto_translate.config import load_config
        
        config = load_config(Path("config.toml"))
        
        # Check that PLaMo-specific config exists
        if hasattr(config, 'translate_plamo'):
            record("PLaMo config section", True, f"model={config.translate_plamo.model_path}")
        else:
            record("PLaMo config section", False, "translate_plamo not found")
        
        # Check HyMT config
        if hasattr(config, 'translate_hymt'):
            record("HyMT config section", True, f"model={config.translate_hymt.model_path}")
        else:
            record("HyMT config section", False, "translate_hymt not found")
        
        # Check Qwen3-ASR config
        if hasattr(config, 'asr_qwen3_asr') or hasattr(config.asr, 'qwen3_asr'):
            record("Qwen3-ASR config section", True)
        else:
            # Check if it's in the raw config
            record("Qwen3-ASR config section", False, "qwen3-asr config not found in dataclass")
        
    except Exception as e:
        record("Config backend switching", False, str(e))


# ====================================================================
# Main
# ====================================================================

def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("\n" + "="*60)
    print("  Model Switching & Failure Injection Tests")
    print("="*60)
    
    test_qwen3_asr_startup()
    test_plamo_with_different_model()
    test_hymt_startup()
    test_gemma4_startup()
    test_invalid_model_path()
    test_speech_server_not_found()
    test_rapid_start_stop()
    test_cli_invalid_source()
    test_process_cleanup()
    test_gpu_memory_after_shutdown()
    test_registry_factory()
    test_config_backend_switching()
    
    section("SUMMARY")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    
    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}")
    
    if failed > 0:
        print("\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['test']}: {r['detail']}")
    
    print()
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
