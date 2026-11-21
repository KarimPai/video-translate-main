"""Whisper Large-v3 Turbo unit checks and manually invoked local smoke tests.

Tests model loading, language detection, mixed-language recovery,
and hallucination prevention for the faster-whisper backend.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

results = []

# Smoke tests require a local CUDA model and are not collected by pytest.
# Running this file directly now fails on the first unsuccessful check.


def record(test_name: str, passed: bool, detail: str = ""):
    results.append({"test": test_name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))
    if not passed:
        raise AssertionError(f"{test_name} failed: {detail}")


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ====================================================================
# 1. Model resolution tests
# ====================================================================


def smoke_model_path_resolution():
    """Test that Whisper resolves the local model path correctly."""
    section("1. Model Path Resolution")

    try:
        from auto_translate.asr.whisper_turbo import WhisperASR
        from auto_translate.config import AsrConfig

        # Test with explicit local path
        cfg = AsrConfig(
            model_path="models/whisper-large-v3-turbo",
            cuda_device=0,
            threads=4,
        )
        path = WhisperASR._resolve_model_path(cfg, is_explicit=True)
        if "whisper-large-v3-turbo" in path:
            record("Explicit local path resolution", True, f"path={path}")
        else:
            record("Explicit local path resolution", False, f"unexpected path={path}")

        # Test with empty model_path — should find the local model via
        # _WHISPER_MODEL default (h2oai) if cached, or raise if not.
        # Since we downloaded to models/ dir, the HF cache may not have it.
        cfg_empty = AsrConfig(model_path="", cuda_device=0, threads=4)
        try:
            cached_path = WhisperASR._resolve_model_path(cfg_empty, is_explicit=False)
            record("HF cache fallback", True, f"path={cached_path}")
        except Exception as e:
            # Expected if h2oai model is not in HF cache (only local download)
            if "not present in the local cache" in str(e) or "No Whisper model available" in str(e):
                record("HF cache fallback (expected miss)", True,
                       "h2oai model not in HF cache — use local model_path")
            else:
                record("HF cache fallback", False, str(e))

        # Test _looks_like_whisper_model
        model_dir = Path("models/whisper-large-v3-turbo")
        if WhisperASR._looks_like_whisper_model(model_dir):
            record("Model directory validation", True, str(model_dir))
        else:
            record("Model directory validation", False, f"not recognized: {model_dir}")

        # Test invalid path raises clear error
        cfg_bad = AsrConfig(model_path="/nonexistent/model", cuda_device=0, threads=4)
        try:
            WhisperASR._resolve_model_path(cfg_bad, is_explicit=True)
            record("Invalid explicit path error", False, "no error raised")
        except Exception as e:
            # Path with / is treated as HF repo ID; error should mention cache or existence
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ["not present", "does not exist", "not a valid", "no whisper model"]):
                record("Invalid explicit path error", True, f"error={type(e).__name__}")
            else:
                record("Invalid explicit path error", False, f"unclear error: {e}")

    except Exception as e:
        record("Model path resolution", False, str(e))


# ====================================================================
# 2. CUDA model loading test
# ====================================================================


def smoke_cuda_model_loading():
    """Test that Whisper loads on CUDA without PyTorch."""
    section("2. CUDA Model Loading")

    try:
        from auto_translate.asr.whisper_turbo import WhisperASR, WhisperError
        from auto_translate.config import AsrConfig

        cfg = AsrConfig(
            model_path="models/whisper-large-v3-turbo",
            cuda_device=0,
            threads=4,
        )

        t0 = time.perf_counter()
        asr = WhisperASR(cfg)
        load_time = time.perf_counter() - t0

        record("CUDA model load", True, f"loaded in {load_time:.1f}s")

        # Verify language detection attributes exist
        if hasattr(asr, "last_detected_language"):
            record("last_detected_language attribute", True,
                   f"default={asr.last_detected_language}")
        else:
            record("last_detected_language attribute", False, "missing")

        if hasattr(asr, "last_detected_language_probability"):
            record("last_detected_language_probability attribute", True,
                   f"default={asr.last_detected_language_probability}")
        else:
            record("last_detected_language_probability attribute", False, "missing")

        # Verify mixed-language recovery method exists
        if hasattr(asr, "recover_japanese_from_mixed_audio"):
            record("recover_japanese_from_mixed_audio method", True)
        else:
            record("recover_japanese_from_mixed_audio method", False, "missing")

        # Verify _detect_language method
        if hasattr(asr, "_detect_language"):
            ja_result = asr._detect_language("こんにちは世界")
            en_result = asr._detect_language("Hello world")
            mixed_result = asr._detect_language("Hello 世界")
            none_result = asr._detect_language("")

            if ja_result == "ja":
                record("Detect Japanese text", True)
            else:
                record("Detect Japanese text", False, f"got {ja_result}")

            if en_result == "en":
                record("Detect English text", True)
            else:
                record("Detect English text", False, f"got {en_result}")

            if mixed_result == "mixed":
                record("Detect mixed text", True)
            else:
                record("Detect mixed text", False, f"got {mixed_result}")

            if none_result == "none":
                record("Detect empty text", True)
            else:
                record("Detect empty text", False, f"got {none_result}")

        asr.close()
        record("CUDA model close", True)

    except WhisperError as e:
        record("CUDA model load", False, f"WhisperError: {e}")
    except Exception as e:
        record("CUDA model load", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 3. Transcription with synthetic audio
# ====================================================================


def smoke_silence_transcription():
    """Test that silence/noise produces no hallucinated output."""
    section("3. Silence Transcription")

    try:
        from auto_translate.asr.whisper_turbo import WhisperASR
        from auto_translate.config import AsrConfig

        cfg = AsrConfig(
            model_path="models/whisper-large-v3-turbo",
            cuda_device=0,
            threads=4,
        )
        asr = WhisperASR(cfg)

        # Pure silence (5 seconds)
        silence = np.zeros(16000 * 5, dtype=np.float32)
        result = asr.transcribe(silence)
        if result == "":
            record("Silence produces no output", True)
        else:
            record("Silence produces no output", False, f"hallucinated: '{result}'")

        # Low-level noise (5 seconds)
        noise = np.random.randn(16000 * 5).astype(np.float32) * 0.001
        result = asr.transcribe(noise)
        if result == "":
            record("Low noise produces no output", True)
        else:
            record("Low noise produces no output", False, f"hallucinated: '{result}'")

        # Very short audio (should be skipped)
        short = np.random.randn(int(16000 * 0.03)).astype(np.float32) * 0.1
        result = asr.transcribe(short)
        if result == "":
            record("Sub-50ms audio skipped", True)
        else:
            record("Sub-50ms audio skipped", False, f"unexpected: '{result}'")

        asr.close()

    except Exception as e:
        record("Silence transcription", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 4. Cancel and state management
# ====================================================================


def smoke_cancel_and_state():
    """Test cancellation, remember, and forget operations."""
    section("4. Cancel and State Management")

    try:
        from auto_translate.asr.whisper_turbo import WhisperASR
        from auto_translate.config import AsrConfig

        cfg = AsrConfig(
            model_path="models/whisper-large-v3-turbo",
            cuda_device=0,
            threads=4,
        )
        asr = WhisperASR(cfg)

        # Test remember/forget
        asr.remember("test transcript 1")
        asr.remember("test transcript 2")
        asr.forget(1)
        record("Remember and forget", True)

        # Test cancel
        asr.cancel_pending()
        record("Cancel pending", True)

        # Test set_context
        asr.set_context("name hint")
        record("Set context", True)

        asr.close()
        record("Close after operations", True)

    except Exception as e:
        record("Cancel and state", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 5. Config profile tests
# ====================================================================


def smoke_config_profile():
    """Test that [asr.whisper] profile loads correctly."""
    section("5. Config Profile Loading")

    try:
        from auto_translate.config import load_config, get_backend_config

        config = load_config(Path("config.toml"))

        # Check whisper profile exists
        if "whisper" in config.asr_profiles:
            profile = config.asr_profiles["whisper"]
            record("Whisper profile exists", True,
                   f"keys={list(profile.keys())}")
        else:
            record("Whisper profile exists", False,
                   f"available: {list(config.asr_profiles.keys())}")

        # Test get_backend_config for whisper
        merged = get_backend_config(config, "asr", "whisper")
        if "model_path" in merged:
            record("Whisper merged config", True,
                   f"model_path={merged['model_path']}")
        else:
            record("Whisper merged config", False, "no model_path")

        # Verify CUDA device is set
        if merged.get("cuda_device", -1) == 0:
            record("CUDA device from profile", True)
        else:
            record("CUDA device from profile", False,
                   f"cuda_device={merged.get('cuda_device')}")

    except Exception as e:
        record("Config profile", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 6. Registry integration tests
# ====================================================================


def smoke_registry_integration():
    """Test Whisper registration and factory creation."""
    section("6. Registry Integration")

    try:
        from auto_translate.registry import get_default_registry

        registry = get_default_registry()

        # Check Whisper is registered
        caps = registry.get_capabilities("asr", "whisper")
        if caps:
            record("Whisper registered", True,
                   f"runtime={caps.runtime}, device={caps.device}")
        else:
            record("Whisper registered", False, "not found")

        # Check capabilities
        if caps and caps.runtime == "faster_whisper":
            record("Whisper runtime", True)
        elif caps:
            record("Whisper runtime", False, f"runtime={caps.runtime}")

        # List all ASR backends
        backends = registry.list_backends("asr")
        asr_list = backends.get("asr", [])
        if "whisper" in asr_list:
            record("Whisper in ASR list", True, f"list={asr_list}")
        else:
            record("Whisper in ASR list", False, f"list={asr_list}")

    except Exception as e:
        record("Registry integration", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 7. Quality guard tests
# ====================================================================


def test_quality_guard():
    """Test hallucination prevention with quality guard."""
    section("7. Quality Guard")

    try:
        from auto_translate.asr.quality_guard import transcript_problem

        # Empty text
        reason = transcript_problem("", 1.0)
        if reason == "empty":
            record("Empty text rejected", True)
        else:
            record("Empty text rejected", False, f"reason={reason}")

        # Phantom outro
        reason = transcript_problem("ご視聴ありがとうございました", 0.5)
        if reason == "phantom_outro":
            record("Phantom outro rejected", True)
        else:
            record("Phantom outro rejected", False, f"reason={reason}")

        # Normal Japanese text should pass
        reason = transcript_problem("こんにちは、今日はいい天気ですね", 3.0)
        if reason is None:
            record("Normal Japanese accepted", True)
        else:
            record("Normal Japanese accepted", False, f"reason={reason}")

        # Short Japanese reaction should pass
        reason = transcript_problem("ええ", 0.8)
        if reason is None:
            record("Short reaction accepted", True)
        else:
            record("Short reaction accepted", False, f"reason={reason}")

        # Implausibly long output for short audio — repeated patterns are caught
        # by repeated_kana_loop or repeated_character. Use truly varied text.
        long_text = "こんにちは世界今日はいい天気ですねゲームを始めましょう" * 10  # 600+ chars
        reason = transcript_problem(long_text, 2.0)
        if reason in ("implausible_length", "implausible_speech_rate"):
            record("Implausible length rejected", True, f"reason={reason}")
        else:
            record("Implausible length rejected", False, f"reason={reason}")

    except Exception as e:
        record("Quality guard", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 8. Text filter tests
# ====================================================================


def test_text_filter():
    """Test text filter for laughter, loops, and noise."""
    section("8. Text Filter")

    try:
        from auto_translate.text_filter import rejection_reason, is_usable_transcript

        # Laughter-only should be rejected
        reason = rejection_reason("あはははははは")
        if reason == "laughter_only":
            record("Laughter rejected", True)
        else:
            record("Laughter rejected", False, f"reason={reason}")

        # Normal text should pass
        if is_usable_transcript("こんにちは世界"):
            record("Normal text accepted", True)
        else:
            record("Normal text accepted", False)

        # Repeated character loop
        reason = rejection_reason("aaaaaa")
        if reason == "repeated_character":
            record("Repeated char rejected", True)
        else:
            record("Repeated char rejected", False, f"reason={reason}")

        # Kana loop
        reason = rejection_reason("あいあいあいあいあいあい")
        if reason == "repeated_kana_loop":
            record("Kana loop rejected", True)
        else:
            record("Kana loop rejected", False, f"reason={reason}")

        # Mixed Japanese with English names should pass
        if is_usable_transcript("Jasonが来ました"):
            record("Mixed JA-EN accepted", True)
        else:
            record("Mixed JA-EN accepted", False)

    except Exception as e:
        record("Text filter", False, f"{type(e).__name__}: {e}")


# ====================================================================
# 9. Language filter tests
# ====================================================================


def test_language_filter():
    """Test language filter for Japanese vs English detection."""
    section("9. Language Filter")

    try:
        from auto_translate.config import LanguageFilterConfig
        from auto_translate.text_filter import LanguageFilter

        cfg = LanguageFilterConfig()
        lf = LanguageFilter(cfg)

        # Japanese text should be kept
        if lf.should_keep("こんにちは", 3.0):
            record("Japanese kept", True)
        else:
            record("Japanese kept", False)

        # Short utterances always kept
        if lf.should_keep("Hello", 1.0):
            record("Short utterance kept", True)
        else:
            record("Short utterance kept", False)

        # Mixed text should be kept
        if lf.should_keep("Hello 世界", 3.0):
            record("Mixed text kept", True)
        else:
            record("Mixed text kept", False)

        # Pure English long text should be dropped
        if not lf.should_keep("This is a long English sentence without any Japanese", 3.0):
            record("Pure English dropped", True)
        else:
            record("Pure English dropped", False)

    except Exception as e:
        record("Language filter", False, f"{type(e).__name__}: {e}")


# ====================================================================
# Main
# ====================================================================


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "="*60)
    print("  Whisper Large-v3 Turbo Integration Tests")
    print("="*60)

    smoke_model_path_resolution()
    smoke_cuda_model_loading()
    smoke_silence_transcription()
    smoke_cancel_and_state()
    smoke_config_profile()
    smoke_registry_integration()
    test_quality_guard()
    test_text_filter()
    test_language_filter()

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
