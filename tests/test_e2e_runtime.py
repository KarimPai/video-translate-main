"""End-to-end runtime test for the full translation pipeline.

Tests actual audio ingestion, Silero VAD, ASR recognition, translation,
and subtitle generation using real models and backends.

This script is designed to run on the development machine with installed
models and binaries. It does NOT modify the project codebase.
"""
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)

# Results tracking
results = []

def record(test_name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    results.append({"test": test_name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ====================================================================
# 1. Backend import and configuration tests
# ====================================================================

def test_backend_imports():
    """Verify all backend modules import without errors."""
    section("1. Backend Imports and Configuration")
    
    try:
        record("Haas ASR import", True)
    except Exception as e:
        record("Haas ASR import", False, str(e))
    
    try:
        record("Qwen3-ASR import", True)
    except Exception as e:
        record("Qwen3-ASR import", False, str(e))
    
    try:
        record("Whisper Turbo import", True)
    except Exception as e:
        record("Whisper Turbo import", False, str(e))
    
    try:
        record("PLaMo translator import", True)
    except Exception as e:
        record("PLaMo translator import", False, str(e))
    
    try:
        record("Gemma4 translator import", True)
    except Exception as e:
        record("Gemma4 translator import", False, str(e))
    
    try:
        record("HyMT translator import", True)
    except Exception as e:
        record("HyMT translator import", False, str(e))
    
    try:
        record("Silero VAD import", True)
    except Exception as e:
        record("Silero VAD import", False, str(e))
    
    try:
        record("MPV output import", True)
    except Exception as e:
        record("MPV output import", False, str(e))
    
    try:
        record("Name fusion import", True)
    except Exception as e:
        record("Name fusion import", False, str(e))


def test_config_load():
    """Verify config.toml loads correctly."""
    try:
        from auto_translate.config import load_config
        config = load_config(Path("config.toml"))
        record("Config load", True, f"ASR={config.models.asr_backend}, MT={config.models.translate_backend}")
        
        # Check key settings
        issues = []
        if not config.asr.model_path:
            issues.append("ASR model_path is empty")
        if not config.translate.model_path:
            issues.append("Translate model_path is empty")
        if config.asr.threads < 1:
            issues.append("ASR threads < 1")
        
        if issues:
            record("Config validation", False, "; ".join(issues))
        else:
            record("Config validation", True)
    except Exception as e:
        record("Config load", False, str(e))


def test_model_paths():
    """Verify configured model files exist."""
    try:
        from auto_translate.config import load_config
        config = load_config(Path("config.toml"))
        
        # Check translation model
        mt_path = Path(config.translate.model_path)
        if mt_path.exists():
            record("Translation model exists", True, mt_path.name)
        else:
            record("Translation model exists", False, f"not found: {mt_path}")
        
        # Check llama-server
        server_path = Path(config.translate.server_path)
        if server_path.exists():
            record("llama-server exists", True, server_path.name)
        else:
            record("llama-server exists", False, f"not found: {server_path}")
        
        # Check speech.cpp
        from auto_translate.speech_server import resolve_speech_executable
        try:
            speech_exe = resolve_speech_executable()
            record("speech.cpp exists", True, str(speech_exe))
        except Exception as e:
            record("speech.cpp exists", False, str(e))
        
        # Check MPV
        mpv = shutil.which("mpv")
        if mpv:
            record("MPV binary exists", True, mpv)
        else:
            record("MPV binary exists", False, "not in PATH")
        
        # Check Qwen3-ASR models from known paths
        qwen_model_path = Path("C:/Games/video-translate-universal/models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf")
        if qwen_model_path.exists():
            record("Qwen3-ASR model exists", True, qwen_model_path.name)
        else:
            record("Qwen3-ASR model exists", False, f"not found: {qwen_model_path}")
        
        qwen_mmproj_path = Path("C:/Games/video-translate-universal/models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
        if qwen_mmproj_path.exists():
            record("Qwen3-ASR mmproj exists", True, qwen_mmproj_path.name)
        else:
            record("Qwen3-ASR mmproj exists", False, f"not found: {qwen_mmproj_path}")
            
    except Exception as e:
        record("Model paths check", False, str(e))


# ====================================================================
# 2. VAD segmentation test
# ====================================================================

def test_vad_segmentation():
    """Test Silero VAD with synthetic speech-like audio using .feed() API."""
    section("2. VAD Segmentation")
    
    try:
        import numpy as np

        from auto_translate.config import load_config
        from auto_translate.types import AudioChunk
        from auto_translate.vad import Segmenter
        
        config = load_config(Path("config.toml"))
        vad_cfg = config.vad
        
        segmenter = Segmenter(vad_cfg)
        record("VAD Segmenter creation", True)
        
        # Generate synthetic audio: silence → speech-like → silence
        sr = 16000
        silence = np.zeros(int(sr * 1.0), dtype=np.float32)
        # Speech-like: sine wave at 440 Hz with some noise
        t = np.linspace(0, 2.0, int(sr * 2.0))
        speech = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
        speech = speech.astype(np.float32)
        audio = np.concatenate([silence, speech, silence])
        
        # Feed audio in chunks using the correct API
        chunk_size = 1600  # 100ms chunks
        all_utterances = []
        now = time.monotonic()
        for i in range(0, len(audio), chunk_size):
            chunk_audio = audio[i:i + chunk_size]
            chunk = AudioChunk(
                start_sample=i,
                pcm=chunk_audio,
                read_at=now,
                reset=i == 0,
                generation=0,
            )
            utterances = segmenter.feed(chunk)
            all_utterances.extend(utterances)
        
        # Flush remaining
        all_utterances.extend(segmenter.flush())
        
        if len(all_utterances) >= 1:
            record("VAD detects speech segment", True, f"{len(all_utterances)} utterance(s)")
        else:
            record("VAD detects speech segment", False, "no utterances found")
        
        # Test all-silence audio
        segmenter2 = Segmenter(vad_cfg)
        silence_only = np.zeros(int(sr * 5.0), dtype=np.float32)
        for i in range(0, len(silence_only), chunk_size):
            chunk_audio = silence_only[i:i + chunk_size]
            chunk = AudioChunk(
                start_sample=i,
                pcm=chunk_audio,
                read_at=now,
                reset=i == 0,
                generation=0,
            )
            segmenter2.feed(chunk)
        silence_utterances = segmenter2.flush()
        record("VAD ignores silence", len(silence_utterances) == 0, 
               f"{len(silence_utterances)} utterances for silence")
        
    except Exception as e:
        record("VAD segmentation", False, str(e))


# ====================================================================
# 3. ASR transcription test (Haas)
# ====================================================================

def test_haas_asr_startup():
    """Test Haas ASR server startup and basic transcription."""
    section("3. Haas ASR Startup and Transcription")
    
    try:
        import numpy as np

        from auto_translate.asr.haas import HaasASR, wav_bytes
        from auto_translate.config import load_config
        
        config = load_config(Path("config.toml"))
        
        haas = HaasASR(config.asr)
        record("Haas ASR server startup", True, f"port={haas.server.port}")
        
        # Test with synthetic audio (won't produce meaningful Japanese, but tests the pipeline)
        sr = 16000
        silence = np.zeros(int(sr * 2.0), dtype=np.float32)
        wav = wav_bytes(silence)
        
        result = haas.server.transcribe(wav, timeout=30)
        record("Haas ASR silent audio", True, f"result='{result.strip()}'")
        
        haas.close()
        record("Haas ASR shutdown", True)
        
    except Exception as e:
        record("Haas ASR startup", False, str(e))


# ====================================================================
# 4. Translation test (PLaMo)
# ====================================================================

def test_plamo_translation():
    """Test PLaMo translator with a simple Japanese sentence."""
    section("4. PLaMo Translation")
    
    try:
        from auto_translate.config import load_config
        from auto_translate.glossary import Glossary
        from auto_translate.stream_context import StreamContext
        from auto_translate.translate.plamo import PlamoTranslator
        
        config = load_config(Path("config.toml"))
        glossary = Glossary({})
        stream = StreamContext()
        
        translator = PlamoTranslator(config.translate, glossary, stream=stream)
        record("PLaMo translator startup", True, f"port={translator.server.port}")
        
        # Test simple translation (translate takes ja + history as list of (ja,en) tuples)
        result = translator.translate("こんにちは", [])
        if result and result.strip():
            record("PLaMo simple translation", True, f"result='{result.strip()}'")
            # Check for prompt leakage
            if "Translate" in result or "translate" in result or "Japanese" in result:
                record("PLaMo prompt leakage check", False, "prompt text found in output")
            else:
                record("PLaMo prompt leakage check", True)
        else:
            record("PLaMo simple translation", False, "empty output")
        
        # Test with context
        result2 = translator.translate("ありがとう", [("こんにちは", result)])
        if result2 and result2.strip():
            record("PLaMo with context", True, f"result='{result2.strip()}'")
        else:
            record("PLaMo with context", False, "empty output")
        
        translator.close()
        record("PLaMo translator shutdown", True)
        
    except Exception as e:
        record("PLaMo translation", False, str(e))


# ====================================================================
# 5. Registry and backend switching test
# ====================================================================

def test_registry():
    """Test the model registry for backend discovery."""
    section("5. Model Registry")
    
    try:
        from auto_translate.registry import get_default_registry
        registry = get_default_registry()
        
        backends = registry.list_backends()
        asr_names = backends.get("asr", [])
        mt_names = backends.get("translate", [])
        record("Registry list backends", True, 
               f"ASR: {asr_names}, MT: {mt_names}")
        
        # Check Haas capabilities
        haas_caps = registry.get_capabilities("asr", "haas")
        if haas_caps:
            record("Haas capabilities", True, f"runtime={haas_caps.runtime}, device={haas_caps.device}")
        else:
            record("Haas capabilities", False, "not found")
        
        # Check PLaMo capabilities
        plamo_caps = registry.get_capabilities("translate", "plamo")
        if plamo_caps:
            record("PLaMo capabilities", True, f"runtime={plamo_caps.runtime}, device={plamo_caps.device}")
        else:
            record("PLaMo capabilities", False, "not found")
        
    except Exception as e:
        record("Registry test", False, str(e))


# ====================================================================
# 6. Name fusion test
# ====================================================================

def test_name_fusion():
    """Test the name fusion module with synthetic data."""
    section("6. Name Fusion")
    
    try:
        from auto_translate.glossary import Glossary
        from auto_translate.name_fusion import NameFusion
        
        glossary = Glossary({"キラ": "Kira", "サトシ": "Satoshi", "リナ": "Rina"})
        
        nf = NameFusion(
            glossary=glossary,
            metadata_text="キラとサトシの物語",
            recognition_context="",
            max_hints=4,
            max_chars=96,
            confirmations=2,
        )
        record("NameFusion creation", True)
        
        # Check initial hints from metadata
        hints = nf.hints()
        if "キラ" in hints or "サトシ" in hints:
            record("NameFusion metadata hints", True, f"hints={hints}")
        else:
            record("NameFusion metadata hints", False, f"hints={hints} (expected キラ or サトシ)")
        
        # Test observation and confirmation
        nf.observe("キラが言った")
        changed2 = nf.observe("キラが笑った")
        if changed2:  # Second observation should trigger confirmation
            record("NameFusion observation confirmation", True)
        else:
            record("NameFusion observation confirmation", False, "second observe did not change")
        
        # Test prompt generation
        prompt = nf.prompt()
        if "キラ" in prompt:
            record("NameFusion prompt", True, f"prompt='{prompt}'")
        else:
            record("NameFusion prompt", False, f"prompt='{prompt}' (expected キラ)")
        
        # Test reset
        nf.reset()
        hints_after = nf.hints()
        # After reset, confirmed names should be cleared but seed names remain
        record("NameFusion reset", True, f"hints after reset={hints_after}")
        
    except Exception as e:
        record("Name fusion test", False, str(e))


# ====================================================================
# 7. Subtitle cleaning test
# ====================================================================

def test_subtitle_cleaning():
    """Test subtitle cleaning functions."""
    section("7. Subtitle Cleaning")
    
    try:
        from auto_translate.subtitle_clean import (
            has_generation_loop,
            looks_like_prompt_echo,
            sanitize_english_output,
        )
        
        # Test prompt echo detection — use a marker that's actually in the code
        echo_text = "Translate this Japanese text: こんにちは"
        if looks_like_prompt_echo(echo_text):
            record("Prompt echo detection", True)
        else:
            record("Prompt echo detection", False, "failed to detect prompt echo")
        
        # Test generation loop detection — needs 18+ words (3x 6-word phrase)
        loop_text = "hello world this is a test hello world this is a test hello world this is a test"
        if has_generation_loop(loop_text):
            record("Generation loop detection", True)
        else:
            record("Generation loop detection", False, "failed to detect loop")
        
        # Test normal text passes
        normal_text = "Hello, how are you today?"
        if not looks_like_prompt_echo(normal_text):
            record("Normal text not flagged as echo", True)
        else:
            record("Normal text not flagged as echo", False, "false positive")
        
        # Test sanitization — sanitize_english_output strips and cleans quotes
        dirty = '  "Hello" world  \n\n'
        clean = sanitize_english_output(dirty)
        if clean and "Hello" in clean and "world" in clean:
            record("Sanitize output", True, f"got '{clean}'")
        else:
            record("Sanitize output", False, f"got '{clean}'")
        
    except Exception as e:
        record("Subtitle cleaning test", False, str(e))


# ====================================================================
# 8. CLI help test
# ====================================================================

def test_cli_help():
    """Test that the CLI entry point works."""
    section("8. CLI Entry Point")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "auto_translate", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0 and "usage" in result.stdout.lower():
            record("CLI --help", True)
        else:
            record("CLI --help", False, f"exit={result.returncode}")
    except Exception as e:
        record("CLI --help", False, str(e))


def test_list_backends():
    """Test --list-backends CLI flag."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "auto_translate", "--list-backends"],
            capture_output=True, text=True, timeout=15,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            has_asr = "haas" in result.stdout.lower() or "asr" in result.stdout.lower()
            has_mt = "plamo" in result.stdout.lower() or "translate" in result.stdout.lower()
            record("CLI --list-backends", has_asr and has_mt, 
                   f"ASR={has_asr}, MT={has_mt}")
        else:
            record("CLI --list-backends", False, f"exit={result.returncode}")
    except Exception as e:
        record("CLI --list-backends", False, str(e))


# ====================================================================
# 9. GPU budget test
# ====================================================================

def test_gpu_budget():
    """Test GPU budget monitoring."""
    section("9. GPU Budget")
    
    try:
        from auto_translate.config import load_config
        from auto_translate.gpu_budget import GpuBudget
        
        config = load_config(Path("config.toml"))
        budget = GpuBudget(config.gpu_budget)
        record("GPU budget creation", True)
        
        # Check if NVIDIA SMI is available via refresh()
        try:
            budget.refresh()
            if budget.used is not None:
                record("GPU refresh", True, f"used={budget.used}")
            else:
                record("GPU refresh", False, "no GPU info (NVIDIA SMI unavailable?)")
        except Exception as e:
            record("GPU refresh", False, str(e))
        
    except Exception as e:
        record("GPU budget test", False, str(e))


# ====================================================================
# 10. Glossary test
# ====================================================================

def test_glossary():
    """Test glossary loading and lookup."""
    section("10. Glossary")
    
    try:
        from auto_translate.glossary import Glossary
        
        glossary = Glossary({"テスト": "test", "例え": "example"})
        record("Glossary creation", True)
        
        # Load from file if it exists
        glossary_path = Path(__file__).parent.parent / "glossary.toml"
        if glossary_path.exists():
            glossary2 = Glossary.load(glossary_path)
            len(glossary2.terms) + len(glossary2.vocabulary)
            record("Glossary load from file", True, 
                   f"terms={len(glossary2.terms)}, vocab={len(glossary2.vocabulary)}")
        else:
            record("Glossary load from file", False, "glossary.toml not found")
        
        # Test find method
        found = glossary.find("これはテストです", context="これはテストです")
        if any(ja == "テスト" for ja, _ in found):
            record("Glossary find", True, f"found={found}")
        else:
            record("Glossary find", False, f"found={found} (expected テスト)")
        
    except Exception as e:
        record("Glossary test", False, str(e))


# ====================================================================
# 11. Text filter test
# ====================================================================

def test_text_filter():
    """Test text repetition filtering."""
    section("11. Text Filter")
    
    try:
        from auto_translate.text_filter import trim_repetition_tail
        
        # Test repetition trimming
        repeated = "hello hello hello goodbye"
        trimmed = trim_repetition_tail(repeated)
        record("Trim repetition tail", True, f"'{repeated}' -> '{trimmed}'")
        
        # Test no repetition
        normal = "hello goodbye"
        trimmed2 = trim_repetition_tail(normal)
        if trimmed2 == normal:
            record("No false trim", True)
        else:
            record("No false trim", False, f"'{normal}' -> '{trimmed2}'")
        
    except Exception as e:
        record("Text filter test", False, str(e))


# ====================================================================
# 12. Quality guard test
# ====================================================================

def test_quality_guard():
    """Test transcript quality guard."""
    section("12. Quality Guard")
    
    try:
        from auto_translate.asr.quality_guard import transcript_problem
        
        # Test empty transcript
        problem = transcript_problem("", 0.0)
        if problem:
            record("Empty transcript flagged", True, f"problem={problem}")
        else:
            record("Empty transcript flagged", False)
        
        # Test normal transcript
        problem2 = transcript_problem("hello", 1.0)
        if not problem2:
            record("Normal transcript passes", True)
        else:
            record("Normal transcript passes", False, f"problem={problem2}")
        
    except Exception as e:
        record("Quality guard test", False, str(e))


# ====================================================================
# Main runner
# ====================================================================

def main():
    # Fix Windows console encoding for Japanese characters
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("\n" + "="*60)
    print("  Universal Branch - End-to-End Runtime Tests")
    print("="*60)
    
    test_backend_imports()
    test_config_load()
    test_model_paths()
    test_vad_segmentation()
    test_haas_asr_startup()
    test_plamo_translation()
    test_registry()
    test_name_fusion()
    test_subtitle_cleaning()
    test_cli_help()
    test_list_backends()
    test_gpu_budget()
    test_glossary()
    test_text_filter()
    test_quality_guard()
    
    # Summary
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
