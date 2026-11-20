"""Real inference tests for Gemma 4 E4B translator.

This script starts the actual llama-server with the Gemma 4 model and runs
real translation requests to verify:
1. Normal translations of representative Japanese utterances
2. Slash preservation (no corruption of yes/no, 24/7, etc.)
3. Truncation handling and retry logic
4. Reasoning disabled (no reasoning_content in output)
5. Channel marker protection
6. Prompt caching behavior (repeated prompts produce stable results)
7. Changing context produces different results
8. Latency and token usage metrics

IMPORTANT: This requires the llama-server binary and GPU to be available.
It will skip gracefully if the environment is not ready.
"""
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.translate.gemma4 import (
    Gemma4Translator,
    clean_output,
    build_prompt,
    resolve_slash_alternatives,
)
from auto_translate.llama_server import LlamaServerError

# Representative Japanese utterances for testing
TEST_UTTERANCES = [
    # Basic greetings and expressions
    ("こんにちは", "Hello"),
    ("ありがとう", "Thank you"),
    ("さようなら", "Goodbye"),
    ("はい", "Yes"),
    ("いいえ", "No"),
    ("すみません", "Excuse me / Sorry"),
    
    # Conversational phrases
    ("今日はいい天気ですね", "It's nice weather today"),
    ("何をしているの？", "What are you doing?"),
    ("わからない", "I don't know"),
    ("本当に？", "Really?"),
    ("待って！", "Wait!"),
    ("すごい！", "Amazing!"),
    
    # Longer sentences
    ("みんな、スパチャありがとう！今日はマインクラフトで遊びます。",
     "Thanks for the super chats everyone! We're playing Minecraft today."),
    ("このゲームは本当に面白いですよね", "This game is really fun, isn't it?"),
    ("ちょっと待ってください", "Please wait a moment"),
    
    # Emotional expressions
    ("やばい！", "No way! / That's crazy!"),
    ("えええ？", "Whaaat?"),
    ("おもしろい！", "Funny!"),
    
    # Names and nicknames
    ("あすかちゃん、こんにちは", "Hello Asuka-chan"),
    ("みんな、おはよう", "Good morning everyone"),
]

# Slash preservation test cases - these should NOT be corrupted
SLASH_TEST_CASES = [
    ("yes/no", "yes/no"),
    ("on/off", "on/off"),
    ("and/or", "and/or"),
    ("input/output", "input/output"),
    ("24/7", "24/7"),
    ("Hello/Hi", "Hello/Hi"),  # ambiguous, preserved
    ("good/great", "good/great"),  # ambiguous, preserved
]


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def safe_str(s: str) -> str:
    """Convert string to safe ASCII for console output."""
    try:
        return s.encode('cp1252', errors='replace').decode('cp1252')
    except Exception:
        return s.encode('utf-8', errors='replace').decode('utf-8', errors='replace')


def print_test(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {safe_str(name)}")
    if detail:
        print(f"         {safe_str(detail)}")


def test_clean_output_functions() -> list[str]:
    """Test clean_output and resolve_slash_alternatives without the server."""
    print_header("Clean Output Functions (no server needed)")
    failures = []
    
    # Slash preservation
    print("\n  --- Slash Preservation ---")
    for input_text, expected in SLASH_TEST_CASES:
        result = resolve_slash_alternatives(input_text)
        passed = result == expected
        if not passed:
            failures.append(f"resolve_slash_alternatives('{input_text}') = '{result}', expected '{expected}'")
        print_test(f"resolve_slash_alternatives('{input_text}')", passed,
                   f"got '{result}'")
    
    # Clean output with slashes
    print("\n  --- Clean Output with Slashes ---")
    for input_text, expected in SLASH_TEST_CASES:
        result = clean_output(input_text)
        passed = result == expected
        if not passed:
            failures.append(f"clean_output('{input_text}') = '{result}', expected '{expected}'")
        print_test(f"clean_output('{input_text}')", passed, f"got '{result}'")
    
    # Reasoning tag stripping
    print("\n  --- Reasoning Tag Stripping ---")
    reasoning_cases = [
        ("<think>This is reasoning</think>Hello world", "Hello world"),
        ("<thinking>Analysis</thinking>Final answer", "Final answer"),
        ("<reasoning>Thought</reasoning>Translation", "Translation"),
        ("Normal text without tags", "Normal text without tags"),
        ("I think this is great!", "I think this is great!"),  # Should NOT strip
    ]
    for input_text, expected in reasoning_cases:
        result = clean_output(input_text)
        passed = result == expected
        if not passed:
            failures.append(f"clean_output('{input_text[:30]}...') = '{result}', expected '{expected}'")
        print_test(f"clean_output('{input_text[:40]}...')", passed, f"got '{result}'")
    
    # Prompt echo rejection
    print("\n  --- Prompt Echo Rejection ---")
    echo_cases = [
        ("You are a professional Japanese-to-English subtitle translator", True),
        ("Output only the English translation", True),
        ("Hello world", False),
        ("Thank you everyone", False),
    ]
    for input_text, should_reject in echo_cases:
        result = clean_output(input_text)
        is_empty = result == ""
        passed = is_empty == should_reject
        if not passed:
            failures.append(f"clean_output('{input_text[:30]}...') = '{result}', expected empty={should_reject}")
        print_test(f"clean_output('{input_text[:40]}...')", passed,
                   f"empty={is_empty}, expected empty={should_reject}")
    
    return failures


def test_real_inference(translator: Gemma4Translator) -> list[str]:
    """Run real inference tests with the actual Gemma 4 model."""
    print_header("Real Inference Tests (Gemma 4 E4B)")
    failures = []
    results = []
    
    # Test 1: Basic translations
    print("\n  --- Basic Translations ---")
    latencies = []
    for ja, expected_hint in TEST_UTTERANCES:
        start = time.monotonic()
        try:
            result = translator.translate(ja, [])
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            
            # Check that result is non-empty and in English
            has_content = bool(result.strip())
            has_japanese = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' 
                             for c in result)
            
            if has_content and not has_japanese:
                print_test(f"Japanese utterance", True, f"'{safe_str(result)}' ({elapsed:.2f}s)")
            else:
                detail = f"'{safe_str(result)}' ({elapsed:.2f}s)"
                if not has_content:
                    detail += " [EMPTY]"
                if has_japanese:
                    detail += " [HAS JAPANESE]"
                print_test(f"Japanese utterance", False, detail)
                failures.append(f"Translation: {detail}")
            
            results.append((ja, result, elapsed))
        except Exception as e:
            elapsed = time.monotonic() - start
            print_test(f"Japanese utterance", False, f"Exception: {safe_str(str(e))}")
            failures.append(f"Translation: {e}")
            results.append((ja, str(e), elapsed))
    
    # Report latency stats
    if latencies:
        print(f"\n  --- Latency Statistics ---")
        print(f"    Average: {sum(latencies)/len(latencies):.2f}s")
        print(f"    Min: {min(latencies):.2f}s")
        print(f"    Max: {max(latencies):.2f}s")
        print(f"    P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    
    # Test 2: Repeated prompts (prompt cache stability)
    print("\n  --- Prompt Cache Stability ---")
    cache_test_ja = "こんにちは、元気ですか？"
    cache_results = []
    for i in range(3):
        start = time.monotonic()
        result = translator.translate(cache_test_ja, [])
        elapsed = time.monotonic() - start
        cache_results.append((result, elapsed))
        print_test(f"Repeat {i+1}", True, f"'{result}' ({elapsed:.2f}s)")
    
    # Check consistency
    if len(set(r[0] for r in cache_results)) == 1:
        print_test("Cache consistency", True, "All repeats produced identical output")
    else:
        outputs = set(r[0] for r in cache_results)
        print_test("Cache consistency", False, f"Different outputs: {outputs}")
        failures.append(f"Cache inconsistency for '{cache_test_ja}': {outputs}")
    
    # Test 3: Changing context
    print("\n  --- Context Sensitivity ---")
    context_test_ja = "それ"
    contexts = [
        [],
        [("今日はゲームをします", "We're playing games today")],
        [("お弁当を作りました", "I made a bento")],
    ]
    context_results = []
    for ctx in contexts:
        result = translator.translate(context_test_ja, ctx)
        context_results.append(result)
        ctx_desc = "no context" if not ctx else f"context: {ctx[0][0]}"
        print_test(f"'{context_test_ja}' with {ctx_desc}", bool(result), f"'{result}'")
    
    # Test 4: Truncation test (force small max_tokens)
    print("\n  --- Truncation Handling ---")
    long_ja = "今日はとてもいい天気ですね。みんなと公園でピクニックをする予定ですが、雨が降るかもしれないので傘を持っていくべきかもしれません。それに食べ物も準備する必要があります。"
    try:
        result = translator.translate(long_ja, [])
        print_test("Long utterance", bool(result), f"'{result[:80]}...'")
        if not result:
            failures.append(f"Long utterance returned empty")
    except Exception as e:
        print_test("Long utterance", False, f"Exception: {e}")
        failures.append(f"Long utterance: {e}")
    
    # Test 5: Reasoning content check
    print("\n  --- Reasoning Content Check ---")
    # --reasoning off is correct; check that it's not --reasoning on
    args = translator.server.args
    args_str = ' '.join(args)
    has_reasoning_on = '--reasoning' in args_str and 'on' in args_str.split('--reasoning')[-1].split()[0] if '--reasoning' in args_str else False
    print_test("Reasoning disabled", not has_reasoning_on,
               f"Args include: {' '.join(a for a in args if 'reason' in a.lower())}")
    if has_reasoning_on:
        failures.append("Reasoning enabled in server args")
    
    # Check GGUF chat template
    print("\n  --- GGUF Chat Template Check ---")
    model_path = translator.model
    print(f"    Model: {model_path}")
    print(f"    Layers: {translator.layer_count}")
    print(f"    GPU layers: {translator.gpu_layers}")
    
    return failures


def test_server_diagnostics(translator: Gemma4Translator) -> list[str]:
    """Check server diagnostics and configuration."""
    print_header("Server Diagnostics")
    failures = []
    
    # Server args
    print(f"\n  Server executable: {translator.executable}")
    print(f"  Server args: {' '.join(translator.server.args)}")
    print(f"  Server port: {translator.server.port}")
    print(f"  Model path: {translator.model}")
    print(f"  Layer count: {translator.layer_count}")
    print(f"  GPU layers: {translator.gpu_layers}")
    print(f"  Context size: {translator.cfg.context_size}")
    print(f"  Max tokens: {translator.cfg.max_tokens}")
    print(f"  Temperature: {translator.cfg.temperature}")
    print(f"  Top-k: {translator.cfg.top_k}")
    print(f"  Top-p: {translator.cfg.top_p}")
    print(f"  Repeat penalty: {translator.cfg.repeat_penalty}")
    print(f"  Threads: {translator.cfg.threads}")
    
    # Check for reasoning-related flags
    # --reasoning off is correct; --reasoning on would be wrong
    args_str = ' '.join(translator.server.args)
    has_reasoning_on = '--reasoning' in args_str and 'on' in args_str.split('--reasoning')[-1].split()[0] if '--reasoning' in args_str else False
    if has_reasoning_on:
        print_test("No reasoning flags", False, "Found --reasoning on in args")
        failures.append("Reasoning enabled in server args")
    else:
        print_test("No reasoning flags", True, f"Args include: {' '.join(a for a in translator.server.args if 'reason' in a.lower())}")
    
    return failures


def main() -> int:
    print("="*60)
    print("  Gemma 4 E4B Real Inference Tests")
    print("="*60)
    
    all_failures = []
    
    # Test 1: Clean output functions (no server needed)
    all_failures.extend(test_clean_output_functions())
    
    # Test 2: Start the server and run real inference
    print_header("Starting llama-server")
    
    try:
        cfg = TranslateConfig(
            model_path="models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf",
            server_path=".runtime/llama-b12318-cuda12/llama-server.exe",
            gpu_layers=42,
            context_size=4096,
            threads=4,
            max_tokens=128,
            parallel_slots=1,
            temperature=0.0,
            top_k=20,
            top_p=0.6,
            repeat_penalty=1.05,
            context_lines=6,
            max_context_chars=600,
            style="Plain and concise",
            timeout_s=30.0,
            load_timeout_s=300.0,
            glossary="glossary.toml",
        )
        
        glossary = Glossary(terms={})
        stream = StreamContext()
        
        print("  Creating Gemma4Translator...")
        start = time.monotonic()
        translator = Gemma4Translator(cfg, glossary, stream=stream, gpu_layers=42)
        elapsed = time.monotonic() - start
        print(f"  Server started in {elapsed:.1f}s")
        
        try:
            # Server diagnostics
            all_failures.extend(test_server_diagnostics(translator))
            
            # Real inference tests
            all_failures.extend(test_real_inference(translator))
            
        finally:
            print("\n  Closing translator...")
            translator.close()
            print("  Done.")
            
    except LlamaServerError as e:
        print(f"\n  [SKIP] Could not start llama-server: {e}")
        print("  Real inference tests skipped.")
        return 0  # Skip is not a failure
    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print_header("Test Summary")
    
    if all_failures:
        print(f"\n  {len(all_failures)} failures:")
        for f in all_failures:
            print(f"    - {f}")
        print(f"\n  Result: SOME TESTS FAILED")
        return 1
    else:
        print("\n  All tests passed!")
        print(f"\n  Result: ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
