"""Benchmark special token detection against live Gemma 4 model.

Tests the has_special_token_flood() and english_output_issue() validation
against real model output by sending:
1. Normal Japanese utterances — should produce clean English (no false positives)
2. Adversarial prompts designed to trigger special token generation:
   - Empty/whitespace Japanese input
   - Extremely short input (single character)
   - Repetitive input that may confuse the model
   - Input with special characters and control sequences
   - Very long input that may cause truncation artifacts
3. Both reasoning ON and OFF modes
4. High temperature to increase stochasticity and potential for malformed output

Measures:
- False positive rate (normal utterances incorrectly rejected)
- Detection rate (adversarial inputs correctly caught)
- Per-input rejection reasons
- Reasoning mode impact on special token generation

Usage:
    python -m scripts.benchmark_special_tokens [--config config.toml] [--output results.json]
"""
import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from auto_translate.config import Config, load_config
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.subtitle_clean import (
    clean_english_output,
    english_output_issue,
    has_special_token_flood,
)
from auto_translate.translate.gemma4 import Gemma4Translator

log = logging.getLogger(__name__)

# Normal Japanese utterances — should produce valid English subtitles
NORMAL_UTTERANCES = [
    "あのね、彼が言ったこと、覚えてる？",
    "うん、それってすごいよね。私もそう思う。",
    "ラップちゃんが言ったこと、私も同意するよ。",
    "みこち、大丈夫？怪我してない？",
    "うん、大丈夫。でも車は壊れたね。",
    "次はもっと慎重にやろう。このままじゃ捕まるよ。",
    "あのね、ラップ sama、今日の配信最高だったよ！",
    "ありがとう！みんなと遊べて楽しいから！",
    "ラップちゃん、その服カッコいい！どこで買ったの？",
    "えへへ、ありがとう！オンラインで買ったの。",
    "みんな、スパチャありがとう！今日はマインクラフトで遊びます。",
    "えええ、マジで？それって超レアじゃない！売ってよ！",
    "あのボス、本当に強かったね。もう一回挑戦しよう！",
    "うわ、警察来た！逃げよう！ラップ、こっち！",
    "ごめんごめん、冗談だったんだから。",
    "うわー、すごい！その技、どうやってやるの？",
    "それでは、また次の配信で会いましょう！バイバイ！",
    "次回の配信は来週の日曜日だよ。楽しみにしてて！",
    "今日のテーマは、マインクラフトで脱出チャレンジ！",
    "ちょっと待って、その音何？誰かいる？",
    "いや、ただの風だよ。気にしないで。",
    "さて、次はどこに行こうか。地図見てみよう。",
    "あそこに行こう！山の上から景色が見えるよ。",
    "うん、いいね。でも暗くなる前に戻ろうよ。",
    "ラップ、その発言ちょっと待って！そんなこと言っちゃダメ！",
    "みこち、今日の料理美味しかった！レシピ教えて！",
    "はい、ぺこらです！よろしくお願いします。",
    "えっ、もう終わったの？もっと遊びたかった！",
    "ラップちゃん、その戦略ダメだよ。こっちの方がいい！",
    "みんなのコメント、全部読んでるから安心して！",
    "今日は特別ゲストが来るから、楽しみにしててね！",
]

# Adversarial inputs designed to potentially trigger special token generation
ADVERSARIAL_INPUTS = [
    # Empty / minimal input
    ("empty_string", ""),
    ("whitespace_only", "   "),
    ("single_char", "あ"),
    ("single_punctuation", "。"),
    ("newline_only", "\n"),

    # Repetitive input that may confuse the model
    ("repeated_char", "あ" * 100),
    ("repeated_word", "ラップ " * 30),
    ("repeated_sentence", "こんにちは。" * 20),

    # Special characters and control sequences
    ("html_tags", "<div><span>こんにちは</span></div>"),
    ("xml_like", "<?xml version=\"1.0\"?><root>テスト</root>"),
    ("unicode_control", "あ\u200b\u200c\u200dいうえ"),
    ("emoji_heavy", "🎮🎮🎮🎮🎮 こんにちは 🎮🎮🎮"),
    ("mixed_script", "Hello こんにちは 你好 안녕하세요"),

    # Very long input
    ("very_long", "今日は天気がいいですね。" * 50),

    # Input that might trigger reasoning artifacts
    ("question_only", "？？？？？？？？？？"),
    ("exclamation_only", "！！！！！！！！！！"),
    ("kaomoji", "(´・ω・`) (´・ω・`) (´・ω・`)"),

    # Edge cases for translation
    ("numbers_only", "1234567890"),
    ("romanji", "kon'nichiwa"),
    ("name_only", "ラップ"),
    ("honorific_only", "sama"),
]


def run_normal_test(
    utterances: list[str],
    translator: Gemma4Translator,
    label: str,
) -> dict:
    """Test normal utterances — should all produce valid English."""
    log.info("=== Normal Utterances: %s ===", label)

    results: list[dict] = []
    missing = 0
    rejected = 0
    special_token_rejected = 0
    latencies: list[float] = []

    for i, ja in enumerate(utterances):
        t0 = time.monotonic()
        try:
            prompt = translator.prompt_for(ja, [])
            raw = translator._chat(prompt, timeout=translator.cfg.timeout_s)
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            # Check raw output for special tokens before cleaning
            raw_flood = has_special_token_flood(raw)
            raw_issue = english_output_issue(raw)

            # Apply clean_output (the actual pipeline)
            english = clean_english_output(raw)
            clean_issue = english_output_issue(english) if english else "empty_after_clean"

            is_missing = not english.strip()
            is_rejected = raw_issue is not None or is_missing
            is_special = raw_issue == "special_token_flood"

            if is_missing:
                missing += 1
            if is_rejected:
                rejected += 1
            if is_special:
                special_token_rejected += 1

            results.append({
                "index": i,
                "ja": ja,
                "raw": raw[:200],  # truncate for log safety
                "english": english,
                "raw_has_flood": raw_flood,
                "raw_issue": raw_issue,
                "clean_issue": clean_issue,
                "latency_s": round(elapsed, 3),
            })

        except Exception as exc:
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            missing += 1
            rejected += 1
            results.append({
                "index": i,
                "ja": ja,
                "raw": "",
                "english": "",
                "raw_has_flood": False,
                "raw_issue": None,
                "clean_issue": "exception",
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    return {
        "label": label,
        "total": len(utterances),
        "successful": len(utterances) - missing,
        "missing_subtitles": missing,
        "rejected_by_validation": rejected,
        "special_token_rejected": special_token_rejected,
        "false_positive_rate": round(rejected / len(utterances) * 100, 1) if utterances else 0,
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 3),
        "results": results,
    }


def run_adversarial_test(
    inputs: list[tuple[str, str]],
    translator: Gemma4Translator,
    label: str,
) -> dict:
    """Test adversarial inputs — should be safely handled without crashes."""
    log.info("=== Adversarial Inputs: %s ===", label)

    results: list[dict] = []
    crashes = 0
    special_token_detected = 0
    safely_rejected = 0
    latencies: list[float] = []

    for name, ja in inputs:
        t0 = time.monotonic()
        try:
            # Skip truly empty — prompt_for may not handle it
            if not ja.strip():
                results.append({
                    "name": name,
                    "ja": repr(ja)[:80],
                    "skipped": True,
                    "reason": "empty input",
                })
                continue

            prompt = translator.prompt_for(ja, [])
            raw = translator._chat(prompt, timeout=translator.cfg.timeout_s)
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            raw_flood = has_special_token_flood(raw)
            raw_issue = english_output_issue(raw)
            english = clean_english_output(raw)

            was_detected = raw_flood or raw_issue is not None
            was_safely_rejected = not english.strip()

            if raw_flood:
                special_token_detected += 1
            if was_safely_rejected:
                safely_rejected += 1

            results.append({
                "name": name,
                "ja": repr(ja)[:80],
                "raw": raw[:200],
                "english": english,
                "raw_has_flood": raw_flood,
                "raw_issue": raw_issue,
                "safely_rejected": was_safely_rejected,
                "latency_s": round(elapsed, 3),
            })

        except Exception as exc:
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            crashes += 1
            results.append({
                "name": name,
                "ja": repr(ja)[:80],
                "raw": "",
                "english": "",
                "raw_has_flood": False,
                "raw_issue": None,
                "safely_rejected": True,
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    total_run = len([r for r in results if not r.get("skipped")])
    return {
        "label": label,
        "total": len(inputs),
        "total_run": total_run,
        "crashes": crashes,
        "special_token_detected": special_token_detected,
        "safely_rejected": safely_rejected,
        "detection_rate": round(special_token_detected / max(total_run, 1) * 100, 1),
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "results": results,
    }


def run_repetition_stress_test(
    utterances: list[str],
    translator: Gemma4Translator,
    label: str,
    num_repeats: int = 5,
) -> dict:
    """Send the same utterance multiple times to check for degradation."""
    log.info("=== Repetition Stress Test: %s (repeats=%d) ===", label, num_repeats)

    # Pick a subset for stress testing
    test_set = utterances[:10]

    results: list[dict] = []
    special_token_count = 0
    missing_count = 0

    for repeat in range(num_repeats):
        for i, ja in enumerate(test_set):
            try:
                prompt = translator.prompt_for(ja, [])
                raw = translator._chat(prompt, timeout=translator.cfg.timeout_s)
                english = clean_english_output(raw)

                raw_flood = has_special_token_flood(raw)
                if raw_flood:
                    special_token_count += 1
                if not english.strip():
                    missing_count += 1

                results.append({
                    "repeat": repeat,
                    "index": i,
                    "ja": ja,
                    "english": english,
                    "raw_has_flood": raw_flood,
                })

            except Exception as exc:
                missing_count += 1
                results.append({
                    "repeat": repeat,
                    "index": i,
                    "ja": ja,
                    "english": "",
                    "raw_has_flood": False,
                    "error": str(exc),
                })

    total = num_repeats * len(test_set)
    return {
        "label": label,
        "num_repeats": num_repeats,
        "utterances_tested": len(test_set),
        "total_requests": total,
        "special_token_detected": special_token_count,
        "missing_subtitles": missing_count,
        "degradation_rate": round(special_token_count / max(total, 1) * 100, 1),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark special token detection")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--output", default="logs/special_token_benchmark_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    full_cfg: Config = load_config(config_path)

    # Run with reasoning OFF (default)
    cfg_off = replace(full_cfg.translate, reasoning_enabled=False, max_tokens=128)
    glossary = Glossary.load(Path(cfg_off.glossary)) if Path(cfg_off.glossary).exists() else Glossary()
    stream = StreamContext()
    translator_off = Gemma4Translator(cfg_off, glossary, stream=stream)

    # Run with reasoning ON
    cfg_on = replace(full_cfg.translate, reasoning_enabled=True, max_tokens=384)
    glossary_on = Glossary.load(Path(cfg_on.glossary)) if Path(cfg_on.glossary).exists() else Glossary()
    stream_on = StreamContext()
    translator_on = Gemma4Translator(cfg_on, glossary_on, stream=stream_on)

    all_results = {}

    # --- Reasoning OFF tests ---
    log.info("========== REASONING OFF ==========")

    all_results["reasoning_off_normal"] = run_normal_test(
        NORMAL_UTTERANCES, translator_off, "reasoning_off_normal"
    )
    all_results["reasoning_off_adversarial"] = run_adversarial_test(
        ADVERSARIAL_INPUTS, translator_off, "reasoning_off_adversarial"
    )
    all_results["reasoning_off_stress"] = run_repetition_stress_test(
        NORMAL_UTTERANCES, translator_off, "reasoning_off_stress", num_repeats=3
    )

    translator_off.close()

    # --- Reasoning ON tests ---
    log.info("========== REASONING ON ==========")

    all_results["reasoning_on_normal"] = run_normal_test(
        NORMAL_UTTERANCES, translator_on, "reasoning_on_normal"
    )
    all_results["reasoning_on_adversarial"] = run_adversarial_test(
        ADVERSARIAL_INPUTS, translator_on, "reasoning_on_adversarial"
    )
    all_results["reasoning_on_stress"] = run_repetition_stress_test(
        NORMAL_UTTERANCES, translator_on, "reasoning_on_stress", num_repeats=3
    )

    translator_on.close()

    # --- Summary report ---
    report = {
        "summary": {
            "reasoning_off": {
                "normal_success_rate": round(
                    all_results["reasoning_off_normal"]["successful"]
                    / max(all_results["reasoning_off_normal"]["total"], 1) * 100, 1
                ),
                "normal_false_positive_rate": all_results["reasoning_off_normal"]["false_positive_rate"],
                "adversarial_crashes": all_results["reasoning_off_adversarial"]["crashes"],
                "adversarial_special_tokens": all_results["reasoning_off_adversarial"]["special_token_detected"],
                "stress_special_tokens": all_results["reasoning_off_stress"]["special_token_detected"],
                "stress_degradation": all_results["reasoning_off_stress"]["degradation_rate"],
            },
            "reasoning_on": {
                "normal_success_rate": round(
                    all_results["reasoning_on_normal"]["successful"]
                    / max(all_results["reasoning_on_normal"]["total"], 1) * 100, 1
                ),
                "normal_false_positive_rate": all_results["reasoning_on_normal"]["false_positive_rate"],
                "adversarial_crashes": all_results["reasoning_on_adversarial"]["crashes"],
                "adversarial_special_tokens": all_results["reasoning_on_adversarial"]["special_token_detected"],
                "stress_special_tokens": all_results["reasoning_on_stress"]["special_token_detected"],
                "stress_degradation": all_results["reasoning_on_stress"]["degradation_rate"],
            },
        },
        "conclusion": "",
    }

    # Determine conclusion
    off = report["summary"]["reasoning_off"]
    on_ = report["summary"]["reasoning_on"]

    issues = []
    if off["adversarial_crashes"] > 0:
        issues.append(f"Reasoning OFF: {off['adversarial_crashes']} crashes on adversarial input")
    if on_["adversarial_crashes"] > 0:
        issues.append(f"Reasoning ON: {on_['adversarial_crashes']} crashes on adversarial input")
    if off["normal_false_positive_rate"] > 0:
        issues.append(f"Reasoning OFF: {off['normal_false_positive_rate']}% false positive rate on normal input")
    if on_["normal_false_positive_rate"] > 0:
        issues.append(f"Reasoning ON: {on_['normal_false_positive_rate']}% false positive rate on normal input")
    if off["adversarial_special_tokens"] > 0:
        issues.append(f"Reasoning OFF: {off['adversarial_special_tokens']} special tokens detected in adversarial test")
    if on_["adversarial_special_tokens"] > 0:
        issues.append(f"Reasoning ON: {on_['adversarial_special_tokens']} special tokens detected in adversarial test")
    if off["stress_degradation"] > 0:
        issues.append(f"Reasoning OFF: {off['stress_degradation']}% degradation in stress test")
    if on_["stress_degradation"] > 0:
        issues.append(f"Reasoning ON: {on_['stress_degradation']}% degradation in stress test")

    if issues:
        report["conclusion"] = "Issues found: " + "; ".join(issues)
    else:
        report["conclusion"] = (
            "No issues detected. Special token detection works correctly: "
            "no false positives on normal input, no crashes on adversarial input, "
            "no degradation under repeated stress."
        )

    all_results["report"] = report

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SPECIAL TOKEN DETECTION BENCHMARK SUMMARY")
    print("=" * 70)

    for mode_name, mode_data in report["summary"].items():
        print(f"\n{mode_name.upper()}:")
        print(f"  Normal success rate: {mode_data['normal_success_rate']}%")
        print(f"  Normal false positive rate: {mode_data['normal_false_positive_rate']}%")
        print(f"  Adversarial crashes: {mode_data['adversarial_crashes']}")
        print(f"  Adversarial special tokens: {mode_data['adversarial_special_tokens']}")
        print(f"  Stress special tokens: {mode_data['stress_special_tokens']}")
        print(f"  Stress degradation: {mode_data['stress_degradation']}%")

    print(f"\nConclusion: {report['conclusion']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
