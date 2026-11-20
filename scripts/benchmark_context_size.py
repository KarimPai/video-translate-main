"""Benchmark Gemma 4 context-size accuracy: 4096 vs 8192 comparison.

Compares translation quality and consistency at different context sizes
to investigate whether reduced context affects Gemma 4 accuracy.

Uses Japanese conversational dialogue containing:
- Ambiguous pronouns
- Multiple speakers
- VTuber names and nicknames
- Game-specific terminology
- Emotional or colloquial expressions
- References to earlier dialogue

Usage:
    python -m scripts.benchmark_context_size [--config config.toml] [--output results.json]
"""
import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from auto_translate.config import Config, TranslateConfig, load_config
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.translate.gemma4 import Gemma4Translator, clean_output

log = logging.getLogger(__name__)

# Japanese utterances designed to test context dependency, covering edge cases.
CONTEXT_SENSITIVE_UTTERANCES = [
    # Ambiguous pronouns — need context to resolve
    ("あのね、彼が言ったこと、覚えてる？", "Um, do you remember what he said?",
     "Pronoun '彼' (he) — ambiguous without prior context"),
    ("うん、それってすごいよね。私もそう思う。", "Yeah, that's amazing. I think so too.",
     "Pronoun 'それ' (that) — refers to previous dialogue"),
    ("ラップちゃんが言ったこと、私も同意するよ。", "I agree with what Lap-chan said.",
     "Reference to another speaker's statement"),

    # Multiple speakers with name references
    ("みこち、大丈夫？怪我してない？", "Mikochi, are you okay? You not hurt, right?",
     "Direct address to named speaker"),
    ("うん、大丈夫。でも車は壊れたね。", "Yeah, I'm fine. But the car's wrecked.",
     "Response from addressed speaker"),
    ("次はもっと慎重にやろう。このままじゃ捕まるよ。", "Let's be more careful next time. We'll get caught if we keep this up.",
     "Continuation of multi-speaker scene"),

    # VTuber names and nicknames
    ("あのね、ラップ sama、今日の配信最高だったよ！", "Um, Lap-sama, today's stream was the best!",
     "VTuber name with honorific"),
    ("ありがとう！みんなと遊べて楽しいから！", "Thanks! It's fun playing with everyone!",
     "Response from addressed VTuber"),
    ("ラップちゃん、その服カッコいい！どこで買ったの？", "Lap-chan, that outfit is cool! Where did you buy it?",
     "Nickname variant 'ちゃん' vs 'sama'"),
    ("えへへ、ありがとう！オンラインで買ったの。", "Hehe, thanks! I bought it online.",
     "Casual response with laugh"),

    # Game-specific terminology
    ("みんな、スパチャありがとう！今日はマインクラフトで遊びます。", "Everyone, thanks for the super chats! We're playing Minecraft today.",
     "Game term 'スパチャ' (super chat)"),
    ("えええ、マジで？それって超レアじゃない！売ってよ！", "Whaaat, seriously? That's super rare! Sell it to me!",
     "Game context — rare item"),
    ("あのボス、本当に強かったね。もう一回挑戦しよう！", "That boss was really strong. Let's try again!",
     "Game terminology — boss fight"),

    # Emotional or colloquial expressions
    ("うわ、警察来た！逃げよう！ラップ、こっち！", "Whoa, the police are here! Let's run! Lap, this way!",
     "Urgent emotional expression"),
    ("ごめんごめん、冗談だったんだから。", "Sorry sorry, it was just a joke.",
     "Colloquial apology"),
    ("うわー、すごい！その技、どうやってやるの？", "Whoa, amazing! How do you do that move?",
     "Excited reaction"),

    # References to earlier dialogue
    ("それでは、また次の配信で会いましょう！バイバイ！", "Well then, see you in the next stream! Bye-bye!",
     "Closing reference to stream context"),
    ("次回の配信は来週の日曜日だよ。楽しみにしてて！", "Next stream is next Sunday. Look forward to it!",
     "Reference to future event"),
    ("今日のテーマは、マインクラフトで脱出チャレンジ！", "Today's theme is a Minecraft escape challenge!",
     "Reference to today's theme"),

    # Additional context-dependent utterances
    ("ちょっと待って、その音何？誰かいる？", "Wait a sec, what's that sound? Is someone there?",
     "Context-dependent question"),
    ("いや、ただの風だよ。気にしないで。", "No, it's just the wind. Don't worry about it.",
     "Response to previous question"),
    ("さて、次はどこに行こうか。地図見てみよう。", "Alright, where should we go next. Let's check the map.",
     "Continuation of exploration"),
    ("あそこに行こう！山の上から景色が見えるよ。", "Let's go over there! You can see the view from the mountain top.",
     "Suggestion with context"),
    ("うん、いいね。でも暗くなる前に戻ろうよ。", "Yeah, sounds good. But let's come back before it gets dark.",
     "Agreement with condition"),
    ("ラップ、その発言ちょっと待って！そんなこと言っちゃダメ！", "Lap, hold on with that comment! You can't say stuff like that!",
     "Interruption in dialogue"),
    ("みこち、今日の料理美味しかった！レシピ教えて！", "Mikochi, today's cooking was delicious! Teach me the recipe!",
     "Reference to today's event"),
    ("はい、ぺこらです！よろしくお願いします。", "Yes, I'm Pekora! Nice to meet you.",
     "Self-introduction"),
    ("えっ、もう終わったの？もっと遊びたかった！", "Huh, it's already over? I wanted to play more!",
     "Reaction to ending"),
    ("ラップちゃん、その戦略ダメだよ。こっちの方がいい！", "Lap-chan, that strategy won't work. This one is better!",
     "Strategic discussion"),
    ("みんなのコメント、全部読んでるから安心して！", "I'm reading all your comments, so don't worry!",
     "Stream interaction"),
    ("今日は特別ゲストが来るから、楽しみにしててね！", "A special guest is coming today, so look forward to it!",
     "Teaser for upcoming event"),
]


def percentile(values: list[float], p: float) -> float:
    """Calculate the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    index = int(n * p / 100.0)
    index = min(index, n - 1)
    return sorted_vals[index]


def chrf_counts(reference: str, hypothesis: str, order: int = 6) -> list[tuple[int, int, int]]:
    """Per-order chrF n-gram counts."""
    from collections import Counter
    ref, hyp = "".join(reference.split()), "".join(hypothesis.split())
    counts = []
    for n in range(1, order + 1):
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        counts.append((sum((r & h).values()), sum(h.values()), sum(r.values())))
    return counts


def chrf_score(pairs: list[tuple[str, str]]) -> float | None:
    """Corpus chrF score on a 0-100 scale."""
    if not pairs:
        return None
    totals = [[0, 0, 0] for _ in range(6)]
    for reference, hypothesis in pairs:
        for total, counts in zip(totals, chrf_counts(reference, hypothesis)):
            for i, value in enumerate(counts):
                total[i] += value
    orders = []
    for match, hyp, ref in totals:
        if ref:
            orders.append((match / hyp if hyp else 0.0, match / ref))
    if not orders:
        return 0.0
    precision = sum(p for p, _ in orders) / len(orders)
    recall = sum(r for _, r in orders) / len(orders)
    if not precision + recall:
        return 0.0
    beta2 = 4.0
    return 100 * (1 + beta2) * precision * recall / (beta2 * precision + recall)


def run_context_test(
    samples: list[tuple[str, str | None, str | None]],
    context_size: int,
    label: str,
    config_path: Path = Path("config.toml"),
) -> dict:
    """Run translations with the given context size configuration."""
    log.info("=== Context Size Benchmark: %s (context=%d) ===", label, context_size)

    full_cfg: Config = load_config(config_path)
    # Override context size for this test — work with the translate section
    cfg = replace(full_cfg.translate, context_size=context_size, max_tokens=128)

    glossary = Glossary.load(Path(cfg.glossary)) if Path(cfg.glossary).exists() else Glossary()
    stream = StreamContext()

    translator = Gemma4Translator(cfg, glossary, stream=stream)

    history: list[tuple[str, str]] = []
    latencies: list[float] = []
    results: list[dict] = []
    missing = 0
    truncated = 0

    total_start = time.monotonic()

    for i, (ja, ref_en, note) in enumerate(samples):
        t0 = time.monotonic()
        try:
            prompt = translator.prompt_for(ja, history)
            english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            if not english:
                missing += 1

            result = {
                "index": i,
                "ja": ja,
                "en": english,
                "reference_en": ref_en or "",
                "note": note or "",
                "latency_s": round(elapsed, 3),
                "error": None,
            }
            results.append(result)

            if english and ja:
                history.append((ja, english))
                if len(history) > translator.cfg.context_lines:
                    history.pop(0)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            missing += 1
            results.append({
                "index": i,
                "ja": ja,
                "en": "",
                "reference_en": ref_en or "",
                "note": note or "",
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    total_elapsed = time.monotonic() - total_start

    # Compute chrF for samples with reference English
    pairs = [(r["reference_en"], r["en"]) for r in results
             if r.get("reference_en") and r.get("en") and not r.get("error")]
    chrf = chrf_score(pairs) if pairs else None

    # Check consistency: run a subset again and compare
    consistency_mismatches = 0
    if len(results) >= 10:
        log.info("Running consistency check (repeating first 10 utterances)...")
        for r in results[:10]:
            try:
                prompt = translator.prompt_for(r["ja"], [])
                english2 = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
                if english2 != r["en"]:
                    consistency_mismatches += 1
                    log.info("Consistency mismatch: '%s' vs '%s'", r["en"], english2)
            except Exception:
                consistency_mismatches += 1

    translator.close()

    summary = {
        "label": label,
        "context_size": context_size,
        "total_samples": len(samples),
        "successful": len(results) - missing,
        "missing_subtitles": missing,
        "total_time_s": round(total_elapsed, 3),
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
        "chrf": round(chrf, 2) if chrf is not None else None,
        "consistency_mismatches": consistency_mismatches,
        "results": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 context-size accuracy")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    samples = CONTEXT_SENSITIVE_UTTERANCES

    # Test 4096 context (current default)
    result_4096 = run_context_test(samples, 4096, "context_4096")

    # Test 8192 context if feasible
    result_8192 = None
    try:
        result_8192 = run_context_test(samples, 8192, "context_8192")
    except Exception as exc:
        log.warning("Could not run 8192 context test: %s", exc)
        result_8192 = {"error": str(exc), "context_size": 8192}

    # Comparison report
    lines = [
        "=" * 70,
        "GEMMA 4 CONTEXT-SIZE BENCHMARK: 4096 vs 8192",
        "=" * 70,
        "",
        f"{'Metric':<40} {'4096 context':>14} {'8192 context':>14} {'Change':>12}",
        "-" * 70,
    ]

    if result_8192 and "error" not in result_8192:
        metrics = [
            ("Total samples", result_4096["total_samples"], result_8192["total_samples"]),
            ("Successful translations", result_4096["successful"], result_8192["successful"]),
            ("Missing subtitles", result_4096["missing_subtitles"], result_8192["missing_subtitles"]),
            ("Total time (s)", result_4096["total_time_s"], result_8192["total_time_s"]),
            ("Latency avg (s)", result_4096["latency_avg_s"], result_8192["latency_avg_s"]),
            ("Latency p50 (s)", result_4096["latency_p50_s"], result_8192["latency_p50_s"]),
            ("Latency p95 (s)", result_4096["latency_p95_s"], result_8192["latency_p95_s"]),
            ("Latency max (s)", result_4096["latency_max_s"], result_8192["latency_max_s"]),
            ("chrF score", result_4096["chrf"] or 0, result_8192["chrf"] or 0),
            ("Consistency mismatches", result_4096["consistency_mismatches"], result_8192["consistency_mismatches"]),
        ]

        for name, v1, v2 in metrics:
            if isinstance(v1, float) and isinstance(v2, float) and v2 != 0:
                change = ((v1 - v2) / abs(v2)) * 100
                change_str = f"{change:+.1f}%"
            elif isinstance(v1, int) and isinstance(v2, int) and v2 != 0:
                change = ((v1 - v2) / abs(v2)) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = ""

            if isinstance(v1, float):
                lines.append(f"{name:<40} {v1:>14.3f} {v2:>14.3f} {change_str:>12}")
            else:
                lines.append(f"{name:<40} {v1:>14} {v2:>14} {change_str:>12}")

    lines.extend([
        "",
        "=" * 70,
        "CONCLUSION",
        "=" * 70,
    ])

    if result_8192 and "error" not in result_8192:
        chrf_4096 = result_4096.get("chrf") or 0
        chrf_8192 = result_8192.get("chrf") or 0
        if chrf_8192 > chrf_4096 + 2:
            lines.append(f"8192 context shows measurable improvement (chrF {chrf_8192:.1f} vs {chrf_4096:.1f})")
        else:
            lines.append(f"No measurable improvement from 8192 context (chrF {chrf_8192:.1f} vs {chrf_4096:.1f})")
            lines.append("Recommendation: Keep 4096 as default to save VRAM.")
    else:
        lines.append("8192 context test failed — cannot compare.")
        lines.append("Recommendation: Keep 4096 as default.")

    report = "\n".join(lines)
    print(report)

    # Save results
    output = {
        "context_4096": result_4096,
        "context_8192": result_8192,
        "report": report,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Results saved to %s", args.output)

    return output


if __name__ == "__main__":
    main()
