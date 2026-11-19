"""Benchmark Gemma 4 reasoning mode: ON vs OFF comparison.

Compares translation quality and latency with reasoning (thinking) mode
enabled versus disabled, using the same set of Japanese utterances.

Metrics compared:
- Translation accuracy (chrF score against reference English)
- Missing subtitles (empty output count)
- Average latency per translation
- P95 latency
- Truncated outputs (finish_reason=length)
- Reasoning-token usage (when reasoning is ON)

Usage:
    python -m scripts.benchmark_reasoning [manifest.json]

If no manifest is provided, uses 30 built-in Japanese utterances.
"""
import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from auto_translate.config import TranslateConfig, load_config
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.translate.gemma4 import Gemma4Translator, clean_output

log = logging.getLogger(__name__)

# 30 Japanese utterances for benchmarking, covering various dialogue patterns.
BENCHMARK_UTTERANCES = [
    ("みんな、スパチャありがとう！今日はマインクラフトで遊びます。",
     "Everyone, thanks for the super chats! We're playing Minecraft today."),
    ("ラップちゃん、ちょっと待って！そのアイテム僕が使いたかったの！",
     "Lap-chan, wait a minute! I wanted to use that item!"),
    ("えええ、マジで？それって超レアじゃない！売ってよ！",
     "Whaaat, seriously? That's super rare! Sell it to me!"),
    ("いやいや、今はダメだよ。あとで考えよう。",
     "No no, not now. Let's think about it later."),
    ("うわ、警察来た！逃げよう！ラップ、こっち！",
     "Whoa, the police are here! Let's run! Lap, this way!"),
    ("みこち、大丈夫？怪我してない？",
     "Mikochi, are you okay? You not hurt, right?"),
    ("うん、大丈夫。でも車は壊れたね。",
     "Yeah, I'm fine. But the car's wrecked."),
    ("次はもっと慎重にやろう。このままじゃ捕まるよ。",
     "Let's be more careful next time. We'll get caught if we keep this up."),
    ("あのね、ラップ sama、今日の配信最高だったよ！",
     "Um, Lap-sama, today's stream was the best!"),
    ("ありがとう！みんなと遊べて楽しいから！",
     "Thanks! It's fun playing with everyone!"),
    ("ちょっと待って、その音何？誰かいる？",
     "Wait a sec, what's that sound? Is someone there?"),
    ("いや、ただの風だよ。気にしないで。",
     "No, it's just the wind. Don't worry about it."),
    ("ラップちゃん、その服カッコいい！どこで買ったの？",
     "Lap-chan, that outfit is cool! Where did you buy it?"),
    ("えへへ、ありがとう！オンラインで買ったの。",
     "Hehe, thanks! I bought it online."),
    ("さて、次はどこに行こうか。地図見てみよう。",
     "Alright, where should we go next. Let's check the map."),
    ("あそこに行こう！山の上から景色が見えるよ。",
     "Let's go over there! You can see the view from the mountain top."),
    ("うん、いいね。でも暗くなる前に戻ろうよ。",
     "Yeah, sounds good. But let's come back before it gets dark."),
    ("ラップ、その発言ちょっと待って！そんなこと言っちゃダメ！",
     "Lap, hold on with that comment! You can't say stuff like that!"),
    ("ごめんごめん、冗談だったんだから。",
     "Sorry sorry, it was just a joke."),
    ("みこち、今日の料理美味しかった！レシピ教えて！",
     "Mikochi, today's cooking was delicious! Teach me the recipe!"),
    ("はい、ぺこらです！よろしくお願いします。",
     "Yes, I'm Pekora! Nice to meet you."),
    ("今日のテーマは、マインクラフトで脱出チャレンジ！",
     "Today's theme is a Minecraft escape challenge!"),
    ("えっ、もう終わったの？もっと遊びたかった！",
     "Huh, it's already over? I wanted to play more!"),
    ("次回の配信は来週の日曜日だよ。楽しみにしてて！",
     "Next stream is next Sunday. Look forward to it!"),
    ("あのボス、本当に強かったね。もう一回挑戦しよう！",
     "That boss was really strong. Let's try again!"),
    ("ラップちゃん、その戦略ダメだよ。こっちの方がいい！",
     "Lap-chan, that strategy won't work. This one is better!"),
    ("うわー、すごい！その技、どうやってやるの？",
     "Whoa, amazing! How do you do that move?"),
    ("みんなのコメント、全部読んでるから安心して！",
     "I'm reading all your comments, so don't worry!"),
    ("今日は特別ゲストが来るから、楽しみにしててね！",
     "A special guest is coming today, so look forward to it!"),
    ("それでは、また次の配信で会いましょう！バイバイ！",
     "Well then, see you in the next stream! Bye-bye!"),
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


def load_utterances(manifest_path: Path | None) -> list[tuple[str, str | None]]:
    """Load utterances from manifest or use built-in samples."""
    if manifest_path and manifest_path.exists():
        text = manifest_path.read_text(encoding="utf-8-sig")
        cases = json.loads(text)
        if isinstance(cases, list) and cases:
            samples = []
            for case in cases:
                ja = case.get("ja", "").strip()
                if ja:
                    en = case.get("en", "").strip() or None
                    samples.append((ja, en))
            if samples:
                return samples
    # Use built-in 30 utterances
    return list(BENCHMARK_UTTERANCES)


def run_mode(
    samples: list[tuple[str, str | None]],
    translator: Gemma4Translator,
    label: str,
) -> dict:
    """Run translations with the given translator configuration."""
    log.info("=== Benchmark: %s (reasoning=%s) ===", label, translator.cfg.reasoning_enabled)

    history: list[tuple[str, str]] = []
    latencies: list[float] = []
    results: list[dict] = []
    truncated = 0
    missing = 0
    reasoning_tokens_total = 0

    total_start = time.monotonic()

    for i, (ja, ref_en) in enumerate(samples):
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
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    total_elapsed = time.monotonic() - total_start

    # Compute chrF for samples with reference English
    pairs = [(r["reference_en"], r["en"]) for r in results
             if r.get("reference_en") and r.get("en") and not r.get("error")]
    chrf = chrf_score(pairs) if pairs else None

    # Get reasoning token counters from diagnostics
    from auto_translate.runtime_diagnostics import _ACTIVE
    counters = {}
    if _ACTIVE is not None:
        snapshot = _ACTIVE.snapshot()
        counters = snapshot.get("counters", {})

    reasoning_tokens_total = counters.get("mt_reasoning_tokens", 0) + counters.get("mt_reasoning_tokens_api", 0)
    truncated = counters.get("mt_truncated", 0)

    summary = {
        "label": label,
        "reasoning_enabled": translator.cfg.reasoning_enabled,
        "total_samples": len(samples),
        "successful": len(results) - missing,
        "missing_subtitles": missing,
        "truncated_outputs": truncated,
        "total_time_s": round(total_elapsed, 3),
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
        "chrf": round(chrf, 2) if chrf is not None else None,
        "reasoning_tokens_total": reasoning_tokens_total if translator.cfg.reasoning_enabled else 0,
        "reasoning_avg_tokens": (reasoning_tokens_total / len(samples)) if translator.cfg.reasoning_enabled and samples else 0,
        "max_tokens_used": (translator.cfg.max_tokens + translator.cfg.reasoning_budget_tokens) if translator.cfg.reasoning_enabled else translator.cfg.max_tokens,
        "results": results,
    }

    return summary


def create_translator_for_mode(
    base_cfg: TranslateConfig,
    reasoning_enabled: bool,
) -> Gemma4Translator:
    """Create a translator with the specified reasoning mode."""
    cfg = replace(
        base_cfg,
        reasoning_enabled=reasoning_enabled,
        # When reasoning is ON, use max_tokens=384 as specified
        max_tokens=384 if reasoning_enabled else base_cfg.max_tokens,
    )
    glossary = Glossary.load(Path(cfg.glossary)) if Path(cfg.glossary).exists() else Glossary()
    stream = StreamContext()
    return Gemma4Translator(cfg, glossary, stream=stream)


def compare_results(on_result: dict, off_result: dict) -> str:
    """Generate a comparison report between reasoning ON and OFF."""
    lines = [
        "=" * 70,
        "GEMMA 4 REASONING BENCHMARK: ON vs OFF",
        "=" * 70,
        "",
        f"{'Metric':<40} {'Reasoning ON':>14} {'Reasoning OFF':>14} {'Change':>12}",
        "-" * 70,
    ]

    metrics = [
        ("Total samples", on_result["total_samples"], off_result["total_samples"]),
        ("Successful translations", on_result["successful"], off_result["successful"]),
        ("Missing subtitles", on_result["missing_subtitles"], off_result["missing_subtitles"]),
        ("Truncated outputs", on_result["truncated_outputs"], off_result["truncated_outputs"]),
        ("Total time (s)", on_result["total_time_s"], off_result["total_time_s"]),
        ("Latency avg (s)", on_result["latency_avg_s"], off_result["latency_avg_s"]),
        ("Latency p50 (s)", on_result["latency_p50_s"], off_result["latency_p50_s"]),
        ("Latency p95 (s)", on_result["latency_p95_s"], off_result["latency_p95_s"]),
        ("Latency max (s)", on_result["latency_max_s"], off_result["latency_max_s"]),
        ("chrF score", on_result["chrf"] or 0, off_result["chrf"] or 0),
        ("Reasoning tokens (total)", on_result["reasoning_tokens_total"], 0),
        ("Reasoning tokens (avg)", on_result["reasoning_avg_tokens"], 0),
        ("max_tokens", on_result["max_tokens_used"], off_result["max_tokens_used"]),
    ]

    for name, v_on, v_off in metrics:
        if isinstance(v_on, float) and isinstance(v_off, float) and v_off != 0:
            change = ((v_on - v_off) / abs(v_off)) * 100
            change_str = f"{change:+.1f}%"
        elif isinstance(v_on, int) and isinstance(v_off, int) and v_off != 0:
            change = ((v_on - v_off) / abs(v_off)) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = ""

        if isinstance(v_on, float):
            lines.append(f"{name:<40} {v_on:>14.3f} {v_off:>14.3f} {change_str:>12}")
        else:
            lines.append(f"{name:<40} {v_on:>14} {v_off:>14} {change_str:>12}")

    lines.append("-" * 70)
    lines.append("")

    # Verdict
    lat_change = 0
    if off_result["latency_avg_s"] > 0:
        lat_change = (on_result["latency_avg_s"] - off_result["latency_avg_s"]) / off_result["latency_avg_s"] * 100

    chrf_on = on_result["chrf"] or 0
    chrf_off = off_result["chrf"] or 0
    chrf_diff = chrf_on - chrf_off

    lines.append("VERDICT:")
    lines.append(f"  Latency change: {lat_change:+.1f}%")
    lines.append(f"  chrF change: {chrf_diff:+.2f}")
    lines.append(f"  Missing subtitles: ON={on_result['missing_subtitles']}, OFF={off_result['missing_subtitles']}")
    lines.append(f"  Reasoning tokens used: {on_result['reasoning_tokens_total']} total")

    if chrf_diff > 0 and lat_change < 50:
        lines.append("  -> Reasoning mode improves accuracy with acceptable latency overhead.")
    elif chrf_diff < -2:
        lines.append("  -> Reasoning mode degrades accuracy. Consider keeping it disabled.")
    elif lat_change > 100:
        lines.append("  -> Reasoning mode doubles latency. Weigh accuracy gains against speed.")
    else:
        lines.append("  -> Mixed results. Evaluate based on your priority (accuracy vs speed).")

    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 reasoning mode: ON vs OFF")
    parser.add_argument("manifest", nargs="?", default=None, type=Path,
                        help="Path to benchmark manifest JSON (optional, uses 30 built-in utterances)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.toml")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load utterances
    samples = load_utterances(args.manifest)
    log.info("Loaded %d utterances for benchmarking", len(samples))

    # Load config
    cfg_path = args.config or Path("config.toml")
    cfg = load_config(cfg_path)

    results: list[dict] = []

    # Run reasoning OFF first (baseline)
    try:
        translator_off = create_translator_for_mode(cfg.translate, reasoning_enabled=False)
        try:
            result_off = run_mode(samples, translator_off, "reasoning=OFF")
            results.append(result_off)
            log.info("Reasoning OFF: %.1f avg latency, %d missing, chrF=%.1f",
                      result_off["latency_avg_s"], result_off["missing_subtitles"],
                      result_off["chrf"] or 0)
        finally:
            translator_off.close()
    except Exception as exc:
        log.error("Reasoning OFF benchmark failed: %s", exc)
        result_off = {"label": "reasoning=OFF", "error": str(exc)}

    # Run reasoning ON
    try:
        translator_on = create_translator_for_mode(cfg.translate, reasoning_enabled=True)
        try:
            result_on = run_mode(samples, translator_on, "reasoning=ON")
            results.append(result_on)
            log.info("Reasoning ON: %.1f avg latency, %d missing, chrF=%.1f, reasoning_tokens=%d",
                      result_on["latency_avg_s"], result_on["missing_subtitles"],
                      result_on["chrf"] or 0, result_on["reasoning_tokens_total"])
        finally:
            translator_on.close()
    except Exception as exc:
        log.error("Reasoning ON benchmark failed: %s", exc)
        result_on = {"label": "reasoning=ON", "error": str(exc)}

    # Compare results
    if isinstance(result_on, dict) and isinstance(result_off, dict) and "error" not in result_on and "error" not in result_off:
        comparison = compare_results(result_on, result_off)
        print("\n" + comparison)

    # Save results
    output_path = args.output or Path("benchmark_reasoning_results.json")
    # Strip per-result details for compact output
    compact_results = []
    for r in results:
        compact = {k: v for k, v in r.items() if k != "results"}
        compact_results.append(compact)
    output_path.write_text(
        json.dumps(compact_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
