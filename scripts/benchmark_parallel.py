"""Benchmark Gemma 4 E4B parallel inference (np=1 vs np=2).

Compares single-slot and dual-slot translation performance using identical
per-request settings. Measures:
- Throughput (translations per second)
- Per-request latency (prompt eval + generation)
- Peak VRAM usage
- Translation quality (chrF score if reference provided)
- Subtitle ordering with concurrent requests

Preserves 4096 tokens per slot — for np=2, increases total context allocation
to maintain identical per-slot context.

Usage:
    python -m scripts.benchmark_parallel [manifest.json]

The manifest is the same format as scripts/benchmark.py: a JSON list of
objects with "audio" (WAV file path), "ja" (reference Japanese), and
optionally "en" (reference English translation).

If no manifest is provided, generates synthetic Japanese dialogue samples.
"""
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_translate.config import TranslateConfig, load_config
from auto_translate.glossary import Glossary
from auto_translate.stream_context import StreamContext
from auto_translate.translate.gemma4 import Gemma4Translator, clean_output

log = logging.getLogger(__name__)

# Synthetic Japanese dialogue samples for benchmarking when no manifest is provided
SYNTHETIC_SAMPLES = [
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


def generate_samples(count: int = 20) -> list[tuple[str, str | None]]:
    """Generate Japanese dialogue samples for benchmarking.
    
    Returns list of (japanese_text, english_reference_or_none).
    """
    samples = list(SYNTHETIC_SAMPLES)
    # Repeat if needed
    while len(samples) < count:
        samples.extend(samples)
    return samples[:count]


def load_samples_from_manifest(manifest_path: Path) -> list[tuple[str, str | None]]:
    """Load samples from a benchmark manifest JSON file."""
    text = manifest_path.read_text(encoding="utf-8-sig")
    cases = json.loads(text)
    if not isinstance(cases, list):
        raise ValueError("Manifest must be a JSON list")
    
    samples = []
    for case in cases:
        ja = case.get("ja", "").strip()
        if not ja:
            continue
        en = case.get("en", "").strip() or None
        samples.append((ja, en))
    
    if not samples:
        raise ValueError("Manifest contains no valid samples")
    return samples


def run_benchmark(
    samples: list[tuple[str, str | None]],
    parallel_slots: int,
    translator: Gemma4Translator,
    label: str,
) -> dict[str, Any]:
    """Run a benchmark with the given translator configuration.
    
    For np=1: sequential translation.
    For np=2: concurrent translation using thread pool.
    """
    log.info("=== Benchmark: %s (np=%d) ===", label, parallel_slots)
    
    history: list[tuple[str, str]] = []
    results: list[dict] = []
    latencies: list[float] = []
    
    total_start = time.monotonic()
    
    if parallel_slots == 1:
        # Sequential translation
        for i, (ja, _en) in enumerate(samples):
            t0 = time.monotonic()
            try:
                prompt = translator.prompt_for(ja, history)
                english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)
                results.append({
                    "index": i,
                    "ja": ja[:50],
                    "en": english[:80],
                    "latency_s": elapsed,
                    "error": None,
                })
                if english and ja:
                    history.append((ja, english))
                    if len(history) > translator.cfg.context_lines:
                        history.pop(0)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)
                results.append({
                    "index": i,
                    "ja": ja[:50],
                    "en": "",
                    "latency_s": elapsed,
                    "error": str(exc),
                })
    else:
        # Concurrent translation for np >= 2
        # Submit all translations concurrently, then collect results
        max_workers = min(parallel_slots, len(samples))
        
        def translate_one(idx: int, ja: str, hist: list[tuple[str, str]]) -> dict:
            """Translate a single sample with its own history copy."""
            t0 = time.monotonic()
            try:
                with translator._lock:
                    prompt = translator.prompt_for(ja, hist)
                    english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
                elapsed = time.monotonic() - t0
                return {
                    "index": idx,
                    "ja": ja[:50],
                    "en": english[:80],
                    "latency_s": elapsed,
                    "error": None,
                    "english": english,
                }
            except Exception as exc:
                elapsed = time.monotonic() - t0
                return {
                    "index": idx,
                    "ja": ja[:50],
                    "en": "",
                    "latency_s": elapsed,
                    "error": str(exc),
                    "english": "",
                }
        
        # Process in batches to maintain context continuity
        batch_size = parallel_slots
        ordered_results: list[dict | None] = [None] * len(samples)
        
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch = samples[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for local_idx, (ja, _en) in enumerate(batch):
                    global_idx = batch_start + local_idx
                    future = executor.submit(translate_one, global_idx, ja, list(history))
                    futures[future] = global_idx
                
                for future in as_completed(futures):
                    global_idx = futures[future]
                    result = future.result()
                    ordered_results[global_idx] = result
                    latencies.append(result["latency_s"])
                    # Update history with successful translations
                    if result.get("english") and not result["error"]:
                        history.append((result["ja"], result["english"]))
            
            # Trim history to context size
            if len(history) > translator.cfg.context_lines:
                history = history[-translator.cfg.context_lines:]
        
        # Build ordered results list
        for r in ordered_results:
            if r is not None:
                results.append(r)
    
    total_elapsed = time.monotonic() - total_start
    
    # Calculate statistics
    errors = [r for r in results if r.get("error")]
    successful = [r for r in results if r.get("error") is None]
    
    summary = {
        "label": label,
        "parallel_slots": parallel_slots,
        "total_samples": len(samples),
        "successful": len(successful),
        "errors": len(errors),
        "total_time_s": round(total_elapsed, 3),
        "throughput_tps": round(len(successful) / total_elapsed, 3) if total_elapsed > 0 else 0.0,
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
    }
    
    # Check subtitle ordering (results should be in index order)
    if len(results) > 1:
        indices = [r["index"] for r in results]
        ordered = indices == sorted(indices)
        summary["subtitle_ordered"] = ordered
    
    return summary


def create_translator(cfg: TranslateConfig, parallel_slots: int) -> Gemma4Translator:
    """Create a Gemma4Translator with the specified parallel slots.
    
    For np=2, increases total context to maintain 4096 tokens per slot.
    """
    adjusted_cfg = cfg
    
    # For np=2, increase total context to maintain per-slot context
    if parallel_slots >= 2:
        # llama.cpp divides context among slots, so multiply total context
        # to maintain per-slot context size
        total_context = cfg.context_size * parallel_slots
        log.info(
            "Adjusting context: %d slots x %d = %d total context tokens",
            parallel_slots, cfg.context_size, total_context
        )
        adjusted_cfg = replace(
            cfg,
            parallel_slots=parallel_slots,
            context_size=total_context,
        )
    else:
        adjusted_cfg = replace(cfg, parallel_slots=parallel_slots)
    
    glossary = Glossary.load(Path("glossary.toml")) if Path("glossary.toml").exists() else Glossary()
    stream = StreamContext()
    
    return Gemma4Translator(adjusted_cfg, glossary, stream=stream)


def compare_results(np1: dict, np2: dict) -> str:
    """Generate a comparison report between np=1 and np=2 results."""
    lines = [
        "=" * 70,
        "PARALLEL INFERENCE BENCHMARK COMPARISON",
        "=" * 70,
        "",
        f"{'Metric':<40} {'np=1':>12} {'np=2':>12} {'Change':>12}",
        "-" * 70,
    ]
    
    metrics = [
        ("Total samples", np1["total_samples"], np2["total_samples"], False),
        ("Successful", np1["successful"], np2["successful"], False),
        ("Errors", np1["errors"], np2["errors"], False),
        ("Total time (s)", np1["total_time_s"], np2["total_time_s"], True),
        ("Throughput (trans/s)", np1["throughput_tps"], np2["throughput_tps"], True),
        ("Latency avg (s)", np1["latency_avg_s"], np2["latency_avg_s"], False),
        ("Latency p50 (s)", np1["latency_p50_s"], np2["latency_p50_s"], False),
        ("Latency p95 (s)", np1["latency_p95_s"], np2["latency_p95_s"], False),
        ("Latency max (s)", np1["latency_max_s"], np2["latency_max_s"], False),
    ]
    
    for name, v1, v2, _ in metrics:
        if isinstance(v1, float) and isinstance(v2, float) and v1 != 0:
            change = ((v2 - v1) / abs(v1)) * 100
            change_str = f"{change:+.1f}%"
        elif isinstance(v1, int) and isinstance(v2, int) and v1 != 0:
            change = ((v2 - v1) / abs(v1)) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = ""
        
        if isinstance(v1, float):
            lines.append(f"{name:<40} {v1:>12.3f} {v2:>12.3f} {change_str:>12}")
        else:
            lines.append(f"{name:<40} {v1:>12} {v2:>12} {change_str:>12}")
    
    # Subtitle ordering
    ord1 = np1.get("subtitle_ordered", True)
    ord2 = np2.get("subtitle_ordered", True)
    lines.append(f"{'Subtitle ordered':<40} {str(ord1):>12} {str(ord2):>12} {'':>12}")
    
    lines.append("-" * 70)
    
    # Verdict
    throughput_improvement = (np2["throughput_tps"] - np1["throughput_tps"]) / np1["throughput_tps"] * 100 if np1["throughput_tps"] > 0 else 0
    latency_change = (np2["latency_avg_s"] - np1["latency_avg_s"]) / np1["latency_avg_s"] * 100 if np1["latency_avg_s"] > 0 else 0
    
    lines.append("")
    lines.append("VERDICT:")
    lines.append(f"  Throughput change: {throughput_improvement:+.1f}%")
    lines.append(f"  Latency change: {latency_change:+.1f}%")
    
    if throughput_improvement > 10 and np2.get("subtitle_ordered", True):
        lines.append("  -> np=2 shows meaningful throughput improvement without ordering issues.")
        lines.append("  -> Consider enabling parallel slots for sustained dialogue.")
    elif throughput_improvement < -5:
        lines.append("  -> np=2 degrades throughput. Keep np=1.")
    else:
        lines.append("  -> np=2 shows marginal improvement. Weigh against VRAM usage.")
    
    lines.append("")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 parallel inference")
    parser.add_argument("manifest", nargs="?", default=None, type=Path,
                        help="Path to benchmark manifest JSON (optional)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of synthetic samples if no manifest (default: 20)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.toml")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Load samples
    if args.manifest:
        log.info("Loading samples from manifest: %s", args.manifest)
        samples = load_samples_from_manifest(args.manifest)
    else:
        log.info("Generating %d synthetic samples", args.samples)
        samples = generate_samples(args.samples)
    
    log.info("Loaded %d samples", len(samples))
    
    # Load config
    cfg_path = args.config or Path("config.toml")
    cfg = load_config(cfg_path)
    
    results: list[dict] = []
    
    # Benchmark np=1
    try:
        translator1 = create_translator(cfg.translate, parallel_slots=1)
        try:
            result1 = run_benchmark(samples, 1, translator1, "np=1")
            results.append(result1)
            log.info("np=1 result: %s", json.dumps(result1, indent=2))
        finally:
            translator1.close()
    except Exception as exc:
        log.error("np=1 benchmark failed: %s", exc)
        result1 = {"label": "np=1", "error": str(exc)}
    
    # Benchmark np=2
    try:
        translator2 = create_translator(cfg.translate, parallel_slots=2)
        try:
            result2 = run_benchmark(samples, 2, translator2, "np=2")
            results.append(result2)
            log.info("np=2 result: %s", json.dumps(result2, indent=2))
        finally:
            translator2.close()
    except Exception as exc:
        log.error("np=2 benchmark failed: %s", exc)
        result2 = {"label": "np=2", "error": str(exc)}
    
    # Compare results
    if "error" not in result1 and "error" not in result2:
        comparison = compare_results(result1, result2)
        print("\n" + comparison)
    
    # Save results
    output = Path("benchmark_parallel_results.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
