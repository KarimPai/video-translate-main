"""Benchmark Gemma 4 prompt caching: cache_prompt ON vs OFF comparison.

Verifies that cache_prompt=True works correctly with Gemma 4 sliding-window
attention by comparing:
- Translation quality (cache ON vs OFF)
- Repeated identical requests (should benefit from cache)
- Changing dialogue history (cache should invalidate correctly)
- Long sequential translation sessions (no cache corruption)
- Prompt evaluation latency (cache hits should be faster)

Checks for:
- Incorrect context reuse
- Unexpected output changes
- Prompt-prefill latency spikes
- Repeated or contaminated translations

Usage:
    python -m scripts.benchmark_prompt_cache [--config config.toml] [--output results.json]
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

# Same utterances as context-size benchmark for comparability.
UTTERANCES = [
    # Ambiguous pronouns
    ("あのね、彼が言ったこと、覚えてる？", "Um, do you remember what he said?"),
    ("うん、それってすごいよね。私もそう思う。", "Yeah, that's amazing. I think so too."),
    ("ラップちゃんが言ったこと、私も同意するよ。", "I agree with what Lap-chan said."),

    # Multiple speakers
    ("みこち、大丈夫？怪我してない？", "Mikochi, are you okay? You not hurt, right?"),
    ("うん、大丈夫。でも車は壊れたね。", "Yeah, I'm fine. But the car's wrecked."),
    ("次はもっと慎重にやろう。このままじゃ捕まるよ。", "Let's be more careful next time. We'll get caught if we keep this up."),

    # VTuber names
    ("あのね、ラップ sama、今日の配信最高だったよ！", "Um, Lap-sama, today's stream was the best!"),
    ("ありがとう！みんなと遊べて楽しいから！", "Thanks! It's fun playing with everyone!"),
    ("ラップちゃん、その服カッコいい！どこで買ったの？", "Lap-chan, that outfit is cool! Where did you buy it?"),
    ("えへへ、ありがとう！オンラインで買ったの。", "Hehe, thanks! I bought it online."),

    # Game terminology
    ("みんな、スパチャありがとう！今日はマインクラフトで遊びます。", "Everyone, thanks for the super chats! We're playing Minecraft today."),
    ("えええ、マジで？それって超レアじゃない！売ってよ！", "Whaaat, seriously? That's super rare! Sell it to me!"),
    ("あのボス、本当に強かったね。もう一回挑戦しよう！", "That boss was really strong. Let's try again!"),

    # Emotional expressions
    ("うわ、警察来た！逃げよう！ラップ、こっち！", "Whoa, the police are here! Let's run! Lap, this way!"),
    ("ごめんごめん、冗談だったんだから。", "Sorry sorry, it was just a joke."),
    ("うわー、すごい！その技、どうやってやるの？", "Whoa, amazing! How do you do that move?"),

    # References to earlier dialogue
    ("それでは、また次の配信で会いましょう！バイバイ！", "Well then, see you in the next stream! Bye-bye!"),
    ("次回の配信は来週の日曜日だよ。楽しみにしてて！", "Next stream is next Sunday. Look forward to it!"),
    ("今日のテーマは、マインクラフトで脱出チャレンジ！", "Today's theme is a Minecraft escape challenge!"),

    # Additional context-dependent
    ("ちょっと待って、その音何？誰かいる？", "Wait a sec, what's that sound? Is someone there?"),
    ("いや、ただの風だよ。気にしないで。", "No, it's just the wind. Don't worry about it."),
    ("さて、次はどこに行こうか。地図見てみよう。", "Alright, where should we go next. Let's check the map."),
    ("あそこに行こう！山の上から景色が見えるよ。", "Let's go over there! You can see the view from the mountain top."),
    ("うん、いいね。でも暗くなる前に戻ろうよ。", "Yeah, sounds good. But let's come back before it gets dark."),
    ("ラップ、その発言ちょっと待って！そんなこと言っちゃダメ！", "Lap, hold on with that comment! You can't say stuff like that!"),
    ("みこち、今日の料理美味しかった！レシピ教えて！", "Mikochi, today's cooking was delicious! Teach me the recipe!"),
    ("はい、ぺこらです！よろしくお願いします。", "Yes, I'm Pekora! Nice to meet you."),
    ("えっ、もう終わったの？もっと遊びたかった！", "Huh, it's already over? I wanted to play more!"),
    ("ラップちゃん、その戦略ダメだよ。こっちの方がいい！", "Lap-chan, that strategy won't work. This one is better!"),
    ("みんなのコメント、全部読んでるから安心して！", "I'm reading all your comments, so don't worry!"),
    ("今日は特別ゲストが来るから、楽しみにしててね！", "A special guest is coming today, so look forward to it!"),
]


def percentile(values: list[float], p: float) -> float:
    """Calculate the p-th percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[idx]


def chrf_score(pairs: list[tuple[str, str]]) -> float | None:
    """Corpus chrF score on a 0-100 scale."""
    if not pairs:
        return None
    from collections import Counter

    def counts(ref: str, hyp: str):
        r, h = "".join(ref.split()), "".join(hyp.split())
        result = []
        for n in range(1, 7):
            rc = Counter(r[i:i + n] for i in range(len(r) - n + 1))
            hc = Counter(h[i:i + n] for i in range(len(h) - n + 1))
            result.append((sum((rc & hc).values()), sum(hc.values()), sum(rc.values())))
        return result

    totals = [[0, 0, 0] for _ in range(6)]
    for reference, hypothesis in pairs:
        for total, c in zip(totals, counts(reference, hypothesis)):
            for i, v in enumerate(c):
                total[i] += v

    orders = []
    for match, hyp, ref in totals:
        if ref:
            orders.append((match / hyp if hyp else 0.0, match / ref))
    if not orders:
        return 0.0
    prec = sum(p for p, _ in orders) / len(orders)
    rec = sum(r for _, r in orders) / len(orders)
    if not prec + rec:
        return 0.0
    beta2 = 4.0
    return 100 * (1 + beta2) * prec * rec / (beta2 * prec + rec)


def make_translator(cache_prompt: bool, config_path: Path = Path("config.toml")) -> Gemma4Translator:
    """Create a translator with the specified cache_prompt setting.
    
    Returns (translator, metrics) where metrics is a list that gets appended
    with per-request diagnostics dicts.
    """
    full_cfg: Config = load_config(config_path)
    cfg = replace(full_cfg.translate, reasoning_enabled=False, max_tokens=128)
    glossary = Glossary.load(Path(cfg.glossary)) if Path(cfg.glossary).exists() else Glossary()
    stream = StreamContext()
    translator = Gemma4Translator(cfg, glossary, stream=stream)

    # Shared list to collect per-request metrics
    metrics: list[dict] = []

    # Monkey-patch _chat to override cache_prompt and collect metrics
    def patched_chat(self, prompt: dict, *, timeout: float, temperature: float | None = None) -> str:
        if self.cfg.reasoning_enabled:
            max_tokens = self.cfg.max_tokens + self.cfg.reasoning_budget_tokens
        else:
            max_tokens = self.cfg.max_tokens
        from auto_translate.llama_server import check_running
        from auto_translate.runtime_diagnostics import increment_counter, record_event
        import time as _time

        body = {
            **prompt,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_prompt": cache_prompt,
        }
        check_running()
        if self._closed:
            return ""

        started = _time.monotonic()
        payload = self.server.chat(body, timeout=timeout)
        check_running()
        request_time = _time.monotonic() - started

        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""

        usage = payload.get("usage")
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        prompt_eval_time = 0.0
        generation_time = 0.0
        cache_hits = 0

        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
            prompt_eval_time = usage.get("prompt_eval_time", 0.0) or 0.0
            generation_time = usage.get("generation_time", 0.0) or 0.0
            cache_hits = usage.get("cache_hits", 0) or 0

        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            increment_counter("mt_truncated")
        elif finish_reason == "stop":
            increment_counter("mt_finish_stop")
        elif finish_reason:
            increment_counter("mt_finish_other")
        else:
            increment_counter("mt_finish_none")

        record_event(
            "mt_request",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
            request_time_s=round(request_time, 3),
            prompt_eval_time_s=round(prompt_eval_time, 4) if prompt_eval_time else None,
            generation_time_s=round(generation_time, 4) if generation_time else None,
            cache_hits=cache_hits,
            cache_enabled=cache_prompt,
            max_tokens=max_tokens,
            reasoning_enabled=self.cfg.reasoning_enabled,
            has_content=bool(content.strip()),
        )

        # Store metrics for the benchmark to read
        metrics.append({
            "prompt_eval_time_s": prompt_eval_time,
            "cache_hits": cache_hits,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        return content

    import types
    translator._chat = types.MethodType(patched_chat, translator)  # type: ignore
    return translator, metrics


def run_cache_test(
    samples: list[tuple[str, str]],
    cache_prompt: bool,
    label: str,
    config_path: Path = Path("config.toml"),
) -> dict:
    """Run translations with the given cache_prompt setting."""
    log.info("=== Prompt Cache Benchmark: %s (cache=%s) ===", label, cache_prompt)

    translator, metrics = make_translator(cache_prompt, config_path)

    history: list[tuple[str, str]] = []
    latencies: list[float] = []
    results: list[dict] = []
    missing = 0

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

            results.append({
                "index": i,
                "ja": ja,
                "en": english,
                "reference_en": ref_en,
                "latency_s": round(elapsed, 3),
                "error": None,
            })

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
                "reference_en": ref_en,
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    total_elapsed = time.monotonic() - total_start

    pairs = [(r["reference_en"], r["en"]) for r in results
             if r.get("reference_en") and r.get("en") and not r.get("error")]
    chrf = chrf_score(pairs) if pairs else None

    translator.close()

    # Extract metrics from the collected list
    prompt_eval_times = [m["prompt_eval_time_s"] for m in metrics]
    cache_hits_list = [m["cache_hits"] for m in metrics]

    return {
        "label": label,
        "cache_prompt": cache_prompt,
        "total_samples": len(samples),
        "successful": len(results) - missing,
        "missing_subtitles": missing,
        "total_time_s": round(total_elapsed, 3),
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
        "prompt_eval_avg_s": round(sum(prompt_eval_times) / len(prompt_eval_times), 4) if prompt_eval_times else 0.0,
        "total_cache_hits": sum(cache_hits_list),
        "chrf": round(chrf, 2) if chrf is not None else None,
        "results": results,
    }


def run_repeated_request_test(
    samples: list[tuple[str, str]],
    cache_prompt: bool,
    label: str,
    config_path: Path = Path("config.toml"),
) -> dict:
    """Test repeated identical requests — cache should speed up the second run."""
    log.info("=== Repeated Request Test: %s (cache=%s) ===", label, cache_prompt)

    translator, _metrics = make_translator(cache_prompt, config_path)

    # Pick a subset for repeated testing
    test_set = samples[:10]

    first_run: list[dict] = []
    second_run: list[dict] = []
    latencies_first: list[float] = []
    latencies_second: list[float] = []

    # First run — builds cache
    for i, (ja, ref_en) in enumerate(test_set):
        t0 = time.monotonic()
        prompt = translator.prompt_for(ja, [])
        english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
        elapsed = time.monotonic() - t0
        latencies_first.append(elapsed)
        first_run.append({"ja": ja, "en": english, "latency_s": round(elapsed, 3)})

    # Second run — should benefit from cache
    for i, (ja, ref_en) in enumerate(test_set):
        t0 = time.monotonic()
        prompt = translator.prompt_for(ja, [])
        english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
        elapsed = time.monotonic() - t0
        latencies_second.append(elapsed)
        second_run.append({"ja": ja, "en": english, "latency_s": round(elapsed, 3)})

    translator.close()

    # Check output consistency
    output_mismatches = 0
    for a, b in zip(first_run, second_run):
        if a["en"] != b["en"]:
            output_mismatches += 1
            log.info("Output mismatch: '%s' vs '%s'", a["en"], b["en"])

    avg_first = sum(latencies_first) / len(latencies_first) if latencies_first else 0
    avg_second = sum(latencies_second) / len(latencies_second) if latencies_second else 0

    speedup = 0.0
    if avg_second > 0 and avg_first > 0:
        speedup = (avg_first - avg_second) / avg_first * 100

    return {
        "label": label,
        "cache_prompt": cache_prompt,
        "samples_tested": len(test_set),
        "first_run_avg_s": round(avg_first, 3),
        "second_run_avg_s": round(avg_second, 3),
        "speedup_percent": round(speedup, 1),
        "output_mismatches": output_mismatches,
        "first_run": first_run,
        "second_run": second_run,
    }


def run_changing_history_test(
    samples: list[tuple[str, str]],
    cache_prompt: bool,
    label: str,
    config_path: Path = Path("config.toml"),
) -> dict:
    """Test that changing dialogue history invalidates cache correctly."""
    log.info("=== Changing History Test: %s (cache=%s) ===", label, cache_prompt)

    translator, _metrics = make_translator(cache_prompt, config_path)

    # Use the same utterance with different history contexts
    target_ja = "うん、それってすごいよね。私もそう思う。"

    history_a = [
        ("あのね、彼が言ったこと、覚えてる？", "Um, do you remember what he said?"),
        ("ラップちゃんが言ったこと、私も同意するよ。", "I agree with what Lap-chan said."),
    ]
    history_b = [
        ("みこち、大丈夫？怪我してない？", "Mikochi, are you okay?"),
        ("あのボス、本当に強かったね。もう一回挑戦しよう！", "That boss was really strong. Let's try again!"),
    ]

    prompt_a = translator.prompt_for(target_ja, history_a)
    en_a = clean_output(translator._chat(prompt_a, timeout=translator.cfg.timeout_s))

    prompt_b = translator.prompt_for(target_ja, history_b)
    en_b = clean_output(translator._chat(prompt_b, timeout=translator.cfg.timeout_s))

    # Also test with empty history
    prompt_empty = translator.prompt_for(target_ja, [])
    en_empty = clean_output(translator._chat(prompt_empty, timeout=translator.cfg.timeout_s))

    translator.close()

    # With temperature=0.0, same input+history should produce same output
    # Different history may or may not change output — just check no corruption
    results = {
        "target_ja": target_ja,
        "with_history_a": en_a,
        "with_history_b": en_b,
        "with_empty_history": en_empty,
        "all_non_empty": bool(en_a.strip() and en_b.strip() and en_empty.strip()),
        "no_special_tokens": all(
            "<unused" not in en for en in [en_a, en_b, en_empty]
        ),
    }

    return results


def run_long_session_test(
    samples: list[tuple[str, str]],
    cache_prompt: bool,
    label: str,
    config_path: Path = Path("config.toml"),
) -> dict:
    """Run a long sequential session to check for cache corruption over time."""
    log.info("=== Long Session Test: %s (cache=%s) ===", label, cache_prompt)

    translator, _metrics = make_translator(cache_prompt, config_path)

    history: list[tuple[str, str]] = []
    latencies: list[float] = []
    results: list[dict] = []
    missing = 0

    # Run all samples twice in sequence (simulating a long session)
    extended = samples + samples

    for i, (ja, ref_en) in enumerate(extended):
        t0 = time.monotonic()
        try:
            prompt = translator.prompt_for(ja, history)
            english = clean_output(translator._chat(prompt, timeout=translator.cfg.timeout_s))
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            if not english:
                missing += 1

            results.append({
                "index": i,
                "ja": ja,
                "en": english,
                "latency_s": round(elapsed, 3),
            })

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
                "latency_s": round(elapsed, 3),
                "error": str(exc),
            })

    translator.close()

    # Check first half vs second half for consistency
    half = len(samples)
    first_half = results[:half]
    second_half = results[half:]

    consistency_mismatches = 0
    for a, b in zip(first_half, second_half):
        if a["en"] != b["en"]:
            consistency_mismatches += 1

    # Check for degradation in the second half
    first_latencies = latencies[:half]
    second_latencies = latencies[half:]
    avg_first = sum(first_latencies) / len(first_latencies) if first_latencies else 0
    avg_second = sum(second_latencies) / len(second_latencies) if second_latencies else 0

    return {
        "label": label,
        "cache_prompt": cache_prompt,
        "total_translations": len(extended),
        "successful": len(extended) - missing,
        "missing_subtitles": missing,
        "consistency_mismatches": consistency_mismatches,
        "first_half_avg_s": round(avg_first, 3),
        "second_half_avg_s": round(avg_second, 3),
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 prompt caching")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config)
    samples = UTTERANCES

    lines = []
    all_results = {}

    # --- Test 1: Cache ON vs OFF (sequential, same history) ---
    lines.append("=" * 70)
    lines.append("TEST 1: Cache ON vs OFF — Sequential Translation")
    lines.append("=" * 70)

    result_on = run_cache_test(samples, cache_prompt=True, label="cache_on", config_path=config_path)
    result_off = run_cache_test(samples, cache_prompt=False, label="cache_off", config_path=config_path)
    all_results["cache_on"] = result_on
    all_results["cache_off"] = result_off

    metrics = [
        ("Total samples", result_on["total_samples"], result_off["total_samples"]),
        ("Successful", result_on["successful"], result_off["successful"]),
        ("Missing subtitles", result_on["missing_subtitles"], result_off["missing_subtitles"]),
        ("Total time (s)", result_on["total_time_s"], result_off["total_time_s"]),
        ("Latency avg (s)", result_on["latency_avg_s"], result_off["latency_avg_s"]),
        ("Latency p50 (s)", result_on["latency_p50_s"], result_off["latency_p50_s"]),
        ("Latency p95 (s)", result_on["latency_p95_s"], result_off["latency_p95_s"]),
        ("Latency max (s)", result_on["latency_max_s"], result_off["latency_max_s"]),
        ("chrF score", result_on["chrf"] or 0, result_off["chrf"] or 0),
        ("Cache hits", result_on["total_cache_hits"], result_off["total_cache_hits"]),
    ]

    lines.append("")
    lines.append(f"{'Metric':<30} {'Cache ON':>12} {'Cache OFF':>12} {'Change':>10}")
    lines.append("-" * 64)

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
            lines.append(f"{name:<30} {v_on:>12.3f} {v_off:>12.3f} {change_str:>10}")
        else:
            lines.append(f"{name:<30} {v_on:>12} {v_off:>12} {change_str:>10}")

    # Check output identity between cache ON and OFF
    output_matches = 0
    for r_on, r_off in zip(result_on["results"], result_off["results"]):
        if r_on["en"] == r_off["en"]:
            output_matches += 1
    lines.append("")
    lines.append(f"Identical output (ON vs OFF): {output_matches}/{len(samples)}")

    # --- Test 2: Repeated identical requests ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("TEST 2: Repeated Identical Requests")
    lines.append("=" * 70)

    repeated_on = run_repeated_request_test(samples, cache_prompt=True, label="repeated_on", config_path=config_path)
    repeated_off = run_repeated_request_test(samples, cache_prompt=False, label="repeated_off", config_path=config_path)
    all_results["repeated_on"] = repeated_on
    all_results["repeated_off"] = repeated_off

    lines.append("")
    lines.append(f"{'Metric':<30} {'Cache ON':>12} {'Cache OFF':>12}")
    lines.append("-" * 54)
    lines.append(f"{'First run avg (s)':<30} {repeated_on['first_run_avg_s']:>12.3f} {repeated_off['first_run_avg_s']:>12.3f}")
    lines.append(f"{'Second run avg (s)':<30} {repeated_on['second_run_avg_s']:>12.3f} {repeated_off['second_run_avg_s']:>12.3f}")
    lines.append(f"{'Speedup %%':<30} {repeated_on['speedup_percent']:>11.1f}% {repeated_off['speedup_percent']:>11.1f}%")
    lines.append(f"{'Output mismatches':<30} {repeated_on['output_mismatches']:>12} {repeated_off['output_mismatches']:>12}")

    # --- Test 3: Changing dialogue history ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("TEST 3: Changing Dialogue History")
    lines.append("=" * 70)

    history_on = run_changing_history_test(samples, cache_prompt=True, label="history_on", config_path=config_path)
    history_off = run_changing_history_test(samples, cache_prompt=False, label="history_off", config_path=config_path)
    all_results["history_on"] = history_on
    all_results["history_off"] = history_off

    lines.append("")
    lines.append(f"Target utterance: {history_on['target_ja']}")
    lines.append("")
    lines.append(f"{'History':<20} {'Cache ON output':<35} {'Cache OFF output':<35}")
    lines.append("-" * 90)
    lines.append(f"{'History A':<20} {history_on['with_history_a'][:33]:<35} {history_off['with_history_a'][:33]:<35}")
    lines.append(f"{'History B':<20} {history_on['with_history_b'][:33]:<35} {history_off['with_history_b'][:33]:<35}")
    lines.append(f"{'Empty':<20} {history_on['with_empty_history'][:35]:<35} {history_off['with_empty_history'][:35]:<35}")
    lines.append("")
    lines.append(f"All outputs non-empty: ON={history_on['all_non_empty']}, OFF={history_off['all_non_empty']}")
    lines.append(f"No special tokens: ON={history_on['no_special_tokens']}, OFF={history_off['no_special_tokens']}")

    # --- Test 4: Long session stability ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("TEST 4: Long Session Stability (2x all samples)")
    lines.append("=" * 70)

    long_on = run_long_session_test(samples, cache_prompt=True, label="long_on", config_path=config_path)
    long_off = run_long_session_test(samples, cache_prompt=False, label="long_off", config_path=config_path)
    all_results["long_on"] = long_on
    all_results["long_off"] = long_off

    lines.append("")
    lines.append(f"{'Metric':<30} {'Cache ON':>12} {'Cache OFF':>12}")
    lines.append("-" * 54)
    lines.append(f"{'Total translations':<30} {long_on['total_translations']:>12} {long_off['total_translations']:>12}")
    lines.append(f"{'Successful':<30} {long_on['successful']:>12} {long_off['successful']:>12}")
    lines.append(f"{'Missing subtitles':<30} {long_on['missing_subtitles']:>12} {long_off['missing_subtitles']:>12}")
    lines.append(f"{'Consistency mismatches':<30} {long_on['consistency_mismatches']:>12} {long_off['consistency_mismatches']:>12}")
    lines.append(f"{'Latency avg (s)':<30} {long_on['latency_avg_s']:>12.3f} {long_off['latency_avg_s']:>12.3f}")
    lines.append(f"{'Latency p95 (s)':<30} {long_on['latency_p95_s']:>12.3f} {long_off['latency_p95_s']:>12.3f}")
    lines.append(f"{'First half avg (s)':<30} {long_on['first_half_avg_s']:>12.3f} {long_off['first_half_avg_s']:>12.3f}")
    lines.append(f"{'Second half avg (s)':<30} {long_on['second_half_avg_s']:>12.3f} {long_off['second_half_avg_s']:>12.3f}")

    # --- Conclusion ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)

    # Check if cache ON produces identical output to OFF
    if output_matches == len(samples):
        lines.append("Cache ON produces identical output to Cache OFF for all utterances.")
    else:
        lines.append(f"WARNING: {len(samples) - output_matches} utterances differ between cache ON and OFF.")

    # Check repeated request speedup
    if repeated_on["speedup_percent"] > 5:
        lines.append(f"Cache ON shows {repeated_on['speedup_percent']:.1f}% speedup on repeated requests — cache is working.")
    elif repeated_on["speedup_percent"] > 0:
        lines.append(f"Cache ON shows minor {repeated_on['speedup_percent']:.1f}% speedup on repeated requests.")
    else:
        lines.append("Cache ON shows no measurable speedup on repeated requests.")

    # Check consistency
    if repeated_on["output_mismatches"] == 0 and repeated_off["output_mismatches"] == 0:
        lines.append("Repeated requests produce consistent output in both modes.")
    else:
        lines.append(f"WARNING: Output mismatches in repeated requests (ON={repeated_on['output_mismatches']}, OFF={repeated_off['output_mismatches']}).")

    # Check long session
    if long_on["consistency_mismatches"] == 0:
        lines.append("Long session with cache ON: no consistency issues.")
    else:
        lines.append(f"WARNING: Long session cache ON has {long_on['consistency_mismatches']} consistency mismatches.")

    if long_on["missing_subtitles"] == 0 and long_off["missing_subtitles"] == 0:
        lines.append("No missing subtitles in long sessions.")

    # Final recommendation
    cache_safe = (
        output_matches == len(samples)
        and repeated_on["output_mismatches"] == 0
        and long_on["consistency_mismatches"] <= 2  # allow minor stochasticity
        and long_on["missing_subtitles"] == 0
    )

    if cache_safe:
        lines.append("")
        lines.append("Recommendation: Keep cache_prompt=True — it works correctly with Gemma 4.")
    else:
        lines.append("")
        lines.append("Recommendation: Investigate cache issues before enabling in production.")

    report = "\n".join(lines)
    print(report)

    # Save results
    output_data = {
        "sequential_comparison": {
            "cache_on": result_on,
            "cache_off": result_off,
        },
        "repeated_requests": {
            "cache_on": repeated_on,
            "cache_off": repeated_off,
        },
        "changing_history": {
            "cache_on": history_on,
            "cache_off": history_off,
        },
        "long_session": {
            "cache_on": long_on,
            "cache_off": long_off,
        },
        "report": report,
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Results saved to %s", args.output)

    return output_data


if __name__ == "__main__":
    main()
