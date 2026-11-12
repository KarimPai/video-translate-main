"""Run three production pipelines against identical, manually annotated local clips."""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_quality import evaluate


def run_clip(profile, clip, cfg, output, env):
    import psutil

    command = [sys.executable, "-m", "auto_translate", str(clip), "--config", str(cfg),
               "--mode", "vod", "--fast", "--no-web", "--verbose"]
    cpu_times = {}
    ram_peak = 0
    start = time.monotonic()
    with output.with_suffix(".txt").open("w", encoding="utf8") as stdout, output.with_suffix(
            ".log").open("w", encoding="utf8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        parent = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                ram = 0
                try:
                    children = parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    break
                for child in [parent, *children]:
                    try:
                        ticks = child.cpu_times()
                        cpu_times[(child.pid, child.create_time())] = ticks.user + ticks.system
                        ram += child.memory_info().rss
                    except psutil.Error:
                        continue
                ram_peak = max(ram_peak, ram)
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.wait():
        raise RuntimeError(f"{profile} failed for {clip}; see {output}.log")
    log = output.with_suffix(".log").read_text(encoding="utf8")
    if profile != "A" and ("Secondary qwen3-asr" not in log or "Releasing optional" in log
                           or "Secondary ASR failed" in log):
        raise RuntimeError(f"{profile} degraded to baseline; see {output}.log")
    if profile == "C" and ("transcript selector ready" not in log or "Selector fallback" in log):
        raise RuntimeError(f"Profile C selector unavailable or failed; see {output}.log")
    lines = output.with_suffix(".txt").read_text(encoding="utf8").splitlines()
    gb = [float(v) for v in re.findall(r"total used GB=([\d.]+)", log)]
    return {"output_ja": "".join(m[1] for line in lines if (m := re.match(r"^\[.*?\] (.*?)    \(vad ", line))),
            "output_en": " ".join(line.strip()[2:] for line in lines if line.strip().startswith("→ ")),
            "elapsed_s": time.monotonic() - start, "sampled_tree_cpu_s": sum(cpu_times.values()),
            "sampled_tree_peak_rss_bytes": ram_peak, "sampled_gpu_peak_gb": max(gb, default=None),
            "sampled_gpu_mean_gb": statistics.mean(gb) if gb else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV: clip,reference_ja,reference_en[,tags]")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"clip", "reference_ja", "reference_en"} <= rows[0].keys():
        parser.error("provide an annotated clip manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    local = Path(env.get("LOCALAPPDATA", ".")) / "HaasPLaMo" / "dual-llama-bin-path.txt"
    if local.exists():
        env["PATH"] = local.read_text().strip() + os.pathsep + env["PATH"]
    for profile in "ABC":
        cfg = args.output / f"{profile}.toml"
        cfg.write_text(f"[dual_asr]\nenabled = {str(profile != 'A').lower()}\n"
                       f"[reconciler]\nenabled = {str(profile == 'C').lower()}\n")
        results = []
        for i, row in enumerate(rows):
            clip = (args.manifest.resolve().parent / row["clip"]).resolve(strict=True)
            with clip.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            metrics = run_clip(profile, clip, cfg, args.output / f"{profile}-{i:03d}", env)
            results.append(dict(row, sha256=digest, **metrics))
        report = {"profile": profile, "scores": evaluate(results), "clips": results}
        (args.output / f"{profile}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf8")


if __name__ == "__main__":
    main()
