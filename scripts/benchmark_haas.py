"""Benchmark Haas ASR transcription latency and detect stalls.

Runs a series of transcription requests against the local speech.cpp server
to measure client-side timing breakdown:
- Audio preparation time (WAV encoding)
- HTTP round-trip time (request sent to response received)
- Total client-side time

Compares three scenarios:
A. Haas alone (baseline)
B. Haas during sustained transcription (queue contention)
C. Reports stall statistics (requests > 2s)

Usage:
    python -m scripts.benchmark_haas [manifest.json]

The manifest is the same format as scripts/benchmark.py: a JSON list of
objects with "audio" (WAV file path) and "ja" (reference text).
"""
import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from auto_translate.asr.haas import HaasASR, wav_bytes
from auto_translate.config import load_config
from auto_translate.speech_server import SpeechCppServer, resolve_speech_executable

log = logging.getLogger(__name__)


def generate_silence(duration_s: float) -> np.ndarray:
    """Generate silent audio at 16 kHz."""
    return np.zeros(int(16000 * duration_s), dtype=np.float32)


def benchmark_single(server: SpeechCppServer, wav: bytes, label: str, timeout: float = 30.0) -> dict:
    """Run a single transcription with timing."""
    t0 = time.monotonic()
    prep_start = time.monotonic()
    # wav is already prepared
    prep_time = time.monotonic() - prep_start
    try:
        text, timing = server.transcribe_with_timing(wav, timeout=timeout)
        elapsed = time.monotonic() - t0
        return {
            "label": label,
            "text": text.strip()[:100],
            "audio_prep_s": prep_time,
            "http_request_s": timing.get("http_request_s", 0.0),
            "total_s": timing.get("total_s", 0.0),
            "wall_clock_s": elapsed,
            "stall": timing.get("http_request_s", 0.0) > 2.0,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {
            "label": label,
            "text": "",
            "audio_prep_s": 0.0,
            "http_request_s": 0.0,
            "total_s": 0.0,
            "wall_clock_s": elapsed,
            "stall": False,
            "error": str(exc),
        }


def benchmark_haas_alone(server: SpeechCppServer, clips: list[bytes], timeout: float = 30.0) -> list[dict]:
    """Scenario A: Haas running alone, sequential requests."""
    results = []
    for i, wav in enumerate(clips):
        result = benchmark_single(server, wav, f"alone_{i}", timeout)
        results.append(result)
    return results


def benchmark_sustained(server: SpeechCppServer, clips: list[bytes], timeout: float = 30.0) -> list[dict]:
    """Scenario B: Sustained transcription to measure queue contention."""
    results = []
    for i, wav in enumerate(clips):
        result = benchmark_single(server, wav, f"sustained_{i}", timeout)
        results.append(result)
    return results


def summarize(results: list[dict], label: str) -> dict:
    """Summarize benchmark results."""
    if not results:
        return {"label": label, "count": 0}
    
    http_times = [r["http_request_s"] for r in results if r["error"] is None]
    total_times = [r["total_s"] for r in results if r["error"] is None]
    stalls = [r for r in results if r.get("stall")]
    errors = [r for r in results if r["error"] is not None]
    
    summary = {
        "label": label,
        "count": len(results),
        "errors": len(errors),
        "stalls": len(stalls),
    }
    
    if http_times:
        http_times_sorted = sorted(http_times)
        summary["http_avg_s"] = round(sum(http_times) / len(http_times), 3)
        summary["http_min_s"] = round(min(http_times), 3)
        summary["http_max_s"] = round(max(http_times), 3)
        summary["http_p50_s"] = round(http_times_sorted[len(http_times_sorted) // 2], 3)
        summary["http_p95_s"] = round(http_times_sorted[int(len(http_times_sorted) * 0.95)], 3)
    
    if total_times:
        total_times_sorted = sorted(total_times)
        summary["total_avg_s"] = round(sum(total_times) / len(total_times), 3)
        summary["total_max_s"] = round(max(total_times), 3)
        summary["total_p95_s"] = round(total_times_sorted[int(len(total_times_sorted) * 0.95)], 3)
    
    if stalls:
        summary["stall_http_times"] = [round(r["http_request_s"], 3) for r in stalls]
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Haas ASR transcription latency")
    parser.add_argument("manifest", nargs="?", default=None, help="JSON manifest with audio clips")
    parser.add_argument("--count", type=int, default=10, help="Number of silence clips to generate")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of each silence clip in seconds")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    cfg = load_config()
    asr_cfg = cfg.asr
    
    # Resolve speech executable
    executable = resolve_speech_executable(
        asr_cfg.server_path if asr_cfg.server_path != ".runtime/llama-b12318-cuda12/llama-server.exe" else None
    )
    
    # Create server
    device = "cpu"
    if asr_cfg.cuda_device is not None:
        device = "gpu"
    
    server = SpeechCppServer(
        model_name="haas-ja-v1.0",
        executable=executable,
        device=device,
        threads=asr_cfg.threads,
        startup_timeout_s=asr_cfg.startup_timeout_s,
    )
    
    try:
        server.start()
        log.info("Haas server started on %s (device=%s)", server.base_url, device)
        
        # Prepare audio clips
        clips = []
        if args.manifest:
            manifest_path = Path(args.manifest)
            if manifest_path.exists():
                cases = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                for case in cases:
                    audio_path = Path(case["audio"])
                    if audio_path.exists():
                        audio_data = np.fromfile(str(audio_path), dtype="<i2").astype(np.float32) / 32768
                        wav = wav_bytes(audio_data, normalize=True)
                        clips.append(wav)
                log.info("Loaded %d clips from manifest", len(clips))
        
        if not clips:
            # Generate silence clips
            log.info("Generating %d silence clips (%.1fs each)", args.count, args.duration)
            for i in range(args.count):
                audio = generate_silence(args.duration)
                wav = wav_bytes(audio, normalize=False)
                clips.append(wav)
        
        log.info("Running benchmark with %d clips", len(clips))
        
        # Scenario A: Haas alone
        log.info("=== Scenario A: Haas alone ===")
        alone_results = benchmark_haas_alone(server, clips, args.timeout)
        alone_summary = summarize(alone_results, "Haas alone")
        
        # Scenario B: Sustained transcription
        log.info("=== Scenario B: Sustained transcription ===")
        sustained_results = benchmark_sustained(server, clips, args.timeout)
        sustained_summary = summarize(sustained_results, "Sustained")
        
        # Print results
        print("\n" + "=" * 60)
        print("Haas ASR Benchmark Results")
        print("=" * 60)
        
        for summary in [alone_summary, sustained_summary]:
            print(f"\n{summary['label']}:")
            print(f"  Count: {summary['count']}")
            print(f"  Errors: {summary['errors']}")
            print(f"  Stalls (>2s): {summary['stalls']}")
            if "http_avg_s" in summary:
                print(f"  HTTP avg: {summary['http_avg_s']}s")
                print(f"  HTTP min: {summary['http_min_s']}s")
                print(f"  HTTP max: {summary['http_max_s']}s")
                print(f"  HTTP p50: {summary['http_p50_s']}s")
                print(f"  HTTP p95: {summary['http_p95_s']}s")
            if "total_avg_s" in summary:
                print(f"  Total avg: {summary['total_avg_s']}s")
                print(f"  Total max: {summary['total_max_s']}s")
                print(f"  Total p95: {summary['total_p95_s']}s")
            if "stall_http_times" in summary:
                print(f"  Stall times: {summary['stall_http_times']}")
        
        # Save results to JSON
        results = {
            "clips": len(clips),
            "clip_duration_s": args.duration,
            "device": device,
            "scenarios": {
                "alone": alone_summary,
                "sustained": sustained_summary,
            },
            "details": {
                "alone": alone_results,
                "sustained": sustained_results,
            },
        }
        
        output_path = Path("logs") / "haas_benchmark.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {output_path}")
        
    finally:
        server.close()


if __name__ == "__main__":
    main()
