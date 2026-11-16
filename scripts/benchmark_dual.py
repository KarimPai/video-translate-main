"""Benchmark production profiles or an opt-in, isolated Haas/Whisper beam/schedule matrix."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmark_quality import evaluate
from auto_translate.quality_report import build_report, recommend_presets, write_report


def _kill_owned(process):
    """Stop this benchmark child and its descendants, never processes by name."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            process.kill()  # backend/worker's Windows Job owns its descendants
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_clip(profile, clip, cfg, output, env, timeout_s=600):
    try:
        import psutil
    except ImportError:
        psutil = None

    command = [sys.executable, "-m", "auto_translate", str(clip), "--config", str(cfg),
               "--mode", "vod", "--fast", "--no-web", "--verbose", "--wait-optional"]
    cpu_times = {}
    ram_peak = None
    timed_out = False
    start = time.monotonic()
    with output.with_suffix(".txt").open("w", encoding="utf8") as stdout, output.with_suffix(
            ".log").open("w", encoding="utf8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   start_new_session=os.name != "nt")
        parent = psutil.Process(process.pid) if psutil else None
        try:
            while process.poll() is None:
                if time.monotonic() - start >= timeout_s:
                    timed_out = True
                    break
                if parent is None:
                    time.sleep(.25)
                    continue
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
                ram_peak = max(ram_peak or 0, ram)
                time.sleep(.25)
        finally:
            if process.poll() is None:
                _kill_owned(process)
    log = output.with_suffix(".log").read_text(encoding="utf8")
    degraded = profile != "A" and ("Secondary whisper-large-v3-turbo" not in log or "Releasing optional" in log
                                   or "Secondary ASR failed" in log)
    if profile == "C" and (not re.search(r"LFM2\.5.*selector ready", log) or "Selector fallback" in log):
        degraded = True
    lines = output.with_suffix(".txt").read_text(encoding="utf8").splitlines()
    gb = [float(v) for v in re.findall(r"GPU 0 used=([\d.]+) GB", log)]
    free = [float(v) for v in re.findall(r"free=([\d.]+) GB", log)]
    return {"output_ja": "".join(m[1] for line in lines if (m := re.match(r"^\[.*?\] (.*?)    \(vad ", line))),
            "output_en": " ".join(line.strip()[2:] for line in lines if line.strip().startswith("→ ")),
            "elapsed_s": time.monotonic() - start,
            "timeout": timed_out, "exit_code": process.wait(), "degraded": degraded,
            "fallback": degraded, "fallback_reason": "profile_degraded" if degraded else None,
            "audio_duration_s": clip_duration(clip),
            "sampled_tree_cpu_s": sum(cpu_times.values()) if psutil else None,
            "sampled_tree_peak_rss_bytes": ram_peak, "sampled_gpu_peak_gb": max(gb, default=None),
            "sampled_gpu_mean_gb": statistics.mean(gb) if gb else None,
            "gpu_free_gb": min(free, default=None),
            "measurement_scope": "whole CLI including cold startup; no playback"}


def clip_duration(clip):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
                                capture_output=True, text=True, timeout=15, check=True)
        duration = float(result.stdout.strip())
        return duration if 0 < duration < float("inf") else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def benchmark_pair(haas, whisper, audio, *, schedule, timeout_s):
    """One call per engine; at most two concurrent requests, only in benchmark."""
    if schedule not in ("sequential", "concurrent"):
        raise ValueError("schedule must be sequential or concurrent")
    results, errors = {}, []

    def decode(name, model):
        started = time.perf_counter()
        try:
            results[name] = model.transcribe(audio, timeout=timeout_s)
        except Exception as exc:
            errors.append((name, exc))
        finally:
            results[name + "_elapsed_s"] = time.perf_counter() - started

    started = time.perf_counter()
    if schedule == "sequential":
        decode("haas", haas)
        if not errors:
            decode("whisper_turbo", whisper)
    else:
        workers = [threading.Thread(target=decode, args=(name, model), daemon=True)
                   for name, model in (("haas", haas), ("whisper_turbo", whisper))]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + timeout_s + 1
        for worker in workers:
            worker.join(max(0., deadline - time.monotonic()))
        if any(worker.is_alive() for worker in workers):
            raise TimeoutError("Bounded concurrent ASR benchmark exceeded deadline")
    if errors:
        name, error = errors[0]
        raise RuntimeError(f"{name}: {type(error).__name__}: {error}") from error
    elapsed = time.perf_counter() - started
    return dict(results, elapsed_s=elapsed, asr_pair_elapsed_s=elapsed, timeout=False, fallback=False)


def matrix_worker(payload_path, result_path):
    """One isolated setting; parent enforces a hard process deadline."""
    from contextlib import ExitStack
    import numpy as np
    from auto_translate.asr.haas import HaasASR
    from auto_translate.asr.whisper_turbo import WhisperTurboASR
    from auto_translate.config import Config, load_config
    from auto_translate.gpu_budget import gpu_memory_gb
    from auto_translate.types import SAMPLE_RATE
    from auto_translate.windows_job import bind_process_tree
    bind_process_tree()
    payload = json.loads(Path(payload_path).read_text(encoding="utf8"))
    cfg = load_config(Path(payload["config"])) if payload.get("config") else Config()
    rows = []
    # Sample outside the measured decode wall time; nvidia-smi can be slow.
    memory_samples = []
    sampling_stop = threading.Event()

    def sample_gpu():
        while not sampling_stop.is_set():
            used, total = gpu_memory_gb()
            if used is not None and total is not None:
                memory_samples.append((used, total - used))
            sampling_stop.wait(1)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    try:
        with ExitStack() as resources:
            haas = HaasASR(cfg.asr)
            resources.callback(haas.close)
            whisper = WhisperTurboASR(model=cfg.dual_asr.secondary_model,
                                      device=cfg.dual_asr.secondary_device,
                                      compute_type=cfg.dual_asr.secondary_compute_type,
                                      beam_size=payload["beam"], startup_timeout=payload["startup_timeout_s"])
            resources.callback(whisper.close)
            warmed = False
            for row in payload["clips"]:
                decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", row["absolute_clip"],
                                          "-t", str(payload["max_clip_s"] + .1), "-f", "f32le", "-ac", "1",
                                          "-ar", str(SAMPLE_RATE), "pipe:1"],
                                         capture_output=True, timeout=60, check=True)
                audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
                duration = len(audio) / SAMPLE_RATE
                if not 0 < duration <= payload["max_clip_s"]:
                    raise ValueError("ASR matrix requires short nonempty clips within --max-clip-s")
                if not warmed:
                    for _ in range(payload["warmup"]):
                        benchmark_pair(haas, whisper, audio, schedule=payload["schedule"],
                                       timeout_s=payload["request_timeout_s"])
                    warmed = True
                for repetition in range(payload["repetitions"]):
                    metrics = benchmark_pair(haas, whisper, audio, schedule=payload["schedule"],
                                             timeout_s=payload["request_timeout_s"])
                    rows.append(dict(row, **metrics, output_ja=metrics["whisper_turbo"],
                                     audio_duration_s=duration, repetition=repetition,
                                     sampled_gpu_peak_gb=max((s[0] for s in memory_samples), default=None),
                                     gpu_free_gb=min((s[1] for s in memory_samples), default=None),
                                     beam=payload["beam"], schedule=payload["schedule"]))
                    Path(result_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    finally:
        sampling_stop.set()
        sampler.join(3)


def hardware_metadata(label=None):
    hardware = {"label": label or f"{platform.system()} {platform.machine()}",
                "python": platform.python_version(), "packages": {}, "gpu": None}
    for package in ("faster-whisper", "ctranslate2", "numpy"):
        try:
            hardware["packages"][package] = version(package)
        except PackageNotFoundError:
            hardware["packages"][package] = None
    try:
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=3, check=True)
        hardware["gpu"] = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return hardware


def run_matrix(args, rows, env):
    reports = []
    hardware = hardware_metadata(args.hardware_label)
    for beam in args.beams:
        order = args.schedules if args.beams.index(beam) % 2 == 0 else list(reversed(args.schedules))
        for schedule in order:
            name = f"beam{beam}-{schedule}"
            settings = dict(beam=beam, schedule=schedule, repetitions=args.repetitions, warmup=args.warmup)
            payload = dict(settings, clips=rows, request_timeout_s=args.request_timeout_s,
                           startup_timeout_s=args.startup_timeout_s, max_clip_s=args.max_clip_s,
                           config=str(args.config.resolve()) if args.config else None)
            payload_path, result_path = args.output / f"{name}-input.json", args.output / f"{name}-rows.json"
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
            output = args.output / name
            started = time.monotonic()
            timed_out = False
            with output.with_suffix(".txt").open("w", encoding="utf8") as stdout, \
                    output.with_suffix(".log").open("w", encoding="utf8") as stderr:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--asr-worker",
                                            str(payload_path.resolve()), str(result_path.resolve())],
                                           stdout=stdout, stderr=stderr, env=env, start_new_session=os.name != "nt")
                try:
                    process.wait(timeout=args.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    if process.poll() is None:
                        _kill_owned(process)
            results = json.loads(result_path.read_text(encoding="utf8")) if result_path.exists() else []
            report = build_report(results, source=str(args.manifest), hardware_label=hardware["label"])
            report.update(settings=settings, hardware=hardware,
                          measurement_scope="warm ASR pair only; quality is Whisper output; no translation/playback",
                          haas_scores=evaluate([dict(row, output_ja=row.get("haas", "")) for row in results]),
                          workload=[row["sha256"] for row in rows], clips=results,
                          run={"timeout": timed_out, "exit_code": process.returncode,
                               "elapsed_s": time.monotonic() - started},
                          complete=(process.returncode == 0 and not timed_out
                                    and len(results) == len(rows) * args.repetitions))
            reports.append(report)
            write_report(report, args.output, stem=name)
    summary = {"hardware": hardware, "recommendations": recommend_presets(reports),
               "reports": [{key: report.get(key) for key in
                            ("settings", "complete", "quality", "metrics", "run", "measurement_scope")} for report in reports]}
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--asr-worker":
        matrix_worker(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV: clip,reference_ja,reference_en[,tags]")
    parser.add_argument("output", type=Path)
    parser.add_argument("--asr-matrix", action="store_true", help="isolated ASR-only experiment; no production scheduling change")
    parser.add_argument("--beams", default="1,3,6")
    parser.add_argument("--schedules", default="sequential,concurrent")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=600, help="hard deadline per production clip or matrix setting")
    parser.add_argument("--request-timeout-s", type=float, default=30)
    parser.add_argument("--startup-timeout-s", type=float, default=180)
    parser.add_argument("--max-clip-s", type=float, default=30)
    parser.add_argument("--config", type=Path, help="base config for ASR matrix only")
    parser.add_argument("--hardware-label", help="record actual GPU/driver/playback settings for reproducibility")
    args = parser.parse_args()
    try:
        args.beams = [int(value) for value in args.beams.split(",")]
        args.schedules = args.schedules.split(",")
        if not args.beams or any(value not in (1, 3, 6) for value in args.beams):
            raise ValueError("--beams must contain only 1,3,6")
        if not args.schedules or any(value not in ("sequential", "concurrent") for value in args.schedules):
            raise ValueError("--schedules must be sequential,concurrent")
        if not 1 <= args.repetitions <= 100 or not 0 <= args.warmup <= 10:
            raise ValueError("repetitions must be 1..100; warmup 0..10")
        if any(not 0 < value <= 3600 for value in
               (args.timeout_s, args.request_timeout_s, args.startup_timeout_s, args.max_clip_s)):
            raise ValueError("timeouts and clip bound must be finite and within (0,3600]")
    except ValueError as exc:
        parser.error(str(exc))
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"clip", "reference_ja", "reference_en"} <= rows[0].keys():
        parser.error("provide an annotated clip manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    local = Path(env.get("LOCALAPPDATA", ".")) / "HaasPLaMo" / "dual-llama-bin-path.txt"
    if local.exists():
        env["PATH"] = local.read_text().strip() + os.pathsep + env["PATH"]
    for row in rows:
        clip = (args.manifest.resolve().parent / row["clip"]).resolve(strict=True)
        with clip.open("rb") as handle:
            row["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        row["absolute_clip"] = str(clip)
    if args.asr_matrix:
        run_matrix(args, rows, env)
        return
    if args.config:
        parser.error("--config currently applies only to --asr-matrix; production A/B/C use controlled defaults")
    for profile in "ABC":
        cfg = args.output / f"{profile}.toml"
        cfg.write_text(f"[dual_asr]\nenabled = {str(profile != 'A').lower()}\n"
                       f"[reconciler]\nenabled = {str(profile == 'C').lower()}\n")
        results = []
        for i, row in enumerate(rows):
            clip = (args.manifest.resolve().parent / row["clip"]).resolve(strict=True)
            with clip.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            metrics = run_clip(profile, clip, cfg.resolve(), args.output / f"{profile}-{i:03d}", env, args.timeout_s)
            results.append(dict(row, sha256=digest, **metrics))
        report = build_report(results, source=str(args.manifest), hardware_label=args.hardware_label)
        report.update(profile=profile, scores=evaluate(results), clips=results,
                      measurement_scope="whole CLI including cold startup; no playback",
                      complete=all(row["exit_code"] == 0 and not row["timeout"] and not row["degraded"]
                                   for row in results))
        write_report(report, args.output, stem=profile)


if __name__ == "__main__":
    main()
