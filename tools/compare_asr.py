"""Benchmark the same YouTube audio and speech segments with 4 ASR settings.

Run from the project folder:
    python tools/compare_asr.py "https://www.youtube.com/watch?v=LE8iGHrakAc"

This is an inference comparison, NOT a ground-truth accuracy score. Inspect
the speech/terms in compare_finetune.txt and listen to saved WAV clips.
Uses the project's existing huggingface-hub dependency for the JA fine-tune.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_URL = "https://www.youtube.com/watch?v=LE8iGHrakAc"
JA_REPO = "kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF"
JA_FILENAME = "Qwen3-ASR-1.7B-JA-Q8_0.gguf"
# Keep the baseline equivalent to the user's current application: Haas on Vulkan
# with GPU decoding, without custom terms. Boosted Haas is a separate experiment.
KEYWORDS = "顔パンパンパン=かおぱんぱんぱん\nパンパンパン=ぱんぱんぱん"
# This clip is about a bread item, not Watame, Botan, or shutters.
# Avoid priming Qwen with irrelevant vocabulary; the earlier prompted run
# hallucinated its exact prompt on six of the 32 speech regions.


def run(args: list[str], *, output: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(">", " ".join(str(x) for x in args[:7]), "..." if len(args) > 7 else "", flush=True)
    result = subprocess.run(args, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if output:
        output.write_text(result.stdout + "\n\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Command exited {result.returncode}: {args[:5]}\n"
            + result.stderr[-3000:]
        )
    return result


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        token = parsed.path.strip("/").split("/")[0]
    elif parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        token = parse_qs(parsed.query).get("v", [""])[0]
    else:
        raise ValueError("Expected a YouTube video URL.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", token):
        raise ValueError("Invalid YouTube video ID.")
    return token


def locate_exe(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Missing {name} on PATH. Check using: where {name}")
    return path


def parse_lines(text: str, expected: int) -> list[str]:
    entries = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "text" in item:
            entries.append(str(item["text"]).strip())
    if len(entries) != expected:
        raise ValueError(
            f"Expected {expected} speech.cpp JSON results but received "
            f"{len(entries)}. See raw logs."
        )
    return entries


def benchmark(
    label: str,
    speech: str,
    model: str,
    wavs: list[Path],
    args: list[str],
    folder: Path,
) -> tuple[float, list[str]]:
    command = [
        speech, "asr", model, "--device", "Vulkan0", "--format", "json",
        *args, *(str(p) for p in wavs)
    ]
    started = time.perf_counter()
    response = run(command, output=folder / f"{label}.log")
    elapsed = time.perf_counter() - started
    return elapsed, parse_lines(response.stdout, len(wavs))


def fetch_japanese_finetune() -> str:
    """Fetch the speech.cpp-compatible conversion of Neosophie's JA fine-tune.

    The original neosophie safetensors checkpoint cannot be loaded by
    speech.cpp. Use this separate, documented speech.cpp Q8_0 conversion,
    not the different llama.cpp GGUF layout.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required by the project; run uv sync "
            "from the test project directory."
        ) from exc
    print(f"Checking Japanese fine-tune: {JA_REPO}/{JA_FILENAME}", flush=True)
    print("First use downloads about 2.18 GB; existing downloads are cached.", flush=True)
    return hf_hub_download(repo_id=JA_REPO, filename=JA_FILENAME)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--seconds", type=int, default=120,
                        help="First N seconds to compare; default 120, maximum 300")
    options = parser.parse_args()
    seconds = max(15, min(options.seconds, 300))
    video_id = youtube_id(options.url)

    speech = locate_exe("speech")
    yt = locate_exe("yt-dlp")
    ffmpeg = locate_exe("ffmpeg")
    folder = Path("logs") / "asr_benchmark" / video_id
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "compare_finetune.txt"
    # Create the report immediately, so an interrupted download or ASR
    # subprocess cannot leave the user with no diagnostic file.
    report_path.write_text(
        f"Video: {options.url}\n"
        f"Compared: first {seconds} seconds\n"
        "Status: Preparing input audio (benchmark not finished).\n",
        encoding="utf-8",
    )
    print(f"Video: {options.url}\nAudio limit: {seconds}s\nResults: {folder}", flush=True)
    print(f"Report initialized: {report_path.resolve()}", flush=True)

    # Download a single audio stream once: all ASR models see identical samples.
    raw_candidates = list(folder.glob("source.*"))
    if raw_candidates:
        original = raw_candidates[0]
        print("Reusing downloaded audio:", original)
    else:
        template = str(folder / "source.%(ext)s")
        yt_args = [
            yt, "--no-playlist", "--no-warnings",
            "--remote-components", "ejs:github", "-f", "bestaudio",
            "-o", template,
        ]
        # Optional cookies file in test project; no browser access required.
        if Path("cookies.txt").is_file():
            yt_args.extend(["--cookies", "cookies.txt"])
        run([*yt_args, options.url], output=folder / "yt-dlp.log")
        raw_candidates = [p for p in folder.glob("source.*") if p.is_file()]
        if not raw_candidates:
            raise RuntimeError("yt-dlp did not create an audio file.")
        original = raw_candidates[0]

    wav = folder / "source_16k.wav"
    if not wav.exists():
        run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(original), "-t", str(seconds),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
        ], output=folder / "ffmpeg.log")

    # One shared VAD pass gives all models identical segmentation. The
    # production app uses FireRedVAD, so actual live latency may differ.
    segment_dir = folder / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments = sorted(segment_dir.glob("*.wav"))
    if not segments:
        run([
            speech, "vad", "silero-vad",
            "--device", "cpu",
            "--threshold", "0.35",
            "--speech-pad-ms", "80",
            "--min-silence-duration-ms", "400",
            "--min-speech-duration-ms", "100",
            "--max-speech-duration-s", "5",
            "--split", str(segment_dir), str(wav),
        ], output=folder / "vad.log")
        segments = sorted(segment_dir.glob("*.wav"))
    if not segments:
        raise RuntimeError("VAD found no speech. Examine source_16k.wav.")

    print(f"VAD extracted {len(segments)} identical speech segments.", flush=True)
    report_path.write_text(
        f"Video: {options.url}\n"
        f"Compared: first {seconds} seconds | {len(segments)} shared VAD segments\n"
        "Status: Preparing Qwen models for inference (benchmark not finished).\n"
        "Original baseline and clean reports are preserved.\n",
        encoding="utf-8",
    )
    # Fetch up front so progress appears in the terminal rather than being
    # swallowed by the recognition subprocess that captures stdout/stderr.
    print("Checking Qwen model cache (download progress, if needed, follows)...", flush=True)
    base_qwen_error = None
    try:
        subprocess.run([speech, "pull", "qwen3-asr-1.7b"], check=True,
                       stdout=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        base_qwen_error = str(exc)
        print(f"Base Qwen download failed: {exc}", file=sys.stderr, flush=True)
    finetune_path = None
    finetune_error = None
    try:
        finetune_path = fetch_japanese_finetune()
    except Exception as exc:
        finetune_error = str(exc)
        print("Fine-tune download failed:", exc, file=sys.stderr, flush=True)
    plans = [
        ("haas_baseline", "haas-ja-v1.0", ["--decode-on-cpu=false"]),
        ("haas_scene_keywords", "haas-ja-v1.0",
         ["--decode-on-cpu=false", "--keywords", KEYWORDS]),
        ("qwen3_unprompted", "qwen3-asr-1.7b", ["--language", "ja"]),
        ("qwen3_ja_finetuned", finetune_path, ["--language", "ja"]),
    ]
    outputs: dict[str, list[str]] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    if base_qwen_error:
        errors["qwen3_unprompted"] = base_qwen_error
    if finetune_error:
        errors["qwen3_ja_finetuned"] = finetune_error

    def save_report(stage: str) -> None:
        lines = [
            f"Video: {options.url}",
            f"Compared: first {seconds} seconds | {len(segments)} shared VAD segments",
            f"Status: {stage}",
            "Elapsed time includes model load; model downloads occur before timing.",
            "Speech.cpp VAD differs from the production FireRedVAD.",
            "No CER/WER is calculated: manually verify the audio/JP reference.",
            "Haas keywords: " + KEYWORDS.replace("\n", " ; "),
            "Qwen prompt: disabled for both base and Japanese fine-tune",
            "Fine-tune source: " + JA_REPO,
            "Fine-tune quantization: speech.cpp Q8_0, not the original BF16 checkpoint",
            "",
        ]
        for label, _, _ in plans:
            value = (f"{timings[label]}s" if label in timings
                     else "ERROR" if label in errors else "PENDING")
            lines.append(f"{label}: {value}")
        for label, error in errors.items():
            lines.append(f"ERROR {label}: {error[:1500]}")
        for index, segment in enumerate(segments):
            lines.append(f"\n--- Segment {index + 1:03d}: {segment.name} ---")
            for label, _, _ in plans:
                text = outputs.get(label)
                lines.append(f"{label}: {text[index] if text else 'N/A'}")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return None

    save_report("Models prepared; starting ASR comparisons.")
    for label, model, flags in plans:
        print(f"\nRunning {label}...", flush=True)
        if not model or (label == "qwen3_unprompted" and base_qwen_error):
            errors[label] = (errors.get(label) or
                             "Fine-tuned model was not downloaded.")
            print(f"SKIPPED {label}: {errors[label]}", file=sys.stderr, flush=True)
            save_report(f"Completed through {label}: skipped due to error.")
            continue
        try:
            elapsed, texts = benchmark(label, speech, model, segments, flags, folder)
            timings[label] = round(elapsed, 3)
            outputs[label] = texts
            print(f"{label}: {elapsed:.1f}s total process time", flush=True)
        except Exception as exc:
            errors[label] = str(exc)
            print(f"FAILED {label}: {exc}", file=sys.stderr, flush=True)
        save_report(f"Completed through {label}.")

    complete = "qwen3_ja_finetuned" in outputs
    save_report("COMPLETE" if complete else "INCOMPLETE - inspect ERROR lines above.")
    print(f"\nReport saved to {report_path.resolve()}", flush=True)
    if not complete:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
