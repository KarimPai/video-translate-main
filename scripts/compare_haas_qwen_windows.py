"""Compare Haas and Qwen3-ASR on identical holoGTA audio windows.

This is a diagnostic ASR-only experiment: no MT, no VAD, no production settings
changed. Output similarity/coverage is NOT a Japanese accuracy score.
Run from the repository with:
  uv run python scripts/compare_haas_qwen_windows.py hologta-stress-test.flac

Models are loaded one at a time and shut down before the next model.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
STANDARD_QWEN_MODEL = "Qwen3-ASR-1.7B-Q8_0.gguf"
STANDARD_QWEN_MMPROJ = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"
# Deliberately short, scene-relevant words. NOT a forced Japanese prompt.
DEFAULT_QWEN_VOCAB = "みこ, みこち, ラミィ, ルイ, まつり, すいちゃん, スイちゃん"
# Local clip time, not timestamp in the original 1:25:00-offset stream.
WINDOWS = {
    # Name, crowded dialogue, monetary-negation and speech-presence excerpts.
    "HG02": (15.90, 19.52),
    "HG05": (190.0, 205.0),
    "HG06": (224.00, 238.40),
    "HG07": (254.18, 268.19),
    "HG10": (350.69, 366.18),
    "HG11": (385.0, 401.0),
    "HG12": (413.95, 425.70),
    "HG13": (425.0, 441.0),
    "HG19": (736.0, 751.0),
    "HG21": (823.0, 838.0),
    "HG26": (1064.50, 1072.40),
    "HG27": (1076.80, 1083.50),
    "HG28": (1097.80, 1103.00),
    "HG29": (1107.10, 1117.60),
    # Full-pipeline regression windows from the 20-minute holoGTA sample:
    # Q14 lost audible words when 14.4s took the split-first path.
    # Q16 preserved text and saved 3.36s when 15.8s was split-first.
    "Q14": (224.0, 238.4),
    "Q16": (611.2, 627.0),
}
MODELS = ("haas", "qwen3-asr")
COLUMNS = (
    "window", "model", "variant", "part", "start_s", "end_s",
    "audio_s", "elapsed_s", "ja_chars", "language", "rejection", "text",
)
JA_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def clean_for_comparison(text: str) -> str:
    """Normalized Japanese script only, for disagreement (not CER)."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(JA_CHAR.findall(normalized))


def similarity(left: str, right: str) -> float | None:
    left, right = clean_for_comparison(left), clean_for_comparison(right)
    if not left or not right:
        return None
    return round(SequenceMatcher(None, left, right, autojunk=False).ratio(), 3)


def split_ranges(length: int, overlap_s: float = 0.4) -> list[tuple[int, int]]:
    """Two near-half slices with a bounded shared audio boundary."""
    if length < SAMPLE_RATE * 3:
        raise ValueError("Window must contain at least 3 seconds of audio")
    overlap_samples = min(int(overlap_s * SAMPLE_RATE), length // 4)
    middle = length // 2
    return [
        (0, min(length, middle + overlap_samples // 2)),
        (max(0, middle - overlap_samples // 2), length),
    ]


def pause_guided_ranges(audio: np.ndarray, target_s: float = 10.0,
                        overlap_s: float = 0.24) -> list[tuple[int, int]]:
    """Experimental <=target_s chunks; search quieter 80ms frames near each cut.

    The energy minimum is NOT a verified speech pause. Boundaries retain a
    small shared context region; outputs are deliberately not auto-merged.
    """
    if not 3.0 <= target_s <= 15.0 or not 0 <= overlap_s <= 0.5:
        raise ValueError("Invalid target or overlap")
    length = len(audio)
    if length <= int(target_s * SAMPLE_RATE):
        return [(0, length)]
    frame = int(0.08 * SAMPLE_RATE)
    radius = int(0.65 * SAMPLE_RATE)
    # Leave room for independent pause-search deviations on both boundaries,
    # the overlap, and one alignment frame. Otherwise a 19s clip can exceed
    # a 10s chunk limit even though the nominal centers fit.
    safety_s = 2 * radius / SAMPLE_RATE + overlap_s + frame / SAMPLE_RATE
    parts = max(2, math.ceil(length / ((target_s - safety_s) * SAMPLE_RATE)))
    boundaries = [0]
    for i in range(1, parts):
        intended = round(i * length / parts)
        lo = max(boundaries[-1] + frame, intended - radius)
        hi = min(length - frame, intended + radius)
        choices = range(lo, hi + 1, frame)
        split = min(choices, key=lambda pos: float(np.mean(
            np.square(audio[pos - frame // 2:pos + frame // 2].astype(np.float64))
        ))) if lo <= hi else intended
        boundaries.append(split)
    boundaries.append(length)
    shared = round(overlap_s * SAMPLE_RATE)
    ranges = [
        (max(0, boundaries[i] - (shared // 2 if i else 0)),
         min(length, boundaries[i + 1] + (shared - shared // 2 if i < parts - 1 else 0)))
        for i in range(parts)
    ]
    if max((right - left) / SAMPLE_RATE for left, right in ranges) > target_s + 0.002:
        # Pathological energy minima should never exceed the stated target.
        raise ValueError("Pause-based split exceeded requested maximum duration")
    return ranges


def decode_window(media: Path, start_s: float, end_s: float, ffmpeg: str) -> np.ndarray:
    """Extract exactly one target window as 16 kHz mono float32 PCM."""
    if not 0 <= start_s < end_s <= 1200:
        raise ValueError("Expected valid local times within the 20-minute clip")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start_s:.3f}", "-i", str(media),
        "-t", f"{end_s - start_s:.3f}", "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {start_s}-{end_s}: "
            + result.stderr.decode("utf-8", "replace")[-1200:]
        )
    if len(result.stdout) % 4:
        raise RuntimeError("ffmpeg generated incomplete float32 samples")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError(f"Missing or invalid audio for {start_s}-{end_s}")
    return samples


def run_case(model: object, model_name: str, window: str,
             start_s: float, audio: np.ndarray, *,
             variant: str, part: int, left: int, right: int) -> dict:
    """Decode the same PCM and reset conversational history for each request."""
    chunk = audio[left:right]
    if hasattr(model, "forget"):
        model.forget()
    begun = time.perf_counter()
    try:
        text = str(model.transcribe(chunk, probe=False) or "")
        rejection = getattr(model, "last_rejection", None) or ""
        language = getattr(model, "last_detected_language", None) or (
            "not_reported" if model_name == "haas" else "und"
        )
    except Exception as exc:
        text = ""
        language = "error"
        rejection = f"{type(exc).__name__}: {exc}"
        logging.exception("%s %s %s part %s failed", model_name, window, variant, part)
    return {
        "window": window, "model": model_name, "variant": variant, "part": part,
        "start_s": round(start_s + left / SAMPLE_RATE, 3),
        "end_s": round(start_s + right / SAMPLE_RATE, 3),
        "audio_s": round(len(chunk) / SAMPLE_RATE, 3),
        "elapsed_s": round(time.perf_counter() - begun, 3),
        "ja_chars": len(clean_for_comparison(text)),
        "language": str(language), "rejection": str(rejection), "text": text,
    }


def run_model(model: object, name: str, windows: dict[str, tuple[float, np.ndarray]],
              *, qwen_split_first: bool = False,
              qwen_short_target_s: float | None = None,
              qwen_quality_recovery: bool = False,
              qwen_vocabulary: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for label, (start_s, audio) in windows.items():
        length = len(audio)
        if name == "qwen3-asr":
            # For shortening A/B, use the production 15s split-first default.
            # Historical --qwen-split-first A/B retains its explicit old-style
            # control (full 256->512->split vs split-first).
            model._prefer_split_on_truncation = bool(
                qwen_short_target_s is not None or qwen_quality_recovery
                or qwen_vocabulary is not None
            )
        if qwen_vocabulary is not None and name == "qwen3-asr":
            # ABA rather than just AB: normal Qwen can differ between two
            # unprompted runs, even at temperature 0. Only a stable A/A
            # control permits a stronger attribution to vocabulary hints.
            old_hint = getattr(model, "_vocabulary_hint", "")
            try:
                model.set_vocabulary_hint("")
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="full", part=0, left=0, right=length,
                ))
                model.set_vocabulary_hint(qwen_vocabulary)
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="vocab_hint", part=0, left=0, right=length,
                ))
                model.set_vocabulary_hint("")
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="control_repeat", part=0, left=0, right=length,
                ))
            finally:
                model.set_vocabulary_hint(old_hint)
        elif qwen_quality_recovery and name == "qwen3-asr":
            # Control: never allow an already-enabled local config to bias
            # the baseline. Restore the original option after the second arm.
            previous_recovery = model.cfg.recover_rejected_speech
            try:
                model.cfg.recover_rejected_speech = False
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="full", part=0, left=0, right=length,
                ))
                model.cfg.recover_rejected_speech = True
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="quality_recovery", part=0, left=0, right=length,
                ))
            finally:
                model.cfg.recover_rejected_speech = previous_recovery
        else:
            rows.append(run_case(
                model, name, label, start_s, audio,
                variant="full", part=0, left=0, right=length,
            ))
        if qwen_split_first and name == "qwen3-asr":
            # A/B comparison on exactly the same long PCM and model instance:
            # default 256->512->split vs experimental 256->split.
            try:
                model._prefer_split_on_truncation = True
                rows.append(run_case(
                    model, name, label, start_s, audio,
                    variant="split_first", part=0, left=0, right=length,
                ))
            finally:
                model._prefer_split_on_truncation = False
        if qwen_short_target_s is not None and name == "qwen3-asr":
            if length > int(qwen_short_target_s * SAMPLE_RATE):
                for part, (left, right) in enumerate(
                    pause_guided_ranges(audio, qwen_short_target_s), start=1,
                ):
                    rows.append(run_case(
                        model, name, label, start_s, audio,
                        variant="short", part=part, left=left, right=right,
                    ))
        for part, (left, right) in enumerate(split_ranges(length), start=1):
            rows.append(run_case(
                model, name, label, start_s, audio,
                variant="split", part=part, left=left, right=right,
            ))
    return rows


def english_report(rows: list[dict], media_name: str) -> str:
    lines = [
        "# Haas vs Qwen3-ASR: holoGTA targeted window experiment",
        "",
        f"Source: {media_name}",
        "",
        "**Not an accuracy ranking.** Same audio is decoded as a full window and "
        "as two overlapping halves. Split halves are displayed separately; "
        "they are not merged into subtitles. Similarity measures agreement, "
        "not whether either transcript is correct. No-speech status is not verified.",
        "",
        "| Window | Haas full JA chars | Qwen full JA chars | Full-text similarity | "
        "Haas split chars | Qwen split chars | Flags |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for window in WINDOWS:
        get = lambda model, variant: [row for row in rows
            if row["window"] == window and row["model"] == model
            and row["variant"] == variant]
        haas = get("haas", "full")
        qwen = get("qwen3-asr", "full")
        if not haas and not qwen:
            continue
        h0, q0 = haas[0] if haas else None, qwen[0] if qwen else None
        hs = get("haas", "split")
        qs = get("qwen3-asr", "split")
        same = similarity(h0["text"], q0["text"]) if h0 and q0 else None
        flags = []
        for record in [r for r in (h0, q0) if r is not None]:
            if not record["ja_chars"]:
                flags.append(f'{record["model"]}: no JA text')
            if record["rejection"]:
                flags.append(f'{record["model"]}: {record["rejection"][:28]}')
        for name, parts in (("Haas", hs), ("Qwen", qs)):
            if parts and any(not r["ja_chars"] for r in parts):
                flags.append(f"{name}: empty split part")
        lines.append(
            f"| {window} | {h0['ja_chars'] if h0 else '-'} | "
            f"{q0['ja_chars'] if q0 else '-'} | "
            f"{same if same is not None else '-'} | "
            f"{sum(r['ja_chars'] for r in hs) if hs else '-'} | "
            f"{sum(r['ja_chars'] for r in qs) if qs else '-'} | "
            f"{'; '.join(flags) if flags else '-'} |"
        )
    qwen_trials = [row for row in rows if row["model"] == "qwen3-asr"
                   and row["variant"] == "split_first"]
    if qwen_trials:
        lines.extend([
            "",
            "## Qwen experimental split-first after a 256-token truncation",
            "",
            "The experimental path has **no effect** unless the full long request "
            "hits its token limit. It retains the existing two-window "
            "acceptance gates. Lower latency is not a verified accuracy gain.",
            "",
            "| Window | Default full time (s) | Split-first time (s) | "
            "Default JA chars | Split-first JA chars | Agreement |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for candidate in qwen_trials:
            base = next((row for row in rows if row["model"] == "qwen3-asr"
                         and row["window"] == candidate["window"]
                         and row["variant"] == "full"), None)
            if not base:
                continue
            sim = similarity(base["text"], candidate["text"])
            lines.append(
                f'| {candidate["window"]} | {base["elapsed_s"]:.3f} | '
                f'{candidate["elapsed_s"]:.3f} | {base["ja_chars"]} | '
                f'{candidate["ja_chars"]} | '
                f'{sim if sim is not None else "-"} |'
            )
    vocabulary_rows = [r for r in rows if r["model"] == "qwen3-asr"
                       and r["variant"] == "vocab_hint"]
    if vocabulary_rows:
        lines.extend([
            "", "## Qwen vocabulary hint experiment (A/B/A)", "",
            "System vocabulary may improve proper nouns but can also prime "
            "hallucinations. The second unprompted control checks whether "
            "differences simply reflect repeated-decode variability.", "",
            "| Window | Control1 (chars) | Vocabulary (chars) | Control2 (chars) | "
            "Unprompted controls agree | Hint-only change | Flags |",
            "|---|---:|---:|---:|---|---|---|",
        ])
        for hinted in vocabulary_rows:
            win = hinted["window"]
            normal = next((r for r in rows if r["model"] == "qwen3-asr"
                           and r["window"] == win and r["variant"] == "full"), None)
            repeated = next((r for r in rows if r["model"] == "qwen3-asr"
                             and r["window"] == win
                             and r["variant"] == "control_repeat"), None)
            if normal is None or repeated is None:
                continue
            same = normal["text"] == repeated["text"] and normal["language"] == repeated["language"]
            unique_change = same and (hinted["text"] != normal["text"]
                                      or hinted["language"] != normal["language"])
            flags = []
            if not same:
                flags.append("baseline_varies")
            if not normal["text"] and hinted["text"]:
                flags.append("hint_on_unverified_silence")
            if hinted["language"] != normal["language"]:
                flags.append("language_changed")
            if hinted["rejection"]:
                flags.append("hint_rejected")
            lines.append(
                f'| {win} | {normal["ja_chars"]} | {hinted["ja_chars"]} | '
                f'{repeated["ja_chars"]} | {"yes" if same else "NO"} | '
                f'{"yes" if unique_change else "no"} | {", ".join(flags) or "-"} |'
            )
    quality_rows = [row for row in rows if row["model"] == "qwen3-asr"
                    and row["variant"] == "quality_recovery"]
    if quality_rows:
        lines.extend([
            "", "## Qwen rejected-speech recovery A/B (opt-in)", "",
            "Only malformed/looping hypotheses that the existing guard would "
            "discard are eligible. **This is not a name or number correction.** "
            "An unchanged result means the new recovery was not useful on that window.", "",
            "| Window | Normal characters | Recovery characters | "
            "Normal rejection | Recovery rejection | Match | Extra time (s) |",
            "|---|---:|---:|---|---|---:|---:|",
        ])
        for retry in quality_rows:
            full = next((r for r in rows if r["model"] == "qwen3-asr"
                         and r["window"] == retry["window"]
                         and r["variant"] == "full"), None)
            if full is None:
                continue
            # Character similarity is agreement, not proof of recognition.
            agreement = similarity(full["text"], retry["text"])
            lines.append(
                f'| {retry["window"]} | {full["ja_chars"]} | '
                f'{retry["ja_chars"]} | {full["rejection"] or "-"} | '
                f'{retry["rejection"] or "-"} | '
                f'{agreement if agreement is not None else "-"} | '
                f'{retry["elapsed_s"] - full["elapsed_s"]:.3f} |'
            )
    short_trials = [row for row in rows if row["model"] == "qwen3-asr"
                    and row["variant"] == "short"]
    if short_trials:
        lines.extend([
            "", "## Qwen experimental shorter segments (pause-guided)", "",
            "This is ASR-only window slicing, NOT a production VAD comparison. "
            "Overlapping cuts can duplicate words; split parts are not merged.", "",
            "| Window | Full JA chars | Short parts JA chars (overlap counted) | "
            "Full time (s) | Short total time (s) | Parts |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for window in WINDOWS:
            parts = [r for r in short_trials if r["window"] == window]
            full = next((r for r in rows if r["model"] == "qwen3-asr"
                         and r["window"] == window and r["variant"] == "full"), None)
            if full is None or not parts:
                continue
            lines.append(
                f'| {window} | {full["ja_chars"]} | '
                f'{sum(r["ja_chars"] for r in parts)} | '
                f'{full["elapsed_s"]:.3f} | '
                f'{sum(r["elapsed_s"] for r in parts):.3f} | {len(parts)} |'
            )
    lines += [
        "",
        "## How to interpret",
        "",
        "- Compare full-text output with the two shorter requests for *each* model.",
        "- A longer transcript may add real speech **or** hallucinated words.",
        "- The split character totals are inflated by any repeated boundary words.",
        "- A full-model miss and coherent shorter outputs suggest testing adaptive "
        "segmentation, but they do not prove those shorter outputs are accurate.",
        "- The CSV and JSONL retain the Japanese hypotheses for automatic "
        "cross-run comparison; no Japanese reading is required.",
        "- No production ASR/VAD settings were changed.",
        "",
    ]
    return "\n".join(lines)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="20-minute FLAC or video clip")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--model", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--window", nargs="+", choices=list(WINDOWS),
                        default=list(WINDOWS), help="Only these regression windows")
    parser.add_argument("--outdir", type=Path, default=ROOT / "benchmarks" / "runs")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable or path")
    parser.add_argument("--qwen-model-dir", type=Path,
                        default=ROOT / "models" / "qwen3-asr",
                        help="Explicit standard Qwen model/projector folder; avoids JA auto-discovery")
    parser.add_argument("--qwen-vocabulary", action="store_true",
                        help="A/B/A standard Qwen with a short recognition vocabulary system prompt")
    parser.add_argument("--qwen-vocabulary-terms", type=str,
                        default=DEFAULT_QWEN_VOCAB,
                        help="Comma-separated Japanese names for --qwen-vocabulary")
    parser.add_argument("--qwen-recover-rejected", action="store_true",
                        help="A/B normal decode versus opt-in salvage after clearly rejected speech")
    parser.add_argument("--qwen-short-target-s", type=float, choices=[8.0, 10.0, 12.0],
                        help="Experimental <=N second pause-guided splits, in addition to full controls")
    parser.add_argument("--qwen-split-first", action="store_true",
                        help="A/B test conditional split-first after Qwen 256-token truncation")
    parser.add_argument("--dry-run", action="store_true", help="Show planned work; load no models")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    media = args.media.expanduser().resolve()
    config_file = args.config.expanduser().resolve()
    if not media.is_file() or not config_file.is_file():
        print("Audio/video or config file not found", file=sys.stderr)
        return 2
    requested_arms = sum(bool(v) for v in (
        args.qwen_short_target_s, args.qwen_split_first, args.qwen_recover_rejected,
        args.qwen_vocabulary,
    ))
    if requested_arms > 1:
        print("Run shortening, split-first, rejected-speech recovery and vocabulary separately",
              file=sys.stderr)
        return 2
    if args.qwen_vocabulary and "qwen3-asr" not in args.model:
        print("--qwen-vocabulary requires --model qwen3-asr", file=sys.stderr)
        return 2
    if args.qwen_vocabulary and not args.qwen_vocabulary_terms.strip():
        print("--qwen-vocabulary-terms must not be blank", file=sys.stderr)
        return 2
    qwen_dir = args.qwen_model_dir.expanduser().resolve()
    qwen_model = qwen_dir / STANDARD_QWEN_MODEL
    qwen_mmproj = qwen_dir / STANDARD_QWEN_MMPROJ
    if not args.dry_run and "qwen3-asr" in args.model:
        missing = [path for path in (qwen_model, qwen_mmproj) if not path.is_file()]
        if missing:
            print("Standard Qwen requires its explicit model/projector pair:",
                  *(str(path) for path in missing), sep="\n  ", file=sys.stderr)
            return 2
    if not args.dry_run and not shutil.which(args.ffmpeg):
        print(f"ffmpeg executable unavailable: {args.ffmpeg}", file=sys.stderr)
        return 2
    windows = dict.fromkeys(args.window)
    print("ASR-only. Models run SEQUENTIALLY. No translation model.")
    print("Windows:", ", ".join(windows))
    print("Models:", ", ".join(args.model))
    print("For each window: one full decode, two overlapping half decodes.")
    if args.qwen_short_target_s:
        print(f"Experimental Qwen pause-guided segments: <= {args.qwen_short_target_s}s")
    if args.qwen_recover_rejected:
        print("A/B: normal Qwen versus opt-in recovery of rejected hypotheses")
    if args.qwen_vocabulary:
        print("A/B/A: normal Qwen, vocabulary hints, normal Qwen repeat")
        print("Vocabulary words:", args.qwen_vocabulary_terms)
    if "qwen3-asr" in args.model:
        print("Standard Qwen model:", qwen_model)
        print("Standard Qwen mmproj:", qwen_mmproj)
    if args.dry_run:
        return 0

    from auto_translate.config import get_backend_config, load_config
    from auto_translate.registry import get_default_registry

    config = load_config(config_file)
    data = {
        name: (WINDOWS[name][0],
               decode_window(media, *WINDOWS[name], ffmpeg=args.ffmpeg))
        for name in windows
    }
    result_dir = args.outdir.expanduser().resolve() / (
        "asr-windows-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    try:
        for model_name in dict.fromkeys(args.model):
            model = None
            try:
                backend_config = get_backend_config(config, "asr", model_name)
                if model_name == "qwen3-asr":
                    backend_config = dict(backend_config)
                    backend_config.update(
                        model_path=str(qwen_model), mmproj_path=str(qwen_mmproj),
                    )
                model = get_default_registry().create_asr(model_name, backend_config)
                rows.extend(run_model(
                    model, model_name, data, qwen_split_first=args.qwen_split_first,
                    qwen_short_target_s=args.qwen_short_target_s,
                    qwen_quality_recovery=args.qwen_recover_rejected,
                    qwen_vocabulary=(args.qwen_vocabulary_terms
                                     if args.qwen_vocabulary else None),
                ))
            finally:
                if model is not None:
                    model.close()
    finally:
        csv_path = result_dir / "haas_qwen_windows.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        jsonl_path = result_dir / "haas_qwen_windows.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = result_dir / "report.md"
        report.write_text(english_report(rows, media.name), encoding="utf-8")
        print("CSV:", csv_path)
        print("JSONL:", jsonl_path)
        print("English report:", report)
    if any(row["language"] == "error" for row in rows):
        print("One or more ASR decodes failed; inspect the JSONL rejection fields.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
