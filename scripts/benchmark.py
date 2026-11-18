"""Evaluate reviewed mono 16 kHz WAV clips with the configured local models.

Cases sharing a ``group`` run in manifest order as one conversation: they share
translation history and ASR continuity, as consecutive cues do in a stream.
"""
import argparse
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import replace
import json
import logging
from pathlib import Path
import time
import unicodedata
import wave

import numpy as np

from auto_translate.__main__ import _load_translator
from auto_translate.asr import asr_context, create_asr
from auto_translate.config import load_config
from auto_translate.glossary import Glossary
from auto_translate.gpu_budget import GpuBudget
from auto_translate.stream_context import StreamContext, describe_stream
from auto_translate.subtitle_clean import clean_english_output
from auto_translate.translate.hymt import useful_context

CHRF_ORDER = 6
CHRF_BETA = 2


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def chrf_counts(reference: str, hypothesis: str) -> list[tuple[int, int, int]]:
    """Per-order (matches, hypothesis n-grams, reference n-grams) over non-space characters."""
    ref, hyp = "".join(reference.split()), "".join(hypothesis.split())
    counts = []
    for n in range(1, CHRF_ORDER + 1):
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        counts.append((sum((r & h).values()), sum(h.values()), sum(r.values())))
    return counts


def chrf(pairs: list[tuple[str, str]]) -> float | None:
    """Corpus chrF (character n-gram F-score, beta 2) on a 0-100 scale."""
    totals = [[0, 0, 0] for _ in range(CHRF_ORDER)]
    for reference, hypothesis in pairs:
        for total, counts in zip(totals, chrf_counts(reference, hypothesis)):
            for i, value in enumerate(counts):
                total[i] += value
    orders = [(match / hyp if hyp else 0.0, match / ref) for match, hyp, ref in totals if ref]
    if not orders:
        return None
    precision = sum(p for p, _ in orders) / len(orders)
    recall = sum(r for _, r in orders) / len(orders)
    if not precision + recall:
        return 0.0
    beta2 = CHRF_BETA ** 2
    return 100 * (1 + beta2) * precision * recall / (beta2 * precision + recall)


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV at 16000 Hz")
        if not 0 < wav.getnframes() <= 60 * 16000:
            raise ValueError(f"{path}: clip duration must be positive and at most 60 seconds")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must be a nonempty JSON list")
    for case in cases:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k].strip() for k in ("audio", "ja")):
            raise ValueError("Each case requires nonempty audio and ja strings")
        for key in ("en", "title", "channel", "description", "group"):
            if key in case and not isinstance(case[key], str):
                raise ValueError(f"Case {key} must be a string")
        if "speech_start_s" in case and not isinstance(case["speech_start_s"], (int, float)):
            raise ValueError("Case speech_start_s must be a number")
        names = case.get("names", {})
        if not isinstance(names, dict) or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
                                              or not v.strip() for k, v in names.items()):
            raise ValueError("Case names must map nonempty Japanese names to English names")
        if any(name not in case["ja"] for name in names):
            raise ValueError("Expected Japanese names must occur in the reference")
    return cases


def summarize(rows: list[dict]) -> dict:
    expected = sum(row["reference_chars"] for row in rows)
    names = sum(row["name_count"] for row in rows)
    duration = sum(row["duration_s"] for row in rows)
    subtitle_latencies = [row["subtitle_ready_s"] for row in rows if row.get("subtitle_ready_s") is not None]
    onset_errors = [row["speech_to_subtitle_s"] for row in rows if row.get("speech_to_subtitle_s") is not None]
    return {
        "cases": len(rows),
        "cer": sum(row["char_errors"] for row in rows) / expected if expected else None,
        "asr_name_recall": sum(row["asr_names_correct"] for row in rows) / names if names else None,
        "mt_name_recall": sum(row["mt_names_correct"] for row in rows) / names if names else None,
        "asr_rtf": sum(row["asr_s"] for row in rows) / duration,
        "mt_rtf": sum(row["mt_s"] for row in rows) / duration,
        "asr_p95_s": float(np.percentile([row["asr_s"] for row in rows], 95)),
        "mt_p95_s": float(np.percentile([row["mt_s"] for row in rows], 95)),
        "subtitle_ready_p95_s": float(np.percentile(subtitle_latencies, 95)) if subtitle_latencies else None,
        "speech_to_subtitle_p95_s": float(np.percentile(onset_errors, 95)) if onset_errors else None,
        "chrf": chrf([(row["reference_en"], row["en"]) for row in rows if row.get("reference_en")]),
        "english_review": "Human review required; chrF and name recall only approximate translation accuracy.",
    }


def translation_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.context_lines is not None:
        overrides["context_lines"] = args.context_lines
    if args.max_context_chars is not None:
        overrides["max_context_chars"] = args.max_context_chars
    if args.context_english:
        overrides["context_english"] = True
    if args.no_context:
        overrides["context_lines"] = 0
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, help="opt in to saving transcripts and results as JSON")
    parser.add_argument("--context-lines", type=int, help="override translate.context_lines")
    parser.add_argument("--max-context-chars", type=int, help="override translate.max_context_chars")
    parser.add_argument("--context-english", action="store_true", help="include earlier English in context")
    parser.add_argument("--no-context", action="store_true",
                        help="translate each cue without dialogue history or stream metadata")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate manifest and audio files without loading models or running inference")
    args = parser.parse_args()
    cases = load_cases(args.manifest)
    base = args.manifest.parent
    for case in cases:
        pcm = read_audio(base / case["audio"])
        case["_duration_s"] = len(pcm) / 16000
    if args.validate_only:
        report = {"cases": len(cases),
                  "manifest": str(args.manifest),
                  "validated": True,
                  "audio_clips": [(c["audio"], c["_duration_s"]) for c in cases],
                  "note": "Validation only; real ASR/MT results require --config and model files."}
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return
    audio = [read_audio(base / case["audio"]) for case in cases]
    cfg = load_config(args.config)
    cfg = replace(cfg, translate=replace(cfg.translate, **translation_overrides(args)))
    glossary = Glossary.load(Path(cfg.translate.glossary))
    log = logging.getLogger("benchmark")
    logging.basicConfig(level=logging.INFO)
    rows = []
    with ExitStack() as resources:
        budget = GpuBudget(cfg.gpu_budget)
        resources.callback(budget.close)
        asr = create_asr(cfg.asr, glossary)
        resources.callback(asr.close)
        budget.refresh()
        translator = _load_translator(cfg, glossary, None, budget, log)
        resources.callback(translator.close)
        history: deque[tuple[str, str]] = deque(maxlen=max(1, cfg.translate.context_lines))
        previous_group = None
        for case, pcm in zip(cases, audio):
            group = case.get("group")
            if group is None or group != previous_group:
                history.clear()
                asr.forget()
            previous_group = group
            stream = describe_stream({key: case.get(key, "") for key in ("title", "channel", "description")},
                                     glossary)
            asr.set_context(asr_context(cfg.asr, glossary, stream.recognition_text()))
            translator.stream = StreamContext() if args.no_context else stream
            started = time.perf_counter()
            ja = asr.transcribe(pcm)
            asr_s = time.perf_counter() - started
            if ja:
                asr.remember(ja)
            started = time.perf_counter()
            en = clean_english_output(translator.translate(ja, list(history))) if ja else ""
            mt_s = time.perf_counter() - started if ja else 0.0
            if ja and useful_context(ja):
                history.append((ja, en))
            ref = normalized(case["ja"])
            hypothesis = normalized(ja)
            names = case.get("names", {})
            errors = edit_distance(ref, hypothesis)
            reference_en = case.get("en", "")
            subtitle_ready_s = asr_s + mt_s
            speech_to_subtitle_s = None
            if ja and "speech_start_s" in case:
                speech_to_subtitle_s = subtitle_ready_s - case["speech_start_s"]
            rows.append({"audio": case["audio"], "group": group, "reference_ja": case["ja"], "ja": ja,
                         "reference_en": reference_en, "en": en,
                         "duration_s": len(pcm) / 16000, "asr_s": asr_s, "mt_s": mt_s,
                         "subtitle_ready_s": subtitle_ready_s, "speech_to_subtitle_s": speech_to_subtitle_s,
                         "reference_chars": len(ref), "char_errors": errors,
                         "cer": errors / len(ref) if ref else None,
                         "chrf": chrf([(reference_en, en)]) if reference_en else None,
                         "name_count": len(names), "asr_names_correct": sum(name in ja for name in names),
                         "mt_names_correct": sum(name.casefold() in en.casefold() for name in names.values()),
                         "asr_rejection": asr.last_rejection})
        report = {"models": {"asr": cfg.asr.haas_model if cfg.asr.backend == "dual" else cfg.asr.model_path,
                             "secondary_asr": cfg.asr.secondary_model if cfg.asr.backend == "dual" else None,
                             "mt": cfg.translate.model_path,
                             "mt_gpu_layers": translator.gpu_layers},
                  "settings": {"context_lines": cfg.translate.context_lines,
                               "max_context_chars": cfg.translate.max_context_chars,
                               "context_english": cfg.translate.context_english,
                               "stream_metadata": not args.no_context},
                  "summary": summarize(rows), "results": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
