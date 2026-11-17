"""Evaluate reviewed mono 16 kHz WAV clips with the configured local models."""
import argparse
from contextlib import ExitStack
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
from auto_translate.subtitle_clean import clean_english_output


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


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
        for key in ("en", "title"):
            if key in case and not isinstance(case[key], str):
                raise ValueError(f"Case {key} must be a string")
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
    return {
        "cases": len(rows),
        "cer": sum(row["char_errors"] for row in rows) / expected if expected else None,
        "asr_name_recall": sum(row["asr_names_correct"] for row in rows) / names if names else None,
        "mt_name_recall": sum(row["mt_names_correct"] for row in rows) / names if names else None,
        "asr_rtf": sum(row["asr_s"] for row in rows) / duration,
        "mt_rtf": sum(row["mt_s"] for row in rows) / duration,
        "asr_p95_s": float(np.percentile([row["asr_s"] for row in rows], 95)),
        "mt_p95_s": float(np.percentile([row["mt_s"] for row in rows], 95)),
        "english_review": "Human review required; name recall is not translation accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, help="opt in to saving transcripts and results as JSON")
    args = parser.parse_args()
    cases = load_cases(args.manifest)
    audio = [read_audio(args.manifest.parent / case["audio"]) for case in cases]
    cfg = load_config(args.config)
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
        translator = _load_translator(cfg, glossary, "", budget, log)
        resources.callback(translator.close)
        for case, pcm in zip(cases, audio):
            title = case.get("title", "")
            asr.set_context(asr_context(cfg.asr, glossary, title))
            translator.title = title.strip()[:200]
            started = time.perf_counter()
            ja = asr.transcribe(pcm)
            asr_s = time.perf_counter() - started
            started = time.perf_counter()
            en = clean_english_output(translator.translate(ja, [])) if ja else ""
            mt_s = time.perf_counter() - started if ja else 0.0
            ref = normalized(case["ja"])
            hypothesis = normalized(ja)
            names = case.get("names", {})
            errors = edit_distance(ref, hypothesis)
            rows.append({"audio": case["audio"], "reference_ja": case["ja"], "ja": ja,
                         "reference_en": case.get("en", ""), "en": en,
                         "duration_s": len(pcm) / 16000, "asr_s": asr_s, "mt_s": mt_s,
                         "reference_chars": len(ref), "char_errors": errors,
                         "cer": errors / len(ref) if ref else None,
                         "name_count": len(names), "asr_names_correct": sum(name in ja for name in names),
                         "mt_names_correct": sum(name.casefold() in en.casefold() for name in names.values()),
                         "asr_rejection": asr.last_rejection})
        report = {"models": {"asr": cfg.asr.model_path, "mt": cfg.translate.model_path,
                             "mt_gpu_layers": translator.gpu_layers}, "summary": summarize(rows), "results": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
