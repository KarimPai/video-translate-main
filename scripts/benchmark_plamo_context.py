"""Compare compact, OverlaySubs-style and hybrid prompts on reviewed utterances.

Input CSV columns: reference_ja, reference_en; optional history_json (a JSON list
of preceding [Japanese, reviewed English] pairs) and video_title. Every profile
gets identical source/history. Prompts are exported without loading a model by
default. --run-model requires an already-running LOCAL PLaMo server: it never
downloads/starts a model or calls an external inference service.

Examples:
  python scripts/benchmark_plamo_context.py reviewed.csv > prompts.jsonl
  python scripts/benchmark_plamo_context.py reviewed.csv --run-model > results.jsonl

Model results include actual elapsed processing times and hard output flags,
not a semantic score. Have a reviewer compare fidelity/names/nouns/viewpoint,
negation, quantities, naturalness, omissions and additions before choosing a
profile. Compact remains the default until reviewed model evidence supports a
change. GPU memory must be measured separately on the target hardware.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.subtitle_clean import english_output_issue
from auto_translate.translate.plamo import PlamoTranslator, _server_state

PROFILES = ("compact", "overlay", "hybrid")


class ExistingLocalTranslator(PlamoTranslator):
    def _start_server(self):
        if _server_state(self.base_url, self.cfg.plamo_model) != "ready":
            raise RuntimeError("Start the pinned local PLaMo server before model benchmarking")


class CaptureClient:
    def __init__(self):
        self.requests = []

    def post(self, url, *, json, timeout):
        self.requests.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": "Preview only."}, "finish_reason": "stop"}],
        })


def preview_translator(cfg, glossary):
    obj = object.__new__(PlamoTranslator)
    obj.cfg, obj.glossary = cfg, glossary
    obj.memory_lines = cfg.memory_lines
    obj.vod_title = ""
    obj.validation_stats = Counter()
    obj.client = CaptureClient()
    obj.url = cfg.plamo_base_url.rstrip("/") + "/chat/completions"
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--glossary", type=Path, default=ROOT / "glossary.toml")
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in ("http", "https") or endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
        parser.error("--base-url must point to the existing local PLaMo server")
    with args.csv_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"reference_ja", "reference_en"}.issubset(rows[0]):
        parser.error("nonempty CSV must include reference_ja and reference_en")
    for index, row in enumerate(rows, 1):
        try:
            history = json.loads(row.get("history_json") or "[]")
        except json.JSONDecodeError as exc:
            parser.error(f"row {index}: invalid history_json: {exc}")
        if not isinstance(history, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(text, str) for text in pair)
            for pair in history
        ):
            parser.error(f"row {index}: history_json must be a list of [Japanese, English] pairs")
        row["history"] = [tuple(pair) for pair in history]
    cfg = TranslateConfig(plamo_base_url=args.base_url)
    glossary = Glossary.load(args.glossary)
    translator = (ExistingLocalTranslator(cfg, glossary) if args.run_model else
                  preview_translator(cfg, glossary))
    timings = {profile: [] for profile in PROFILES}
    try:
        for index, row in enumerate(rows, 1):
            for profile in PROFILES:
                cfg.context_profile = profile
                translator.vod_title = " ".join((row.get("video_title") or "").split())[:200]
                result = {"row": index, "profile": profile, "source_ja": row["reference_ja"],
                          "reference_en": row["reference_en"]}
                if args.run_model:
                    started = time.monotonic()
                    output = translator.translate(row["reference_ja"], row["history"])
                    elapsed = time.monotonic() - started
                    timings[profile].append(elapsed)
                    result.update(output_en=output, elapsed_s=round(elapsed, 4),
                                  output_issue=english_output_issue(output),
                                  diagnostics=translator.last_name_diagnostics)
                else:
                    translator._chat(row["reference_ja"], row["history"], timeout=cfg.timeout_s)
                    result["request"] = translator.client.requests.pop()
                    result["model_run"] = False
                print(json.dumps(result, ensure_ascii=False))
        print(json.dumps({"summary": {
            "model_run": args.run_model, "utterances_per_profile": len(rows),
            "median_processing_s": {p: round(statistics.median(v), 4) if v else None for p, v in timings.items()},
            "semantic_accuracy": "requires blinded human review of actual model outputs",
            "vram": "not measured", "default_profile_changed": False,
        }}, ensure_ascii=False))
    finally:
        if args.run_model:
            translator.close()


if __name__ == "__main__":
    main()
