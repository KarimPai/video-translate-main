"""Optional offline Japanese→English A/B: run Index and Qwen on identical JP lines.

Run each server separately to fit an 8 GB GPU. This script NEVER launches or
downloads a model and does not change the pipeline's sole Index translator.
Results are evidence for human inspection; this is not an automatic quality score.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_translate.glossary import Glossary
from auto_translate.translate.index import _clean, build_index_prompt


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def request_json(url: str, payload: dict | None = None, *, timeout: float = 20) -> dict:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> None:
    base = args.url.rstrip("/")
    models = request_json(base + "/models", timeout=4).get("data", [])
    if not models:
        raise SystemExit("No active local llama.cpp model detected at " + base)
    model = args.model or str(models[0]["id"])
    dataset = read_jsonl(args.dataset)
    glossary = Glossary.load(ROOT / "glossary.toml")
    previous: list[tuple[str, str]] = []
    results: list[dict] = []
    for index, case in enumerate(dataset, 1):
        ja = str(case["ja"]).strip()
        if args.backend == "index":
            prompt = build_index_prompt(
                ja, previous, glossary, context_lines=2,
            )
        else:
            history = " / ".join(j for j, _ in previous[-2:])
            prompt = (
                "Translate only the current Japanese into natural casual English "
                "gaming-stream subtitles. Keep negations, intent, slang, names "
                "and unfinished fragments. Do not repeat the reference dialogue.\n"
                + ("Previous Japanese (reference only): " + history + "\n" if history else "")
                + "Current Japanese to translate:\n" + ja
            )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 160,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if args.backend == "index":
            payload["stop"] = ["〖", "【约束", "Constraint Requirements", "Hard Requirement"]
        start = time.monotonic()
        english = ""
        error = ""
        try:
            response = request_json(base + "/chat/completions", payload, timeout=args.timeout)
            english = _clean(str(response["choices"][0]["message"].get("content") or ""))
        except (OSError, KeyError, IndexError, ValueError) as exc:
            error = str(exc)
        elapsed = time.monotonic() - start
        results.append({
            "id": case.get("id", index),
            "ja": ja,
            "reference": case.get("reference", ""),
            "en": english,
            "seconds": round(elapsed, 3),
            "error": error,
        })
        print(f"[{index}/{len(dataset)}] {ja} -> {english or 'ERROR: ' + error} ({elapsed:.2f}s)",
              flush=True)
        # Reference dialogue stays Japanese even if the model output is wrong.
        previous.append((ja, english))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved:", args.output)


def compare(args: argparse.Namespace) -> None:
    left, right = read_jsonl(args.index), read_jsonl(args.qwen)
    if [item["ja"] for item in left] != [item["ja"] for item in right]:
        raise SystemExit("Input Japanese differs. Both models must use the SAME dataset.")
    for a, b in zip(left, right):
        print("\nJP:", a["ja"])
        if a.get("reference"):
            print("Reference:", a["reference"])
        print("Index:", a.get("en") or a.get("error"))
        print("Qwen: ", b.get("en") or b.get("error"))
    for name, data in (("Index", left), ("Qwen", right)):
        durations = [row["seconds"] for row in data if row.get("en")]
        count = len(durations)
        print(f"{name}: {count}/{len(data)} successful; "
              f"mean {statistics.mean(durations):.3f}s, "
              f"median {statistics.median(durations):.3f}s" if durations
              else f"{name}: no successful translations")
    print("Human review required: compare meaning, names, negation and naturalness.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    run_cmd = commands.add_parser("run", help="translate identical saved Japanese using one running server")
    run_cmd.add_argument("--backend", required=True, choices=("index", "qwen"))
    run_cmd.add_argument("--url", required=True, help="already-running llama.cpp base URL e.g. http://127.0.0.1:8002/v1")
    run_cmd.add_argument("--model", default="", help="optional server-reported model alias")
    run_cmd.add_argument("--dataset", type=Path,
                         default=ROOT / "tests/data/gta_translation_eval.jsonl")
    run_cmd.add_argument("--output", type=Path, required=True)
    run_cmd.add_argument("--timeout", type=float, default=25.0)
    cmp_cmd = commands.add_parser("compare", help="compare two saved JSONL outputs")
    cmp_cmd.add_argument("--index", type=Path, required=True)
    cmp_cmd.add_argument("--qwen", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
