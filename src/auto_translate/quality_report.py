"""Offline, dependency-light quality and performance reports.

Only annotated references establish Japanese accuracy. Comparison logs provide
observations, never ground truth. Missing measurements stay explicitly absent.
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def normalize(text: str) -> str:
    """NFKC and whitespace removal; preserve punctuation and Japanese script."""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def _number(value):
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _first_number(row, *keys):
    return next((v for key in keys if (v := _number(row.get(key))) is not None), None)


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.casefold() in ("true", "false", "1", "0"):
        return value.casefold() in ("true", "1")
    return None


def _tags(row):
    value = row.get("tags") or []
    if isinstance(value, str):
        value = re.split(r"[,;|]", value)
    return sorted({str(tag).strip().casefold() for tag in value if str(tag).strip()})


def _percentile(values, proportion):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values, reason):
    return {"available": bool(values), "samples": len(values),
            "p50": _percentile(values, .5), "p95": _percentile(values, .95),
            "maximum": max(values, default=None),
            "unavailable_reason": None if values else reason}


def evaluate(rows: list[dict], *, use_jiwer: bool = False) -> dict:
    """Retain the original benchmark's corpus CER, with annotation coverage."""
    references = []
    outputs = []
    reference_chars = char_errors = missed = spurious = flags = error_cues = 0
    from .translation_guard import suspicious_translation
    for row in rows:
        out = normalize(row.get("output_ja") or "")
        if out and row.get("output_en"):
            flags += bool(suspicious_translation(out, row["output_en"]))
        if "reference_ja" not in row or row["reference_ja"] is None:
            continue
        ref = normalize(row["reference_ja"])
        references.append(ref)
        outputs.append(out)
        edits = distance(ref, out)
        reference_chars += len(ref)
        char_errors += edits
        error_cues += bool(edits)
        missed += bool(ref) and not bool(out)
        spurious += not bool(ref) and bool(out)
    result = {
        "aligned_utterances": len(rows), "annotated_utterances": len(references),
        "reference_characters": reference_chars, "character_errors": char_errors,
        "japanese_character_error_rate": (round(char_errors / reference_chars, 6)
                                          if reference_chars else None),
        "japanese_error_cues": error_cues,
        "missing_japanese_cues": missed, "spurious_japanese_cues": spurious,
        "english_guard_flags_for_review": flags,
        "english_semantic_accuracy": "requires human assessment",
        "cer_backend": "stdlib Levenshtein",
        "unavailable_reason": (None if reference_chars else
                               "No nonempty, manually annotated Japanese references"),
    }
    if use_jiwer:
        try:
            from jiwer import process_characters
        except ImportError:
            result["jiwer"] = {"available": False, "reason": "Optional jiwer is not installed"}
        else:
            # Both implementations receive the same explicitly normalized text.
            try:
                score = process_characters(references, outputs)
                result["jiwer"] = {"available": True, "cer": score.cer}
            except ValueError as exc:
                result["jiwer"] = {"available": False, "reason": str(exc)}
    return result


def _duration(row):
    value = _first_number(row, "audio_duration_s", "duration_s")
    if value is not None:
        return value
    start, end = _number(row.get("start_s")), _number(row.get("end_s"))
    return end - start if start is not None and end is not None and end > start else None


def _latency(row):
    value = _first_number(row, "cue_latency_s", "latency_s")
    if value is not None:
        return value
    timings = row.get("timings")
    if isinstance(timings, dict):
        read = _number(timings.get("audio_read"))
        end = _first_number(timings, "output_at", "mt_end")
        if read is not None and end is not None and read > 0 and end >= read:
            return end - read
    return None


def _rate(values, reason):
    return {"available": bool(values), "observed_cues": len(values),
            "count": sum(values), "rate": sum(values) / len(values) if values else None,
            "unavailable_reason": None if values else reason}


def _fallback(row):
    explicit = _boolean(row.get("fallback"))
    if explicit is not None:
        return explicit
    if "fallback_reason" in row:
        return bool(row.get("fallback_reason"))
    return True if row.get("stage") == "asr_fallback" else None


def _timeout(row):
    explicit = _boolean(row.get("timeout"))
    if explicit is not None:
        return explicit
    reason = str(row.get("fallback_reason") or row.get("error_type") or "")
    if re.search(r"timeout|timed[ _]out", reason, re.I):
        return True
    # A completed reconciliation establishes no reported timeout for that
    # optional selection. Other fallbacks such as backlog are not timeouts.
    if row.get("stage") == "reconciliation" or "fallback_reason" in row:
        return False
    return None


def build_report(rows: list[dict], *, source: str | None = None,
                 hardware_label: str | None = None, use_jiwer: bool = False) -> dict:
    """Summarize aligned rows. Unit-bearing fields are documented in PERFORMANCE.md."""
    groups = defaultdict(list)
    for row in rows:
        for tag in _tags(row):
            groups[tag].append(row)
    quality = evaluate(rows, use_jiwer=use_jiwer)
    per_tag = {tag: evaluate(items, use_jiwer=use_jiwer)
               for tag, items in sorted(groups.items())}
    error_groups = {}
    for label, aliases in {"names": {"name", "names", "proper_names"},
                           "numbers": {"number", "numbers", "numeric"},
                           "negation": {"negation", "negative"}}.items():
        subset = [row for row in rows if aliases.intersection(_tags(row))]
        reviewed = [v for row in subset
                    if (v := _boolean(row.get(label + "_error"))) is not None]
        error_groups[label] = {
            "tagged_cues": len(subset), "japanese": evaluate(subset, use_jiwer=use_jiwer),
            "human_reviewed_errors": _rate(reviewed, "No explicit human review labels"),
            "note": "Japanese text errors on tagged cues; not an English semantic score",
        }
    latencies = [value for row in rows if (value := _latency(row)) is not None]
    elapsed_duration = [(elapsed, duration) for row in rows
                        if (elapsed := _first_number(row, "elapsed_s", "processing_elapsed_s")) is not None
                        and (duration := _duration(row)) is not None and duration > 0]
    rtfs = [elapsed / duration for elapsed, duration in elapsed_duration]
    gpu = [value for row in rows
           if (value := _first_number(row, "sampled_gpu_peak_gb", "gpu_used_gb")) is not None]
    free = [value for row in rows if (value := _number(row.get("gpu_free_gb"))) is not None]
    fallback = [v for row in rows if (v := _fallback(row)) is not None]
    timeouts = [v for row in rows if (v := _timeout(row)) is not None]
    metrics = {
        "cue_latency_s": _distribution(latencies, "No cue-end arrival/output timing measurements"),
        "real_time_factor": dict(_distribution(rtfs, "No paired elapsed_s and audio_duration_s"),
                                 aggregate=(sum(e for e, _ in elapsed_duration) /
                                            sum(d for _, d in elapsed_duration)
                                            if elapsed_duration else None)),
        "sampled_device_vram_gb": _distribution(gpu, "No device-wide GPU memory samples"),
        "sampled_free_vram_gb": dict(_distribution(free, "No GPU free-memory samples"),
                                     minimum=min(free, default=None)),
        "timeout_rate": _rate(timeouts, "No explicit timeout outcomes"),
        "fallback_rate": _rate(fallback, "No explicit optional-ASR fallback outcomes"),
    }
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "source": source, "hardware_label": hardware_label,
              "rows": len(rows), "quality": quality, "per_tag": per_tag,
              "error_groups": error_groups, "metrics": metrics,
              "limitations": [
                  "Logs without manual references cannot measure recognition accuracy.",
                  "English guard flags are review signals; English semantics require human review.",
                  "VRAM is sampled device-wide usage, including other applications; peaks between samples may be missed.",
                  "RTF describes only the supplied elapsed measurement scope; ASR-pair benchmarks exclude translation/playback.",
                  "Rates cover only cues with recorded outcomes; unavailable outcomes are not counted as successes.",
              ]}
    report["recommendations"] = recommend(report)
    return report


def recommend(report: dict) -> list[str]:
    """Conservative suggestions triggered only by supplied measurements."""
    metrics = report["metrics"]
    suggestions = []
    free = metrics["sampled_free_vram_gb"]["minimum"]
    if free is not None and free < .55:
        suggestions.append("Measured free GPU memory fell below 0.55 GB: compare a smaller beam, CPU selector, and reduced PLaMo offload on the same clips before saving a hardware preset.")
    rtf = metrics["real_time_factor"]["aggregate"]
    if rtf is not None and rtf > 1:
        suggestions.append("Measured processing RTF exceeds 1: this measured setup cannot sustain real time at that scope. Compare beam 1/3/6 and selective secondary ASR.")
    timeout = metrics["timeout_rate"]["rate"]
    fallback = metrics["fallback_rate"]["rate"]
    if timeout is not None and timeout > 0:
        suggestions.append("Timeouts were recorded: inspect per-cue reasons and benchmark lower-cost settings before increasing deadlines.")
    if fallback is not None and fallback > 0:
        suggestions.append("Optional-ASR fallbacks were recorded: examine reasons before treating this run as a full dual-ASR accuracy comparison.")
    if not report["quality"]["reference_characters"]:
        suggestions.append("Add time-aligned manual Japanese references and names/numbers/negation tags before comparing accuracy or choosing a quality preset.")
    if not suggestions:
        suggestions.append("No measured issue triggered a setting change. Compare repeated runs on identical annotated clips before saving this hardware preset.")
    if not metrics["sampled_device_vram_gb"]["available"]:
        suggestions.append("GPU memory measurements are unavailable; no GPU-capacity recommendation can be made.")
    return suggestions


def recommend_presets(reports: list[dict]) -> list[str]:
    """Rank only complete measurements of the same workload and scope."""
    complete = [report for report in reports if report.get("complete")
                and report["quality"]["japanese_character_error_rate"] is not None
                and report["metrics"]["real_time_factor"]["aggregate"] is not None]
    if not complete:
        return ["No complete annotated timing runs: no measured hardware preset can be recommended."]
    first = complete[0]
    if not first.get("workload") or not first.get("hardware") or not first.get("measurement_scope"):
        return ["Workload fingerprints, hardware, and measurement scope are required before comparing presets."]
    comparable = [report for report in complete
                  if report.get("workload") == first.get("workload")
                  and report.get("measurement_scope") == first.get("measurement_scope")
                  and report.get("hardware") == first.get("hardware")]
    if len(comparable) < 2:
        return ["Fewer than two complete comparable configurations: collect matched runs before choosing a preset."]
    best = min(comparable, key=lambda report: (report["quality"]["japanese_character_error_rate"],
                                              report["metrics"]["real_time_factor"]["aggregate"]))
    return [f"On the recorded hardware and clips, {json.dumps(best.get('settings', {}), sort_keys=True)} "
            f"had the lowest measured Japanese CER ({best['quality']['japanese_character_error_rate']:.2%}); "
            f"aggregate RTF was {best['metrics']['real_time_factor']['aggregate']:.3f}. "
            "This is a measured candidate, not an automatic configuration change. Validate live playback separately."]


def _cue_key(row):
    if "start_s" not in row or "end_s" not in row:
        return None
    return (str(row.get("session_id") or ""), str(row.get("generation", 0)),
            round(float(row["start_s"]), 3), round(float(row["end_s"]), 3))


def rows_from_comparisons(records: list[dict]) -> list[dict]:
    """Merge pipeline stages once per cue; retain session/seek boundaries."""
    grouped = {}
    standalone = []
    for record in records:
        try:
            key = _cue_key(record)
        except (TypeError, ValueError, OverflowError):
            continue
        if key is None:
            if "reference_ja" in record or "output_ja" in record:
                standalone.append(dict(record))
            continue
        row = grouped.setdefault(key, {})
        old_reason = row.get("fallback_reason")
        old_timeout = _timeout(row)
        row.update(record)
        if old_reason and not row.get("fallback_reason"):
            row["fallback_reason"] = old_reason
        if old_timeout:
            row["timeout"] = True
        # A late fallback callback must not overwrite an already emitted cue.
        for priority, field in ((4, "final_ja"), (3, "after_term_repair"), (2, "chosen"), (1, "haas")):
            if record.get(field) is not None and priority >= row.get("_output_priority", 0):
                row["output_ja"] = record[field]
                row["_output_priority"] = priority
                break
        if "final_en" in record:
            row["output_en"] = record["final_en"]
    for row in grouped.values():
        row.pop("_output_priority", None)
    return [*grouped.values(), *standalone]


def load_rows(path: Path) -> tuple[list[dict], int]:
    """Read paired CSV, benchmark JSON, or comparison JSONL; count bad log lines."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle)), 0
    if path.suffix.casefold() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data if isinstance(data, list) else data.get("clips", data.get("rows", []))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("JSON report input must contain a list of clip/cue rows")
        return rows, 0
    records, bad_lines = [], 0
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON object required")
                records.append(record)
            except (ValueError, TypeError):
                bad_lines += 1
    return rows_from_comparisons(records), bad_lines


def _attach_annotations(rows, path):
    """Join only explicit cue IDs; never align unrelated transcripts by order."""
    annotations, _ = load_rows(path)
    by_key = {}
    for annotation in annotations:
        try:
            key = _cue_key(annotation)
        except (TypeError, ValueError, OverflowError):
            continue
        if key is None:
            continue
        if key in by_key:
            raise ValueError("Duplicate annotation cue key; include session_id and generation")
        by_key[key] = annotation
    matched = 0
    for row in rows:
        annotation = by_key.get(_cue_key(row))
        if annotation is not None:
            row.update({key: value for key, value in annotation.items()
                        if key.startswith("reference_") or key == "tags" or key.endswith("_error")})
            matched += 1
    return matched


def write_report(report: dict, output_dir: Path, *, stem: str = "quality-report") -> dict:
    """Write standalone UTF-8 JSON and safe HTML; return paths for GUI callers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
    json_path, html_path = output_dir / f"{stem}.json", output_dir / f"{stem}.html"
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(payload, encoding="utf-8")
    cards = []
    for name, metric in report["metrics"].items():
        if metric["available"]:
            value = (f"{metric['rate']:.1%} ({metric['count']}/{metric['observed_cues']})"
                     if "rate" in metric else
                     f"p50 {metric['p50']:.3f} · p95 {metric['p95']:.3f} · n={metric['samples']}")
        else:
            value = "Unavailable — " + metric["unavailable_reason"]
        cards.append(f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>")
    tag_rows = []
    for tag, score in report["per_tag"].items():
        cer = score["japanese_character_error_rate"]
        tag_rows.append(f"<tr><th>{html.escape(tag)}</th><td>{'Unavailable' if cer is None else f'{cer:.2%}'}</td>"
                        f"<td>{score['annotated_utterances']}</td></tr>")
    content = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
               "<meta name='viewport' content='width=device-width,initial-scale=1'>"
               "<title>Subtitle quality report</title><style>"
               "body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;line-height:1.5}"
               "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border:1px solid #bbb;text-align:left}"
               "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:1rem}"
               "</style><h1>Subtitle quality report</h1>"
               f"<p>{html.escape(report.get('source') or 'Paired benchmark data')} · "
               f"{html.escape(report.get('hardware_label') or 'Hardware not recorded')}</p>"
               "<p>CER requires manual Japanese references. English semantics require human review.</p>"
               "<h2>Measured performance</h2><table>" + "".join(cards) + "</table>"
               "<h2>Japanese CER by tag</h2><table><tr><th>Tag</th><th>CER</th><th>Annotated cues</th></tr>"
               + ("".join(tag_rows) or "<tr><td colspan='3'>No annotated tags supplied</td></tr>")
               + "</table><h2>Recommendations from this data</h2><ul>"
               + "".join(f"<li>{html.escape(item)}</li>" for item in report["recommendations"])
               + "</ul><h2>Full report and limitations</h2><pre>" + html.escape(payload) + "</pre></html>")
    html_path.write_text(content, encoding="utf-8")
    return {"json_path": json_path.resolve(), "html_path": html_path.resolve(), "report": report}


def generate_report(input_path: Path, output_dir: Path | None = None, *,
                    annotations_path: Path | None = None, hardware_label: str | None = None,
                    use_jiwer: bool = False) -> dict:
    """GUI/CLI API: generate JSON and HTML without loading any inference model."""
    input_path = Path(input_path)
    rows, bad_lines = load_rows(input_path)
    matched = _attach_annotations(rows, Path(annotations_path)) if annotations_path else 0
    report = build_report(rows, source=str(input_path), hardware_label=hardware_label,
                          use_jiwer=use_jiwer)
    report["invalid_log_lines"] = bad_lines
    report["attached_annotations"] = matched
    return write_report(report, output_dir or input_path.parent / "reports")
