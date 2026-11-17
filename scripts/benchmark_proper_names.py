"""Reproduce text-only name regressions or run explicitly reviewed local audio.

Text mode loads the legacy matchers from an immutable Git commit, without a
second checkout. It measures deterministic resolution and span repair only.
Audio mode runs workers sequentially; it never manufactures missing results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import time
import tomllib
import types
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASELINE = "fd3aae8c9b174f5cbaceed2666cd5f873d301c80"
# Readings change no spoken form: a short Matsuri can never become 夏色まつり.
_READINGS = {"夏色": "なついろ", "桜": "さくら", "祭り": "まつり", "祭": "まつり",
             "巫女": "みこ", "様": "さま", "先輩": "せんぱい", "君": "くん"}


def _legacy(ref):
    """Use the actual legacy functions, rather than reimplementing a baseline."""
    git = ["git"]
    if (ROOT / "work/git/HEAD").is_file():
        git.append("--git-dir=" + str(ROOT / "work/git"))

    def source(path):
        return subprocess.check_output(git + ["show", f"{ref}:{path}"], cwd=ROOT,
                                       text=True, encoding="utf-8")

    hints = types.ModuleType("legacy_name_hints")
    exec(compile(source("src/auto_translate/name_hints.py"), "legacy_name_hints", "exec"),
         hints.__dict__)
    glossary_module = types.ModuleType("legacy_glossary")
    exec(compile(source("src/auto_translate/glossary.py"), "legacy_glossary", "exec"),
         glossary_module.__dict__)
    data = tomllib.loads(source("glossary.toml"))
    glossary = glossary_module.Glossary(data.get("terms", {}), nicknames=data.get("nicknames"))
    parsed = ast.parse(source("src/auto_translate/translate/plamo.py"))
    selected = []
    constants = {"_HOLOGTA_TITLE", "_HOLOGTA_TERMS"}
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_current_glossary_matches":
            selected.append(node)
    namespace = {"re": re, "Glossary": glossary_module.Glossary,
                 "matsuri_matches": hints.matsuri_matches}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "legacy_plamo_matcher", "exec"),
         namespace)
    return glossary, namespace["_current_glossary_matches"], hints.preserve_spoken_names


def _expected_spans(row):
    result = []
    for item in row["expected"]:
        source = item["source"]
        starts = [match.start() for match in re.finditer(re.escape(source), row["ja"])]
        ordinal = item.get("occurrence", 0)
        if not source or ordinal >= len(starts):
            raise ValueError(f"Invalid reference span in {row['id']}")
        start = starts[ordinal]
        result.append(dict(item, start=start, end=start + len(source)))
    return result


def _legacy_mentions(ja, pairs):
    occupied = set()
    mentions = []
    for source, target in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(re.escape(source), ja):
            positions = set(range(match.start(), match.end()))
            if positions & occupied:
                continue
            occupied.update(positions)
            mentions.append(dict(start=match.start(), end=match.end(), source=source,
                                 target=target, classification="PERSON"))
    return mentions


def _summarize(rows):
    totals = dict(person_occurrences=0, names_preserved=0, negative_slots=0,
                  false_positive_names=0, literal_noun_occurrences=0,
                  literal_nouns_preserved=0, exact_decisions=0, decisions=0,
                  deterministic_repair_cases=0, deterministic_repairs_correct=0,
                  abstained_occurrences=0)
    for row in rows:
        predicted = {(item["start"], item["end"]): item for item in row["mentions"]}
        expected = _expected_spans(row)
        expected_person = {(item["start"], item["end"]) for item in expected
                           if item["classification"] == "PERSON"}
        for item in expected:
            decision = predicted.get((item["start"], item["end"]), {})
            actual = decision.get("classification", "UNKNOWN")
            totals["decisions"] += 1
            totals["exact_decisions"] += actual == item["classification"]
            totals["abstained_occurrences"] += actual == "UNKNOWN"
            if item["classification"] == "PERSON":
                totals["person_occurrences"] += 1
                totals["names_preserved"] += (actual == "PERSON"
                                              and decision.get("target") == item["target"])
            else:
                totals["negative_slots"] += 1
                if item["classification"] == "COMMON_NOUN":
                    totals["literal_noun_occurrences"] += 1
                    totals["literal_nouns_preserved"] += actual != "PERSON"
        if not expected:
            totals["negative_slots"] += 1
        totals["false_positive_names"] += sum(
            item["classification"] == "PERSON" and (item["start"], item["end"]) not in expected_person
            for item in row["mentions"])
        if "expected_english" in row:
            totals["deterministic_repair_cases"] += 1
            totals["deterministic_repairs_correct"] += row["output_english"] == row["expected_english"]

    def ratio(numerator, denominator):
        return round(totals[numerator] / totals[denominator], 6) if totals[denominator] else None

    return dict(totals,
                name_recall=ratio("names_preserved", "person_occurrences"),
                false_positive_rate=ratio("false_positive_names", "negative_slots"),
                literal_noun_preservation=ratio("literal_nouns_preserved", "literal_noun_occurrences"),
                deterministic_repair_exact_match=ratio("deterministic_repairs_correct", "deterministic_repair_cases"))


def text_benchmark(fixtures, baseline):
    from auto_translate.name_hints import preserve_spoken_names
    from auto_translate.proper_names import name_glossary, resolve_names

    data = json.loads(fixtures.read_text(encoding="utf-8"))
    old_glossary, old_matches, old_repair = _legacy(baseline)
    reports = {"baseline": [], "new": []}
    name_targets = {item["target"] for row in data["cases"] for item in row["expected"]
                    if item["classification"] == "PERSON"}
    for original in data["cases"]:
        row = dict(original)
        args = {key: row[key] for key in ("haas", "whisper", "previous_ja", "video_title",
                                         "confirmed_participants", "classifications") if key in row}
        mentions = resolve_names(row["ja"], **args)
        new_row = dict(row, mentions=[item.as_dict() for item in mentions])
        old_pairs = old_matches(row["ja"], row.get("video_title", ""), old_glossary,
                                row.get("previous_ja", ""))
        old_row = dict(row, mentions=_legacy_mentions(
            row["ja"], [(source, target) for source, target in old_pairs if target in name_targets]))
        if "raw_english" in row:
            old_row["output_english"] = old_repair(row["ja"], row["raw_english"], old_pairs)
            new_row["output_english"] = preserve_spoken_names(
                row["ja"], row["raw_english"], name_glossary(mentions), mentions=mentions)
        reports["baseline"].append(old_row)
        reports["new"].append(new_row)
    return {"evidence": "synthetic text only; no model inference or acoustic accuracy measured",
            "baseline_commit": baseline, "fixture_version": data["version"],
            "metric_notes": {
                "name_recall": "exact expected PERSON span and spoken English form",
                "false_positive_rate": "unexpected PERSON spans / expected nonperson slots plus no-name cases",
                "literal_noun_preservation": "literal source occurrence not classified PERSON; no English semantic claim",
                "deterministic_repair_exact_match": "safe span repair against authored English expectations; no PLaMo inference",
                "english_semantic_accuracy": "requires real inference and human review"},
            "scores": {profile: _summarize(rows) for profile, rows in reports.items()},
            "cases": reports}


def _pcm(path):
    import numpy as np

    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
            raise ValueError("Audio benchmark accepts only 16 kHz mono signed 16-bit PCM WAV")
        data = stream.readframes(stream.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def audio_benchmark(args):
    """Yield actual worker observations, including unsupported/failed profiles."""
    from importlib.metadata import PackageNotFoundError, version
    from auto_translate.asr.haas import HaasASR
    from auto_translate.asr.whisper_turbo import WhisperTurboASR
    from auto_translate.config import AsrConfig
    from auto_translate.proper_names import resolve_names

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError("Manifest must contain reusable, human-reviewed clips; none supplied")
    loaded = []
    for row in clips:
        if (row.get("human_reviewed") is not True or row.get("reusable") is not True
                or not row.get("provenance") or "reference_ja" not in row):
            raise ValueError("Each clip requires human_reviewed=true, reusable=true, provenance and reference_ja")
        path = (args.manifest.resolve().parent / row["clip"]).resolve(strict=True)
        loaded.append((row, path, _pcm(path)))
    try:
        whisper_version = version("faster-whisper")
    except PackageNotFoundError:
        whisper_version = None
    base_hints = {}

    def record(row, path, profile, settings, output, elapsed, diagnostics=None, failure=None):
        reference = re.sub(r"\s", "", row["reference_ja"])
        hypothesis = re.sub(r"\s", "", output) if output is not None else None
        mentions = (resolve_names(output, previous_ja=row.get("previous_ja", ""),
                                  confirmed_participants=row.get("confirmed_participants", ()))
                    if output is not None else [])
        expected = row.get("expected", [])
        person_targets = [item["target"] for item in expected if item["classification"] == "PERSON"]
        emitted = [item.target for item in mentions if item.classification == "PERSON"]
        return dict(profile=profile, clip=row["clip"],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), provenance=row["provenance"],
                    reference_ja=row["reference_ja"], output_ja=output, settings=settings,
                    seconds=elapsed, failure=failure, diagnostics=diagnostics,
                    char_edits=_edit_distance(reference, hypothesis) if hypothesis is not None else None,
                    reference_chars=len(reference),
                    name_reference_occurrences=len(person_targets),
                    resolved_names=emitted, expected_names=person_targets,
                    names_recalled=sum(min(emitted.count(name), person_targets.count(name))
                                       for name in set(person_targets)),
                    false_positive_names=sum(max(0, emitted.count(name) - person_targets.count(name))
                                             for name in set(emitted)),
                    faster_whisper=whisper_version,
                    english_translation_correctness="not measured; ASR-only experiment")

    # One Haas worker, then one Whisper worker at a time. No extra GPU residency.
    haas = HaasASR(AsrConfig(haas_model=args.haas_model, haas_device=args.haas_device,
                           haas_keywords_enabled=True))
    try:
        for row, path, audio in loaded:
            started = time.perf_counter()
            try:
                output = haas.transcribe(audio)
            except Exception as exc:
                yield record(row, path, "haas-baseline", {}, None, time.perf_counter() - started,
                             failure=f"{type(exc).__name__}: {exc}")
                raise
            yield record(row, path, "haas-baseline", {}, output, time.perf_counter() - started,
                         getattr(haas, "last_keyword_diagnostics", {}))
            # References and title never seed hints. Only baseline-heard,
            # context-confirmed current names can be tested as vocabulary.
            mentions = resolve_names(output, previous_ja=row.get("previous_ja", ""),
                                      confirmed_participants=row.get("confirmed_participants", ()))
            hints = list(dict.fromkeys(item.source for item in mentions
                                       if item.classification == "PERSON"))[:4]
            base_hints[row["clip"]] = hints
            for alpha in (1.0, 1.5, 2.0):
                started = time.perf_counter()
                try:
                    vocabulary = []
                    for hint in hints:
                        reading = hint
                        for spelling, kana in _READINGS.items():
                            reading = reading.replace(spelling, kana)
                        vocabulary.append(f"{hint}={reading}")
                    output = haas.transcribe(audio, keywords="\n".join(vocabulary), keywords_alpha=alpha)
                    yield record(row, path, f"haas-keywords-{alpha:g}",
                                 {"alpha": alpha, "guard": True, "hints": vocabulary}, output,
                                 time.perf_counter() - started, getattr(haas, "last_keyword_diagnostics", {}))
                except Exception as exc:
                    yield record(row, path, f"haas-keywords-{alpha:g}", {"alpha": alpha}, None,
                                 time.perf_counter() - started, failure=f"{type(exc).__name__}: {exc}")
                    raise
    finally:
        haas.close()

    for beam in args.beams:
        whisper = WhisperTurboASR(model=args.whisper_model, device=args.whisper_device,
                                  compute_type=args.whisper_compute_type, beam_size=beam)
        try:
            for index, (row, path, audio) in enumerate(loaded):
                modes = ("none", "hotwords", "initial_prompt")
                if index % 2:
                    modes = tuple(reversed(modes))
                for mode in modes:
                    hints = "、".join(base_hints.get(row["clip"], ()))
                    options = ({"hotwords": hints} if mode == "hotwords" else
                               {"initial_prompt": hints} if mode == "initial_prompt" else {})
                    started = time.perf_counter()
                    try:
                        output = whisper.transcribe(audio, timeout=args.timeout, **options)
                        yield record(row, path, f"whisper-beam-{beam}-{mode}",
                                     {"beam_size": beam, "prompt_mode": mode, "hints": hints}, output,
                                     time.perf_counter() - started, getattr(whisper, "last_confidence", {}))
                    except Exception as exc:
                        yield record(row, path, f"whisper-beam-{beam}-{mode}",
                                     {"beam_size": beam, "prompt_mode": mode}, None,
                                     time.perf_counter() - started, failure=f"{type(exc).__name__}: {exc}")
                        raise
        finally:
            whisper.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    text = commands.add_parser("text", help="deterministic fixtures; no models or audio")
    text.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures/proper_name_cases.json")
    text.add_argument("--baseline", default=BASELINE)
    text.add_argument("--output", type=Path)
    audio = commands.add_parser("audio", help="actual reusable, human-reviewed local PCM clips")
    audio.add_argument("manifest", type=Path)
    audio.add_argument("output", type=Path)
    audio.add_argument("--haas-model", default="haas-ja-v1.0")
    audio.add_argument("--haas-device", default="Vulkan0")
    audio.add_argument("--whisper-model", default="h2oai/faster-whisper-large-v3-turbo")
    audio.add_argument("--whisper-device", choices=("cuda", "cpu"), default="cuda")
    audio.add_argument("--whisper-compute-type", default="int8_float16")
    audio.add_argument("--beams", type=int, choices=(6, 8, 10), nargs="+", default=[6, 8])
    audio.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.mode == "text":
        report = json.dumps(text_benchmark(args.fixtures, args.baseline), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
        else:
            print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        aggregate = {}
        with args.output.open("x", encoding="utf-8") as stream:
            for record in audio_benchmark(args):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                totals = aggregate.setdefault(record["profile"], dict(
                    attempted_clips=0, successful_clips=0, character_edits=0, reference_chars=0,
                    names_recalled=0, expected_name_occurrences=0, false_positive_names=0))
                totals["attempted_clips"] += 1
                if record["failure"] is None:
                    totals["successful_clips"] += 1
                    totals["character_edits"] += record["char_edits"]
                    totals["reference_chars"] += record["reference_chars"]
                    totals["names_recalled"] += record["names_recalled"]
                    totals["expected_name_occurrences"] += record["name_reference_occurrences"]
                    totals["false_positive_names"] += record["false_positive_names"]
            for profile, totals in aggregate.items():
                totals["character_error_rate"] = (totals["character_edits"] / totals["reference_chars"]
                                                    if totals["reference_chars"] else None)
                totals["resolved_name_recall"] = (totals["names_recalled"] / totals["expected_name_occurrences"]
                                                   if totals["expected_name_occurrences"] else None)
                stream.write(json.dumps(dict(stage="summary", profile=profile, **totals,
                                             literal_nouns="requires human review",
                                             final_translation_correctness="not measured")) + "\n")


if __name__ == "__main__":
    main()
