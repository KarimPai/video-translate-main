# Proper-name benchmark protocol

`scripts/benchmark_proper_names.py` separates authored text regressions from actual acoustic measurements. A clean text result cannot establish Haas, Whisper, LFM or PLaMo model accuracy.

## Deterministic baseline comparison

Run from the existing checkout:

```console
PYTHONPATH=src python scripts/benchmark_proper_names.py text --output work/proper-name-text-results.json
```

The default baseline is the audited remote HEAD, `fd3aae8c9b174f5cbaceed2666cd5f873d301c80`. The script reads its original `glossary.py`, `name_hints.py`, `glossary.toml` and `_current_glossary_matches` implementation through `git show`; it does not create a second checkout or approximate the old logic. Its output includes every fixture, prediction and numerator/denominator so the aggregate can be checked.

The 32 authored fixtures cover Matsuri and Miko names, literal festivals and shrine maidens, mixed meanings, kana and kanji variants, confirmed preceding Japanese dialogue, verified participants, missing and misleading titles, weak or contradictory ASR peers, short nicknames, honorifics and absent names. These expectations are semantic test inputs, not a representative speech corpus. Captions, chat, model timeouts and malformed outputs belong to the regression suite, rather than this text matcher benchmark.

The reported measures are:

- **Name recall:** exact expected `PERSON` occurrence with its spoken English form. Expanding Matsuri to Natsuiro Matsuri counts as a miss.
- **False-positive rate:** unexpected person occurrences divided by authored nonperson occurrence slots, plus one negative slot per no-name case. Inspect absolute counts with the rate.
- **Literal-noun preservation:** a literal source occurrence is not classified as a person. This does not assert that a model translated the entire sentence correctly.
- **Deterministic repair exact match:** safe English span correction against authored output expectations. The English inputs are fixtures; no PLaMo model runs here.
- **Abstention:** `UNKNOWN` is retained when a title or common grammar cannot distinguish the meaning.

`COMMON_NOUN` and `UNKNOWN` both prevent a name glossary hint, but remain separately visible in the case diagnostics. Sparse legacy matchers did not classify ordinary nouns, so their missing occurrences are recorded as `UNKNOWN` for exact-decision diagnostics.

An initial run of fixture version 1 against the audited baseline produced the following **text-only** results. Re-run the command above after changing either the resolver or fixtures.

| Measure | Baseline | Resolver |
|---|---:|---:|
| Expected spoken name occurrences retained | 17 / 20 | 20 / 20 |
| Unexpected person names / negative slots | 5 / 16 | 0 / 16 |
| Literal noun occurrences preserved | 7 / 7 | 7 / 7 |
| Deterministic English repairs exactly matching fixtures | 23 / 24 | 24 / 24 |

These are authored functional cases, not statistically representative rates. The improvement in English fixture repairs is not a measured improvement in PLaMo translation accuracy. The six ambiguous occurrences intentionally left `UNKNOWN` by the resolver are visible in the JSON report.

## Recorded audio experiments

No reusable recorded clips or human-reviewed transcripts are bundled. No real ASR accuracy result is supplied by this change. Keep production keyword boosting and prompt hints opt-in until representative measurements justify a change.

Create a JSON manifest alongside appropriately reusable 16 kHz mono signed 16-bit PCM WAV files:

```json
{
  "clips": [
    {
      "clip": "reviewed-001.wav",
      "human_reviewed": true,
      "reusable": true,
      "provenance": "Describe permission, origin and human transcription review",
      "reference_ja": "まつりが文化祭に来た",
      "previous_ja": "夏色まつりと話した",
      "confirmed_participants": ["Matsuri"],
      "expected": [
        {"source": "まつり", "classification": "PERSON", "target": "Matsuri"},
        {"source": "祭", "classification": "COMMON_NOUN"}
      ]
    }
  ]
}
```

The example describes a manifest schema, not an existing clip or measured result. Include positive and negative examples, kana/kanji ASR variations, literal references, mixed sentences, overlapping/quiet speech, silence, music and no-name clips. Review transcripts and contexts independently of model output. Never derive a hint from a reference transcript or a video title.

With the application's speech.cpp, model files, faster-whisper and CUDA runtime installed:

```console
PYTHONPATH=src python scripts/benchmark_proper_names.py audio clips/manifest.json work/proper-name-audio.jsonl
```

On Windows PowerShell, set `$env:PYTHONPATH = 'src'` before the Python command. Use the same installed runtime and model files for every profile. The harness runs one Haas worker, closes it, and then runs one Whisper worker at a time. It preserves the Japanese-only transcription settings and uses the production worker adapters and their deadlines. It does not load PLaMo or a second persistent LLM.

The experiment matrix is:

| Recognizer | Compared settings |
|---|---|
| Haas | No keywords; guarded keywords at alpha 1.0, 1.5 and 2.0 |
| Whisper Turbo | Beam 6 and 8, each with no hint, `hotwords` and `initial_prompt` |
| Optional Whisper experiment | Add `--beams 6 8 10` |

Keywords and prompt hints come only from names actually heard in the no-keyword Haas baseline and confirmed by the local resolver. At most four current spoken forms are retained. Reference transcripts and titles never seed hints. This deliberately excludes a name that the baseline did not hear; it is a conservative comparison of supported name biasing, not an oracle-vocabulary experiment. A profile with no grounded candidates is effectively another unprompted decode and must not be reported as a successful hint experiment.

The JSONL preserves audio hashes, human provenance, transcripts, exact settings, actual keyword capability/fallback diagnostics, confidence diagnostics, elapsed time and failures. Compare only profiles that actually activated the requested feature: an unsupported-option fallback is not evidence that keyword boosting was tested. Model scores are uncalibrated diagnostic signals. Character edits are surface errors; manually review their linguistic significance. The harness records expected and resolved name counts, but a full acoustic benchmark also needs independent human review of literal nouns and final English translations. No final translation correctness is claimed by this ASR-only harness.

Use a balanced holdout set to choose strengths and beam size. Report the baseline and every attempted configuration, including false names, unchanged results, unsupported configurations and failures. Warmup/order can affect timing; the harness reverses Whisper prompt order on alternating clips, but is intended for accuracy comparison rather than a latency claim. CPU/GPU decoding, other quantization and full Large-v3 require separate controlled runs with the same clips, model versions, settings and resource telemetry. The production defaults should change only when those measurements justify it.
