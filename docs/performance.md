# Measuring quality and latency

Production recognition stays sequential until hardware measurements justify a
change. Optional models warm after MPV loads video. Short offline A/B/C runs
explicitly wait for optional startup so they cannot silently become primary-only
runs. No RTX 3070 performance gain is assumed.

## Reports

Use the Transcript tab's Performance report action, or:

```sh
uv run --no-sync auto-translate --quality-report logs/asr-comparison.jsonl
```

This writes `logs/reports/quality-report.html` and `.json`. Records merge by
session, seek generation, and media timestamps. Reports show p50/p95 latency,
fallback/timeout outcomes, and RTF/VRAM when actually supplied. Unrecorded outcomes
are unknown, not success. Device VRAM includes other applications and sampling
can miss short peaks. Old logs may lack timing/resource measurements. English
guard flags are review signals, not English semantic scores.

For accuracy assessment, prepare a UTF-8 CSV with one aligned utterance per row:
`reference_ja,output_ja,reference_en,output_en,tags`. Use pipe-separated tags such
as `names|numbers|negation|overlap|quiet|silence`. An empty reference marks silence;
omit the reference field in JSON when unknown. Character errors include false
speech on silence. Optional `names_error`, `numbers_error`, and `negation_error`
columns record explicit human decisions.

```sh
uv run --no-sync python scripts/benchmark_quality.py paired.csv --output reports --hardware-label "RTX 3070, driver VERSION"
```

For optional JiWER scoring install `uv sync --locked --extra quality`, then add
`--jiwer`. Japanese CER is preferable to whitespace-based WER. Annotate logs only
using exact `session_id,generation,start_s,end_s` identities:

```python
from pathlib import Path
from auto_translate.quality_report import generate_report

generate_report(Path("logs/asr-comparison.jsonl"), Path("reports"),
                annotations_path=Path("annotations.csv"))
```

## Controlled ASR experiments

Prepare short local clips and a manifest `clip,reference_ja,reference_en,tags`,
with paths relative to the manifest. Include quiet reactions, proper names,
numbers, negations, overlapping voices, code switching, silence, and repeated
genuine speech. Choose a new output folder:

```sh
uv run --no-sync python scripts/benchmark_dual.py clips/manifest.csv benchmark-results --asr-matrix --beams 1,3,6 --schedules sequential,concurrent --repetitions 3 --warmup 1 --hardware-label "RTX 3070, driver VERSION"
```

Each beam/schedule runs in a bounded isolated process. Each engine is
single-flight; concurrent means at most one Haas and one Whisper call together.
The matrix measures warm ASR pair time and Japanese recognition, excluding
translation/playback. Model/device/driver failures invalidate a setting. Parent
deadlines terminate only owned children. Input fingerprints and package/hardware
metadata support reproducibility. Models load only on explicit benchmark runs.

Without `--asr-matrix`, the script compares production profiles A (Haas), B
(dual ASR), and C (dual ASR plus selector), including cold startup. It flags
degraded optional paths. Do not compare RTF across these two measurement scopes.
Preset recommendations require complete runs on identical annotated workloads,
hardware, and scope; they never overwrite `config.toml` automatically.

Before selecting a live preset, repeat with video decoding/VSR and live drafts
as actually used. Record omissions, names/numbers/negation errors, p95 latency,
and physical GPU headroom. Drafts reuse Whisper but still consume compute and
remain disabled by default. No actual hardware benchmark was run for this change.
