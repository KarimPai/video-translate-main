# Qwen3-ASR: controlled Hololive vocabulary hint experiment

**Status:** Experimental only. Normal playback sends **no vocabulary system
message** and the regular config.toml remains unchanged.

## Why this test

The model already produces useful Japanese on chaotic holoGTA, but sometimes
misrecognizes names (e.g., ルイ vs ルイス), and can hallucinate in music or
quiet windows. Official Qwen3-ASR supports domain-specific vocabulary through
a **system** message. The standard Qwen3-ASR GGUF conversion also documents
this use with llama.cpp's /v1/chat/completions API.

References:
- https://huggingface.co/docs/transformers/model_doc/qwen3_asr
- https://huggingface.co/minimalcorp/Qwen3-ASR-1.7B-GGUF

Your installed llama-server build has **not yet been runtime-verified** with
the extra system message. The experiment detects HTTP errors rather than
silently claiming that llama.cpp accepted or applied the hint.

The A/B/A run uses three identical, contiguous audio requests for each fixed
reference window, all using the **standard** Q8_0 checkpoint and projector,
with Qwen's auto language detection unchanged:

1. A: no system message (current production)
2. B: system message with just a few Japanese name candidates
3. A: no system message again (control for natural decoding variability)

The script also runs its historical unhinted two-half diagnostics for each
window. It restores the original hint after each A/B/A group.

The default hint list is:
`みこ, みこち, ラミィ, ルイ, まつり, すいちゃん, スイちゃん`

Do not infer that a hinted spelling is correct solely because it appeared.
Hints may *prime* words that were not spoken. These are **candidates**, not
a glossary replacement. No subtitle text is automatically rewritten.
Never include a number like 30万円 in this hint: that would bias the separate
money/negation evaluation in HG12.

## PowerShell 7: sync and CPU-only tests

From the existing repo directory:

```powershell
cd C:\Games\video-translate-main
git fetch origin main
git status --short -- src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_vocabulary.py
```

Only if the three files have **no local edits to preserve**:

```powershell
git restore --source origin/main -- src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_vocabulary.py
uv run pytest -q --basetemp ".\.pytest-qwen-vocab" tests/test_qwen_vocabulary.py tests/test_compare_haas_qwen_windows.py tests/test_audit_20261122.py
```

Stop VideoTranslate.exe and ensure no leftover project llama-server process
is using the same GPU. Then run **one line**:

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG02 HG06 HG07 HG10 HG11 HG12 HG13 HG21 HG26 HG27 HG28 HG29 --qwen-vocabulary
```

A smaller optional test (fewer GPU calls):

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG02 HG07 HG12 HG13 HG27 HG28 HG29 --qwen-vocabulary
```

Custom candidates are supported, e.g.:

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG02 HG07 HG13 HG27 HG28 HG29 --qwen-vocabulary --qwen-vocabulary-terms "みこ, みこち, ラミィ, ルイ, まつり"
```

The script explicitly selects the standard Q8 model and mmproj from
models/qwen3-asr (ignores the installed Qwen-JA model). It loads **one**
ASR model; no translator, no second ASR, no production config changes.
Do not combine --qwen-vocabulary with --qwen-short-target-s,
--qwen-recover-rejected, or --qwen-split-first.

## Upload and interpretation

The output directory `benchmarks\runs\asr-windows-...` contains:

- `haas_qwen_windows.jsonl`: all Japanese hypotheses, language tags,
  rejection reasons, variant labels and exact clip-local times
- `haas_qwen_windows.csv`: same data
- `report.md`: summary of A/B/A stability and warning flags

For each named scene listen to the actual audio and judge:
- HG02 — whether it recognizes みこ without expanding a nickname
- HG07 — spoken ルイ, not ルイス; ensure names don't appear without sound
- HG13 — spoken まつり, without converting it to a generic term later
- HG10 — すいちゃん vs unrelated phrases in shouting/crosstalk
- HG12 — ensure number and negation are **unchanged** without numerical hints
- HG27–HG29 — unverified low activity, check for extra phantom names

An A/B/A sequence is *consistent* with a vocabulary benefit only when two
unprompted runs agree and the hint improves a name actually heard in audio.
This still does **not** prove CER improvement. If unprompted controls differ,
label the experiment **inconclusive** on that window.

If the extra system turn triggers a llama.cpp HTTP error, do not edit the
production pipeline to force it. Inspect the JSONL language='error' entry
and the llama-server build before going further.
