# Qwen3-ASR: crosstalk context-window experiment

**Status: experimental only.** No production VAD, Qwen decoder, ITN,
translation backend, model, system vocabulary, or auto-merging changes.

## Why

The uploaded HG06, HG10, HG13 and Q16 audio contains several overlapping
voices. Past 10/12-second splitting, vocabulary prompts and 160ms recovery
overlap did not yield consistently better transcripts. For a fair next test,
we need to know whether **different amounts of surrounding spoken context**
help the same central exchange without cutting a name or the sentence ending.

The existing ASR script now runs this optional **A/B/A** for each scene:
- A1: tight approximately 6-second excerpt around an approximate focus point
- B: larger approximately 10-second excerpt containing all of A1
- A2: the exact same 6-second audio samples as A1, for stochastic variability

Normal full-window and separate half-window diagnostics are still present.
The script reports the exact media timestamps of every request. None of the
three contextual crops is automatically published as subtitles or merged.

**Important**: the short and long crops cover different durations.
Different total character counts are NOT an accuracy metric and are not
directly comparable. Listen to the **common ~6-second region** and compare
its words, names and omissions; do not count text outside that shared area.
A1/A2 agreement is a useful control but does not prove correctness.

These are *approximate*, manually selected clip-local focus centers based on
the 16kHz WAVs uploaded in this conversation; they are not forced alignments,
diarization, speaker labels, or confirmed phrase boundaries:

| Window | Start in FLAC | Focus inside excerpt | Target phenomenon |
|---|---:|---:|---|
| HG06 | 224.00s | +9.0s | Repeated practice visit, microphone mention |
| HG10 | 350.69s | +11.0s | Drink requests, Suichan / Lamy crosstalk |
| HG13 | 425.00s | +13.0s | Matsuri / final repeated invitation |
| Q16 | 611.20s | +11.5s | Descending, waiting, X-key instructions |

If the intended phrase is outside the small shared region, the result is
**not evaluable** for that phrase; it is not an ASR omission. The clip is
only 20 minutes long and the windows are relative to it, not the original
1h25m stream offset.

## PowerShell 7

Run from your existing project root. First inspect local changes:

```powershell
cd C:\Games\video-translate-main
git fetch origin main
git status --short -- scripts/compare_haas_qwen_windows.py tests/test_compare_haas_qwen_windows.py
```

If the displayed files have no local edits you need to retain:

```powershell
git restore --source origin/main -- scripts/compare_haas_qwen_windows.py tests/test_compare_haas_qwen_windows.py
uv run pytest -q --basetemp ".\.pytest-qwen-focus" tests/test_compare_haas_qwen_windows.py tests/test_qwen_trunc_overlap.py tests/test_audit_20261122.py
```

Close VideoTranslate.exe and check no orphaned Qwen/llama-server processes
remain. Run **one** benchmark command:

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG06 HG10 HG13 Q16 --qwen-crosstalk-focus
```

This loads just the pinned standard Qwen3-ASR Q8_0 GGUF and matching
mmproj from `models\qwen3-asr`, not the Japanese fine-tune.
No PLaMo, Whisper, Haas, Gemma, or additional GPU inference model
is loaded. Regular VAD segmentation remains unchanged.

## Reading the experiment

The output directory `benchmarks\runs\asr-windows-...` contains
`haas_qwen_windows.jsonl`, `haas_qwen_windows.csv`, and `report.md`.

Inspect the `focus_tight`, `focus_wide`, and `focus_tight_repeat`
rows first. They have correct audio start/end timestamps. The report
flags `baseline_varies`, `language_changed`, and any error/rejection.
Do **not** merge transcripts from short and long crops.

Optional phrase checks **to verify by listening** (not ground-truth CER):
- HG06: 練習見に来ました; ensure both instances are real in the crop
- HG10: early/late ドリンク出して, then Suichan and Lamy names
- HG13: 絶対もう一回 versus distorted ending
- Q16: 今降りるから and the X-key instruction

A candidate is useful only if manual listening supports it *and* it
does not worsen names, negation, amounts, or speaker turns. Keep
unverified competing lines unscored.

The repository already contains `scripts/prepare_asr_reference.py`
for human-corrected Japanese reference sheets and CER. Use it for
accuracy claims rather than treating repeated model hypotheses as truth.
No ITN work is included in this test.
