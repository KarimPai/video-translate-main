# Qwen3-ASR — selective rejected-speech recovery (experimental)

Status: committed to main, **OFF by default**. Requires real-audio verification
before treating this as an improvement.

## What it does

Qwen3-ASR already retries output-token truncation with a bounded 15s
split-first optimization. That is unchanged.

New opt-in: if the normal Japanese ASR transcript is **already rejected** by
the shared quality guard for one of these reasons:

- repeated_character
- repeated_kana_loop
- implausible_length
- implausible_speech_rate

...the backend may try two shorter audio decodes, sequentially, within the
original utterance deadline. Normal accepted dialogue is never retried.

The recovery is permitted only if:
- The source transcript is language=ja, at least 6 seconds long, and not a VAD
  negative probe.
- More than 1.25s remains in the existing decode budget.
- Playback generation is still current and the model has not been closed.
- Both halves produce nonempty, fully accepted transcripts in the **same
  language**, and the result is Japanese and differs from the rejected output.
- The joined text also passes the existing quality guard.

If any step fails, the original rejection and language metadata are restored.
No partial transcript is published. A recovery exception is logged and
contained rather than aborting playback.

**Not fixed by this mechanism:** errors such as Rui→Luis, 300,000→100,000,
wrong negation, misheard names, or missed dialogue when the initial transcript
looks structurally valid. Those require audio-grounded tests, not a guessed
text-confidence threshold. Recovering more characters is not guaranteed
to mean higher accuracy.

## Controlled PowerShell 7 experiment

From the existing project directory:

```powershell
cd C:\Games\video-translate-main
git fetch origin main
git status --short -- src/auto_translate/config.py src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_rejected_recovery.py tests/test_compare_haas_qwen_windows.py
```

If you have **no personal local edits** to preserve in those files:

```powershell
git restore --source origin/main -- src/auto_translate/config.py src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_rejected_recovery.py tests/test_compare_haas_qwen_windows.py
uv run pytest -q --basetemp ".\.pytest-qwen-rejected" tests/test_qwen_rejected_recovery.py tests/test_compare_haas_qwen_windows.py tests/test_audit_20261122.py
```

Then close VideoTranslate.exe, confirm no leftover project ASR workers are
running, and compare baseline + opt-in recovery with the **same** standard
Qwen3-ASR Q8_0 model/projector and fixed audio:

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG05 HG06 HG07 HG10 HG11 HG12 HG13 HG21 HG26 HG27 HG28 HG29 --qwen-recover-rejected
```

The new `quality_recovery` rows in the JSONL/CSV are the experimental arm.
The `full` rows in the same file explicitly force recovery off as control.
The script restores your prior setting after each paired test. Do not combine
this flag with `--qwen-short-target-s` or `--qwen-split-first`.
Audio windows still have separate historical half-window diagnostics.

Review `benchmarks\runs\asr-windows-...\haas_qwen_windows.jsonl` and
`report.md`. In the earlier 10s/12s transcripts, there were no eligible
mechanical rejections, so **matching results are expected** on the same audio.
A zero-change test only validates no visible regression, not a quality gain.

## Optional normal playback setting — do NOT enable until verified

Only after actual audio and stable GPU tests demonstrate a benefit, add this
key to your **existing** `[asr.qwen3-asr]` profile:

```toml
recover_rejected_speech = true
```

Keep explicit standard Qwen model and matching mmproj paths, especially now
that Qwen-JA is also installed. Do not create a duplicate TOML section.

## Operational scope

Only `src/auto_translate/config.py`,
`src/auto_translate/asr/qwen3_asr.py`, the existing ASR window benchmark,
and CPU-only tests were changed. No production VAD tuning, GUI changes,
model downloads or modifications to your user `config.toml`.
