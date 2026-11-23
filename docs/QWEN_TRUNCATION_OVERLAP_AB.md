# Qwen3-ASR: token-limit recovery boundary A/B/A

**Goal:** test whether protecting a few audio samples across Qwen's
token-limit recovery cut prevents clipped words in holoGTA, without globally
shortening VAD utterances or changing accepted transcripts.

**Status:** Experimental, default OFF. No change to normal playback.

## What was changed

When a Qwen request exceeds its token budget, the existing production
recovery searches for a lower-energy point near the middle and divides
the utterance into two shorter independent ASR requests. Before this
experiment, the two requests touched exactly at the cut without any shared
audio. That is still the **production** behavior.

The diagnostic adds an *optional* 0.16s or 0.24s total audio overlap
centered on the same low-energy split point. Both halves remain subject to
the existing Japanese language, transcript quality, deadline, and generation
cancellation checks. There is **no** automatic de-duplication or transcript
rewriting. If both halves independently speak the same word at the edge,
the joined text could contain a duplicate.

The diagnostic invokes the real two-window recovery directly even when a
full-window decode does not hit its token limit. This isolates boundary
behavior, but does **not** claim that the normal production path would
have retried that scene.

To account for normal decode variability, it compares three passes on
the same exact PCM and checkpoint:

- A1: zero-overlap recovery control
- B: experimental 160ms total overlap
- A2: zero-overlap recovery control again

Only when A1 and A2 agree can a different B transcript be attributed more
credibly to the overlap. Neither a changed transcript nor additional
characters prove more accurate recognition without listening.

## Windows testing the hypothesis

- HG06: crosstalk and microphone mention; long full-window decode
- HG07: spoken Rui and Lamy, including suffixes
- HG10: crowded shouting and Suichan
- HG11: difficult rapid dialogue
- HG13: Matsuri scene, previously expensive long decode
- HG21: overlapping dialogue
- Q16: known 15.8s truncation recovery example

## PowerShell 7: sync and unit-test

From the existing project folder, check for local edits **before**
restoring anything:

```powershell
cd C:\Games\video-translate-main
git fetch origin main
git status --short -- src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_trunc_overlap.py
```

If the printed status is empty for those three files:

```powershell
git restore --source origin/main -- src/auto_translate/asr/qwen3_asr.py scripts/compare_haas_qwen_windows.py tests/test_qwen_trunc_overlap.py
uv run pytest -q --basetemp ".\.pytest-qwen-overlap" tests/test_qwen_trunc_overlap.py tests/test_compare_haas_qwen_windows.py tests/test_audit_20261122.py
```

If those files contain uncommitted personal edits, save or reconcile them
instead of overwriting them with git restore.

## GPU test on the existing 20-minute holoGTA clip

Close VideoTranslate.exe and leftover project llama-server workers first.
Run the command below as a **single PowerShell line**:

```powershell
uv run python .\scripts\compare_haas_qwen_windows.py ".\hologta-stress-test.flac" --model qwen3-asr --window HG06 HG07 HG10 HG11 HG13 HG21 Q16 --qwen-trunc-overlap-s 0.16
```

This runs the standard Qwen3-ASR Q8_0 GGUF and matching mmproj. It
does not load PLaMo, Gemma, Haas, or Qwen-JA. It does not change
config.toml, production VAD, or MPV settings.

Outputs:
- `benchmarks\runs\asr-windows-...\haas_qwen_windows.jsonl`
- `benchmarks\runs\asr-windows-...\haas_qwen_windows.csv`
- `benchmarks\runs\asr-windows-...\report.md`

For manual review focus on actual words spoken within about **0.5 seconds
either side of the boundary**. A low-energy frame is only a heuristic:
game audio or another speaker may mask the real speech break. Compare
lost words against duplicate syllables and words on both sides.

If real audio confirms less loss and no extra repeated dialogue, repeat on
a separate clip before considering opt-in production changes. Otherwise
keep 0 overlap in production.

**Not in scope:** ITN, vocabulary/system prompts, global 10s/12s VAD caps,
switching ASR models, changing proper-name spellings, or generic text
normalization. These are independent experiments.
