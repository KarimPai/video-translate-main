# holoGTA audio-checked Japanese ASR reference — 30 windows

This set is based on the downloaded `hologta-stress-test.mp4` clip (the original stream's **01:25:00–01:45:00** section). All CSV `start_s` and `end_s` timestamps refer to **the 20-minute local clip**, not the original livestream.

## What is included

- `benchmarks/reference/hologta_30.csv`: 30 fixed time windows totaling about **6 minutes 6 seconds**; 27 speech candidates and **three unverified low-activity candidates**. The latter are not claimed to be silent. All windows require an actual audio check.
- `scripts/prepare_asr_reference.py`: loads prior 14s/0.28 VAD ASR run JSONLs, populates a review CSV, optionally extracts **16 kHz mono WAVs** using FFmpeg, and calculates Japanese character error rate (CER) from manually verified references.
- `tests/test_asr_reference.py`: CPU-only regression tests; no models run.

## Prepare on the project PC

From PowerShell in `C:\Games\video-translate-main`:

```powershell
uv run python .\scripts\prepare_asr_reference.py prepare `
  ".\hologta-stress-test.mp4" --extract-audio
```

The script searches `benchmarks/runs` for the latest **matched** 14-second VAD / 0.28 threshold runs with identical clip size. Each ASR can be paired with **any translation backend** (for example `haas-plamo8b`, `haas-hymt`, or `qwen3-asr-gemma4`), because this is a Japanese-ASR evaluation. You can override any source with `--haas PATH`, `--qwen PATH`, or `--whisper PATH`. If an ASR run is unavailable, the script warns and leaves that hypothesis column empty with `source_unavailable` in its boundary field. The missing model is **excluded** from scoring, not counted as a recognition failure. Audio extraction and human-reference preparation still proceed, even if all ASR logs are missing. It does **not** rerun GPU models. For reliable three-way comparisons, supply all three run JSONLs.

The output is `benchmarks/reference/working/hologta_30_review.csv`, `sources.json`, and 30 clips in `benchmarks/reference/working/audio/HG01.wav`, etc.

**Do not rerun prepare after editing the review sheet.** It refuses to overwrite an existing review CSV unless `--overwrite` is explicitly passed, which would discard corrections.

## Human verification

In the review sheet, for each row:

1. Listen to the matching `audio/HGxx.wav`, ideally using headphones. The candidate Haas, Qwen, and Whisper columns are *hints only*, not ground truth.
2. Fill `reference_ja` with **the Japanese actually audible**. Preserve spoken names as heard; do not invent unclear words to make a grammatical sentence. For code-switching, transcribe the audible speech, and note uncertainty in `review_note`. If unable to confidently correct a row, leave it `pending`.
3. Set `speech_present` to `yes` if intelligible speech occurs, or `no` only after confirming that the audio contains **no human speech**. For `no`, leave `reference_ja` empty.
4. Change `verification_status` to `verified` **only** once the audio check is complete. Do not mark a row verified merely because multiple models agree.
5. Use `haas_boundary`, `qwen_boundary`, `whisper_boundary` to identify partial/full-run segmentation mismatches. Review and adjust those hypotheses manually when comparing fixed windows.

**Reference accuracy caveat:** These candidate transcriptions come from models processing the *whole 20-minute clip*, where ASR utterance boundaries may differ. Midpoint alignment cannot divide a model's text at exact word timestamps. The resulting CER is an **exploratory** whole-run benchmark even after audio correction. For rigorous model rankings, rerun each ASR on the exact same 30 WAVs and score those hypotheses against the verified references.

## Score the verified rows

```powershell
uv run python .\scripts\prepare_asr_reference.py score
```

Outputs `scores.csv`, `scores.json`, and `window_errors.csv` under `benchmarks/reference/working/scores`.

- **CER:** character-level edit distance divided by the number of reference characters, pooled across *verified* speech windows. NFKC normalization removes punctuation/spacing only; no semantic equivalence is assumed. CER can exceed 100% if a model hallucinates excessive text.
- **Empty-speech-window rate:** proportion of verified speech windows where model's aligned Japanese hypothesis is empty. This is **not** a word-level speech-recall measurement.
- **False-speech-window rate:** proportion of verified non-speech windows with nonempty model text. Negative controls only count once an audio reviewer confirms no speech.
- **Boundary warnings:** number of scored windows where a full-run ASR cue extends outside the fixed reference window; such results must be manually inspected before interpreting CER.

No accuracy numbers are reported for unverified rows. Zero verified rows returns an explanatory exit status 2.

## Interpretation and next engineering steps

- Prioritize **Haas** as current practical ASR baseline; do not switch based on transcript length or cue counts alone.
- Once 20–30 rows are verified, inspect the highest-CER windows. Separate missing speech, speaker overlap, name confusion, game sounds, and ASR repetition/hallucination into independent failure categories.
- Fix only evidence-supported issues with regression cases. Keep VAD at `threshold=0.28`, `min_speech=120ms`, `min_silence=400ms`, `pad=300ms`, `soft_cap=14s` for the baseline.
- Then compare the **same Japanese reference** to translated English separately. Translation changes must not be counted as ASR improvements.


## No Japanese reading ability? Run the automatic disagreement report

If you cannot transcribe Japanese, **do not mark model output as verified** or
attempt to compute CER. You can still prioritize the next engineering fixes
without manual Japanese reference text:

```powershell
uv run python .\scripts\analyze_asr_disagreements.py
```

This reads the review CSV generated by `prepare` and creates an English
summary at `benchmarks/reference/working/analysis/report.md`, detailed
`disagreements.csv`, and machine-readable `summary.json`. It **does not**
modify the review CSV or run any model, and it needs no GPU.

The report calls out mismatched text, missing hypotheses, difficult/name-sensitive
windows, and cue boundary mismatches. A model that disagrees with two others
is **not thereby proven wrong**. Windows where all three outputs are empty
have **unverified speech presence**, not verified silence. These proxy signals
cannot establish ASR accuracy or generate a legitimate CER ranking.

To obtain actual accuracy later, use independently Japanese-verified
transcripts (a fluent reviewer or genuinely reliable, synchronized source
subtitles). In the meantime, keep the tested Haas / Silero baseline and focus
regressions on the reported high-risk windows.
