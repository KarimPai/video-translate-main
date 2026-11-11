# Haas Japanese ASR + PLaMo 3 Translate 2B — 2026-11-11 Retest

**Experimental branch:** `feature/haas-plamo-retest-20261111`, forked from the latest Haas/PLaMo `main` baseline. This is not the VoicePing/Whisper AST pipeline. `main` is unchanged.

## Active pipeline

`YouTube live/VOD or local source → FFmpeg original 16 kHz mono audio → FireRedVAD (CUDA) → Haas-ja-v1.0 (Vulkan speech.cpp) → PLaMo 3 Translate 2B Q4_K_M (local llama.cpp) → English-only MPV captions`.

- MPV plays the original video with the native timeline and seekbar. Source ingest is audio-only for direct-video playback.
- Terminal prints `JP:` and `EN:` lines; MPV overlays only English, up to three stacked lines.
- Japanese YouTube VOD captions are optional evidence and can assist Haas misrecognition.
- `glossary.toml` and the original title/channel context remain active for proper names and terms. Existing conservative Watame/Watanabe correction is preserved.
- No VoicePing, Whisper, separate LLM beyond PLaMo, Sherpa diarization, forced aligner or new audio classifier.
- Vocal separation and DeepFilterNet are disabled by default; the original speech signal is used.

## First-time Windows setup (separate from main and VoicePing)

Requires Python 3.12+, uv, FFmpeg, yt-dlp, mpv, a working Vulkan `speech` CLI from speech.cpp, Haas-ja-v1.0 model, and a GPU-enabled `llama-server` for PLaMo.

```bat
cd /d C:\Games && git clone -b feature/haas-plamo-retest-20261111 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-haas-plamo-retest && cd /d C:\Games\video-translate-haas-plamo-retest && uv sync && uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v && "Start translate.bat"
```

Choose video quality **1=360p, 2=480p, 3=720p (default), 4=1080p**. VODs prefer direct HTTPS/DASH sources with lower-quality fallback; live sources prefer HLS.

### Optional 480p RTX Video Super Resolution (experimental)

When selecting **2 (480p)** in `Start translate.bat`, answer **Y** to try NVIDIA RTX VSR 3x enhancement (~1440p output). It is **off by default**; other quality choices use unchanged MPV playback. Enable **Video Super Resolution** and manually choose **High** under **NVIDIA App > System > Video**; MPV cannot change NVIDIA's quality tier.

CLI: `--mpv --video-quality 480 --hwdec auto-safe --rtx-vsr`, or opt in through `[mpv] rtx_vsr = true` in your local config. Windows and MPV D3D11 hardware decoding are required. The filter is applied only after MPV confirms an input video height of 480p or less. It is removed if unavailable, hardware decoding falls back to software, or 30-second health samples find at least 15 dropped or 30 delayed frames. Manual/cache pauses are excluded from performance alerts. The underlying MPV session, native seeking, FireRedVAD, Haas ASR, PLaMo translation, and English subtitles remain unchanged.

**Verify in practice:** MPV accepting the filter does not prove NVIDIA AI enhancement is running. Check the NVIDIA App RTX Video indicator, `logs\\mpv.log`, and `logs\\run.log` for `RTX VSR playback check` or `RTX VSR disabled`. Compare the same 480p VOD at 30/60 fps with VSR on and off, and monitor VRAM via `nvidia-smi`; performance and activation are unverified until tested on your RTX 3070. For CPU-only tests run `uv run --no-sync python -m unittest discover -s tests -p "test_rtx_vsr.py" -v`.


## Accuracy-first FireRedVAD defaults (experimental)

The retest branch now prioritizes **retaining audible Japanese speech over minimum subtitle latency**. No new model is installed: Haas-ja-v1.0 (speech.cpp Vulkan) and PLaMo 3 Translate 2B are unchanged. The VAD receives the original 16-kHz mono audio, without GTCRN, DeepFilterNet, or diarization.

```toml
[vad]
threshold = 0.38
neg_threshold = 0.25
min_speech_ms = 200
min_silence_ms = 550
speech_pad_ms = 320
onset_ms = 70
prob_smooth_frames = 5
short_reply_ms = 120
short_reply_probability = 0.85
max_utterance_s = 14.0
max_split_extension_s = 2.0
live_max_utterance_s = 7.0
live_max_split_extension_s = 1.0
split_pause_min_ms = 120
split_context_ms = 160
fallback_s = 0.0
```

- **Speech sensitivity:** 0.45→0.38 threshold and 100→70 ms onset gate aim to recover quieter words and rapid reactions; the 200 ms sustained-speech requirement remains, with the existing high-confidence short-reply exception.
- **Sentence completeness:** 200→320 ms speech padding, 500→550 ms end silence, and a longer quiet-point split search aim to avoid clipping beginnings/endings and splitting sentences early.
- **VOD:** 14 s soft utterance cap / 16 s hard cap instead of 11 s / 12 s. Haas has full-context attention, but very long chunks can still lose words, so the hard cap remains bounded.
- **Live:** 7 s soft / 8 s hard cap for the existing 9-second playback delay. This is a timing safeguard, not a claim that shorter segments recognize better; live needs comparative testing.
- **Anti-hallucination:** VAD-negative fallback probing stays **off**. Avoided enabling speculative extra decoding on music, and kept translation repeat/prompt-leak rejection.
- **Overrides:** Your local `config.toml` continues to override these defaults. Check effective values in `logs\\run.log` under `FireRedVAD accuracy-first`.

This favors recall and context; it can also pass more game audio to Haas and may create false positives. Validate on identical HoloGTA material, especially short reactions, proper names, negations and music-only portions. No measured recognition improvement is claimed until comparison.

## PLaMo context and safeguards

```toml
[translate]
plamo_hf_model = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
plamo_context = 4096
plamo_gpu_layers = 99
max_tokens = 192
context_lines = 4
memory_lines = 8
glossary = "glossary.toml"
remove_fillers = true
collapse_repetitions = true
suppress_runaway_loops = true
max_identical_cues = 2
neutral_subjects = true
retry_suspicious = true
retry_timeout_s = 15.0
```

PLaMo's timeout is now 30 s (previously 20 s) to tolerate rare long completions; this is a failure timeout, **not** an intentional subtitle delay. Its output allowance is 192 tokens (previously 120), reducing truncation risk with longer VOD chunks. Context remains at 4096, with 4 dialogue lines and 8 memory lines to limit inaccurate context carryover and VRAM use.

PLaMo's dedicated translation template still uses the Japanese transcript, relevant glossary matches, video metadata, and short recent JP/EN pairs. Unlike VoicePing ASR, PLaMo is the **translation** model; its existing translation template is retained. Dialogue is now limited to 4 recent pairs and Japanese-only scene memory to 8 lines to minimize unreliable prior English context.

New post-processing filters obvious filler/laughter-only English, long repeated sentence loops, recognizable template echoes, and the third or later identical nearby multiword subtitle. It **does not paraphrase** the Japanese text or arbitrarily change names/negation. Long-loop rejection discards the suspicious cue entirely rather than guessing which parts were spoken.

## PLaMo accuracy and context optimizations (2026-11-11)

This experiment **keeps Haas-ja-v1.0 and PLaMo 3 Translate 2B Q4_K_M**. No additional LLM, diarization model or ASR is loaded. The default temperature remains 0 and the PLaMo context window remains 4096.

- **Context-supported ASR terminology:** In Holo-tagged videos only, a small exact correction map fixes recognized variants such as `コロナイブGTA` → `ホロライブGTA` and `フォロメン` → `ホロメン`. Never fuzzy-replace other dialogue or modify negation/numbers, and never override a reliable aligned caption that explicitly contains the original variant.
- **Dynamic scene glossary:** HoloGTA titles enable a limited selection of police, gang, hospital and bakery terms; each is sent only when its Japanese phrase occurs in the current transcript. Longest matches win. Normal static `glossary.toml` remains active everywhere.
- **Shorter, less contaminating memory:** `context_lines = 4`, `memory_lines = 8`. Past PLaMo English terms are no longer fed back as tentative facts. Recent Japanese and terms from the verified fixed glossary are still available; history still resets on seeks.
- **Neutral English subjects:** PLaMo now favors subjectless/impersonal English when Japanese omits its actor. Do not infer I, he, she, him, or her from the streamer identity, video title or past English subtitles; use singular they only when a third-person human is established but gender is not. Preserve explicitly stated/strongly implied Japanese first person and gender. The guard detects unsupported pronouns and requests one stricter context-free PLaMo retry; conservative postprocessing removes common unsupported gendered phrasing without reassigning the speaker. Set `[translate] neutral_subjects = false` to disable this new guard/fallback. No diarization or additional model.
- **Selective validation:** Detect obvious short positive-reaction reversals, missing negation on selected short Japanese patterns, numerals disagreeing when both source and English use digits, selected proper-name omissions, and prompt/loop leakage. Suspicious results get **one** context-reduced PLaMo retry. A second suspicious output does not replace the first; pre-existing output filters still apply. This is heuristic detection, not proof that other translations are correct.
- **Bounds:** `retry_suspicious = true` and `retry_timeout_s = 15.0`; most utterances stay one pass. The 30s normal PLaMo timeout is unchanged. Set `[translate] retry_suspicious = false` in `config.toml` for A/B comparisons.
- **Live/VOD metadata:** Available title and channel context now apply to both live and VOD, but never establish which player spoke.

To validate without model downloads:
```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

The correction vocabulary is intentionally small to minimize false positive name substitutions. The heuristic retry can still fail or worsen a result; check your actual HoloGTA transcript, negations, names, peak VRAM, and subtitle latency against the pre-optimization commit. Long-sentence boundary joining is deliberately not automatic because it can shift subtitle timestamps, especially during live viewing.

## Playback and reliability

- **MPV VOD starts after approximately five seconds of translated audio lead**, and later seeks rebuild this buffer while retaining the original source's seek timeline.
- MPV VOD audio read-ahead is capped at **75 seconds** relative to video position. During intentional manual pause, the reader eventually parks instead of processing a whole VOD in advance. Set `[ingest] vod_max_audio_lead_s = 0` to disable.
- Worker failures are reported to the main runner instead of silently hanging. The queue watchdog excludes intentional VOD pause and audio read-ahead parking.
- Session-only subtitles and processed-region caching are preserved, including reuse after backward seeking.
- MPV safe hardware decode and the existing native OSC seekbar are preserved.
- For live playback, the configured original delay remains **9 seconds** (not a hard subtitle timestamp offset).

## Scope and limitations

This is a **port of compatible VoicePing/Whisper project findings**, not a model swap. Haas has no Whisper-style `no_speech_prob` or comparable decoder metrics, so Whisper retry thresholds were not copied. Qwen-specific natural-language prompts were not copied because they were emitted as subtitles in a prior VoicePing test. The optional language/audio-event detector ideas have not been integrated: FireRedVAD identifies speech but does not verify Japanese source language.

The more permissive VAD and 16-second VOD segments are **experiments**, not yet measured for Haas accuracy or latency. Speech.cpp's Haas implementation uses greedy TDT decoding, not a switchable beam search, and its published F16 versus Q8_0 CER is almost identical; no unverified decoding/model replacement was applied. Existing local untracked `config.toml` overrides these defaults, so first retest with a clean checkout or inspect local overrides. Live testing and native Windows GPU/Vulkan testing are still required.

**Trade-off:** A 16-second uninterrupted VOD utterance cannot be completed before its final audio arrives, so some subtitles may arrive later than your preferred 5–10 seconds. Live uses the shorter cap to reduce that risk. You can tune these values in `config.toml` without changing the code.

Run offline regression tests:

```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Check `logs\run.log` for application errors and `logs\mpv.log` for player warnings. Benchmark with the same VOD/time ranges as the earlier Haas/PLaMo baseline to compare recognition completeness, laughter hallucinations, names, negation, lead after seeks, 95th-percentile subtitle latency, and peak VRAM.
