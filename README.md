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

## Shareable Windows ZIP installation

If you received the code-only ZIP (no models), extract the **entire archive** and double-click **`Setup.bat`** in the extracted project folder. Keep **`scripts/Setup-Windows.ps1`** with it. The setup script:

- Requires Windows x64, an NVIDIA CUDA-capable GPU with a working Vulkan driver, internet access, and Windows Package Manager (`winget`).
- Uses WinGet to install **uv, yt-dlp, FFmpeg, and mpv** when missing.
- Installs Python 3.12 and the project dependencies using `uv sync --locked`.
- Installs the official `speech.cpp` Windows Vulkan build when missing and downloads **Haas-ja-v1.0**.
- Downloads the official **llama.cpp Windows CUDA 12.4** executable and matching runtime into `%LOCALAPPDATA%\\HaasPLaMo`. GitHub-provided SHA256 digests are checked when available.
- Leaves **PLaMo 3 Translate 2B Q4_K_M** (~1.81 GB) to auto-download on first launch, when `Start translate.bat` starts its own local llama.cpp server.

After successful setup, double-click **`Start translate.bat`**. If any installer requests UAC permission, follow Windows prompts. Rerunning `Setup.bat` is safe; installed packages/models are reused. Setup has not been run on another Windows PC, so version-specific installer issues may still need adjustment. Driver installation and hardware compatibility are not automated.

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
- **Neutral noun ownership:** Subtitle roles and ordinary noun phrases avoid invented `their/his/her/my/our/your` when the Japanese provides no owner: `最初のお客さん` → `First customer!`, not `Their first customer!` or `Your first customer!`. In a longer English sentence use `the first customer`; explicitly sourced Japanese ownership such as `お前の`, `俺の`, or `彼女の` stays intact. One context-reduced retry is attempted when PLaMo invents possession, followed by a limited grammatical fallback. This is not a global removal of all English possessives.
- **Neutral English subjects:** PLaMo now favors subjectless/impersonal English when Japanese omits its actor. Do not infer I, he, she, him, or her from the streamer identity, video title or past English subtitles; use singular they only when a third-person human is established but gender is not. Preserve explicitly stated/strongly implied Japanese first person and gender. The guard detects unsupported pronouns and requests one stricter context-free PLaMo retry; conservative postprocessing removes common unsupported gendered phrasing without reassigning the speaker. Set `[translate] neutral_subjects = false` to disable this new guard/fallback. No diarization or additional model.
- **Selective validation:** Detect obvious short positive-reaction reversals, missing negation on selected short Japanese patterns, numerals disagreeing when both source and English use digits, selected proper-name omissions, and prompt/loop leakage. Suspicious results get **one** context-reduced PLaMo retry. A second suspicious output does not replace the first; pre-existing output filters still apply. This is heuristic detection, not proof that other translations are correct.
- **Bounds:** `retry_suspicious = true` and `retry_timeout_s = 15.0`; most utterances stay one pass. The 30s normal PLaMo timeout is unchanged. Set `[translate] retry_suspicious = false` in `config.toml` for A/B comparisons.
- **Live/VOD metadata:** Available title and channel context now apply to both live and VOD, but never establish which player spoke.

To validate without model downloads:
```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Gender-neutral fallback does not erase an explicit gender reference in Japanese if a retry still disagrees; such cases remain flagged in logs for review. Common short subjectless reactions can also be phrased as `Finished.`, `Found it.`, `Got it.`, or `Not sure.` when the Japanese supports that wording.

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


## Quality and reliability pass (2026-11-12)

- **Adaptive FireRedVAD:** 9s VOD / 5.5s live natural-pause targets, requiring a sustained 180ms low-confidence gap. The existing 14/16s VOD and 7/8s live maximum chunk rules still apply. Override `adaptive_split = false` in `[vad]` for an A/B comparison.
- **Live cues:** playback uses the shared HLS source; stale completed cues more than 7s behind the mapped MPV clock are dropped rather than resurfaced out of context. Override `live_late_window_s` in `[mpv]`.
- **PLaMo fragments:** short context-dependent Japanese utterances receive the immediately preceding Japanese cue as reference-only context. Existing translation history remains bounded; disable with `fragment_context = false` in `[translate]`.
- **Missed VOD speech:** an empty unusable Haas result is retried once with an additional 200ms edge silence *only* when FireRedVAD confidence is at least .80 and duration is 0.6-10s. This does not add new speech or force hallucination-prone probing. No extra live retries; toggle `retry_empty_speech = false` in `[asr]`.
- **Review counters:** PLaMo logs which heuristics triggered a retry and whether the candidate was accepted; runtime logs the recent pipeline p95 latency and live processing-budget misses. These are not validated quality scores.
- **Reliability:** bounded Haas speech.cpp response wait (30s), explicit Haas and owned llama.cpp cleanup, offline unit tests and a CSV benchmark tool.

Measure Japanese CER from aligned manually reviewed reference clips with `python scripts/benchmark_quality.py YOUR_ALIGNED_REFERENCE.csv`. The CSV columns are `reference_ja,output_ja,reference_en,output_en`. Human review of English meaning, proper names, negation and subtitle synchronization remains necessary. Do not interpret the English guard-flag count as translation accuracy.

CPU-only verification: `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v`. For full validation, test long HoloGTA VOD clips, 30+ minute live HLS playback, pausing, backward/forward seeking, and process exit on the actual Windows/Vulkan/CUDA machine before relying on the new settings.

## Livestream timing and validation

Use the existing configuration. The default live startup delay is 9 seconds and
`[mpv] live_late_window_s` defaults to 7. Keep FFmpeg and its companion
`ffprobe.exe` on PATH. This change adds no Python packages or GPU models.

The local writer preserves source timestamps for live video, including variable
frame intervals, and copies AAC. FFmpeg can derive missing PTS from available
decode timestamps. Each new cache directory gets a local fragment probe for
overlapping audio/video presentation times. On recovery from separate renditions,
the resolver prefers a combined rendition within the selected height ceiling.
Unusable timestamps or failed verification are logged. Recovery has three attempts.

FFmpeg sends packed PCM on stdout and post-resampling frame metadata on stderr.
Each frame record specifies its integer media timestamp, sample rate and exact byte
count. The reader consumes both in order and checks continuity. This reuses the
installed FFmpeg build and the same local cache. VAD receives sample coordinates
in the shared media timeline. With MPV start-time rebasing disabled, later playlist
opens can observe a different segment and still use the same timestamps.
The scheduler compares MPV audio playback PTS with its video playback position
and the decoded audio span before rendering. A player reset starts a fresh epoch.

On-time cues use the original speech interval. An eligible late cue can receive
a two-second emergency display, limited to one per two seconds of playback.
A pause can retain up to 256 pending cues. The existing five-second VOD translation
lead and direct source timeline have their own playback path.

### Windows live acceptance runs

1. Run the offline test suite. Record the commit, stream URL, selected quality,
   FFmpeg/MPV versions and live delay. Save a separate run log for each session.
2. Watch 30 minutes at 480p, 2 hours at 720p and 6 hours at 1080p where available.
   Record the actual quality if YouTube falls back. At startup, every 30 minutes
   and before ending, capture several clearly identifiable spoken cues with audio.
   Compare first audible speech with first displayed text, targeting 250–500 ms
   on controlled material. Record late translations separately.
3. Pause and resume, allow a brief network interruption, and observe a recovery.
   Check for an epoch change on reconnect, cleared old text, and correct alignment
   after playback resumes. A normal segment rotation or short pause should not
   cause a reload. Close MPV and check that its owned processes exit.
4. Inspect `logs/run.log` for the first segment, epoch, verified state, audio PTS
   range, player PTS, late counters, and VAD/ASR/translation/cue-ready percentiles.
   The player A/V delta and its change since the first verified observation describe
   the two IPC readings. Measuring spoken-to-visible alignment requires the recording.
   The cue-ready latency starts at estimated local arrival of the final speech sample.
5. Play a VOD, seek forward and backward, and confirm the original playback position
   and cached subtitles. Record any repeated pauses or increasing delay.

Timing references: [FFmpeg timestamp options](https://ffmpeg.org/ffmpeg.html),
[FFmpeg audio frame metadata](https://ffmpeg.org/ffmpeg-filters.html#ashowinfo),
[MPV playback properties](https://mpv.io/manual/master/#properties),
[HLS rendition synchronization](https://www.rfc-editor.org/rfc/rfc8216#section-6.2.4),
[PyAV time bases](https://pyav.basswood.io/docs/stable/api/time.html).
