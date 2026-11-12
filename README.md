# Experimental Haas + Nemotron 3.5 + LFM-JP + PLaMo

**Branch:** `feature/haas-nemotron-lfm-reconciler-20261113` — isolated from the previous stable dual-ASR branch.

| Role | Model | Runtime / precision |
| --- | --- | --- |
| Primary Japanese ASR | Haas-ja-v1.0, 616M | speech.cpp Vulkan0 |
| Secondary Japanese ASR | NVIDIA Nemotron 3.5 Streaming 0.6B | NeMo-Speech.cpp CUDA, official Q8_0 GGUF (~742 MB download) |
| Japanese A/B/U reconciler | LiquidAI LFM2.5-1.2B-JP-202606 | llama.cpp Q4_K_M (~731 MB download), GPU when admitted, CPU otherwise |
| JA → EN translation | PLaMo 3 Translate 2B | llama.cpp Q4_K_M with budget-based partial GPU offload |

The two ASRs receive the same FireRedVAD-segmented Japanese audio. Nemotron uses its **own** NeMo-Speech.cpp runtime; it is NOT compatible with the existing speech.cpp secondary worker. The secondary runs through a persistent local HTTP server so the model is not reloaded on every utterance. Its transcription uses `language=ja-JP` and never directly translates audio into English.

The reconciler selects exactly A (Haas), B (Nemotron), or U (uncertain; preserve Haas), using original Japanese hypotheses, recent Japanese and glossary context. It does not listen to audio and cannot guarantee a correct decision when both hypotheses are plausible. PLaMo retains the neutral-subject translation behavior. If Nemotron or the selector fails or cannot be admitted under the GPU budget, the system falls back to the existing Haas path rather than silently activating Qwen.

## Adaptive GPU memory policy and CPU FireRedVAD (2026-11-13)

**No fixed 7.5 GB application ceiling or forced playback stop.** The RTX 3070 retains its physical 8 GiB VRAM limit. `nvidia-smi` reports device-wide `memory.used,memory.total`, and admission uses the measured **free** memory rather than fixed used-memory thresholds. Model placement can use VRAM above 7.5 decimal GB when available.

The pipeline reserves approximately 0.65 GB at startup for transient MPV/CUDA allocations. GPU selector shedding occurs only if measured device free memory falls to 0.55 GB or less; optional ASR is also released if free memory reaches 0.25 GB. These are **physical headroom guardrails**, not a fixed used-memory cap. The app **never stops MPV solely because total usage passes 7.5 GB**. Device OOM remains possible; Windows WDDM/driver allocations and RTX VSR may vary during playback.

FireRedVAD now defaults to CPU (`[vad] device = "cpu"`) and uses the **same** FireRed streaming model, 10 ms frame probabilities, segmentation, thresholds, padding and seek resets. `device = "cuda"` remains available for A/B testing. CPU avoids establishing a PyTorch CUDA context for the tiny model and may free driver/buffer VRAM. The exact VRAM saving requires measurement.

To prioritize resident Haas, Nemotron and GPU LFM on an 8-GiB-class card, PLaMo is limited to **12 GPU layers** and its llama.cpp server uses **one inference slot** (`-np 1`), rather than auto-allocating four parallel contexts. PLaMo's 4096 context and translation template remain unchanged.

Older local `config.toml` GPU fields `target_gb`, `optimization_threshold_gb`, `emergency_threshold_gb`, and `maximum_gb` are ignored automatically. To override new defaults, use:

```toml
[gpu_budget]
startup_headroom_gb = 0.65
low_free_gb = 0.55
emergency_free_gb = 0.25
sample_interval_s = 2.0

[vad]
device = "cpu"
```

**Verify:** use `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v` then a 10–15-minute test of the same VOD with RTX VSR on and off. Inspect `logs/run.log` for `GPU 0 used=... total=... free=...`, `FireRed VAD initialized on CPU`, `Nemotron ASR ... decoded in ...`, `LFM2.5 ... ready`, and any fallback events. Also check `logs/plamo-server.log` for `n_slots = 1`. The new policy has **not** been validated yet on your Windows GPU.

---

## Windows NeMo-Speech.cpp prerequisite

Nemotron requires an additional executable **nemo-speech.exe**. The existing speech.cpp installation does not provide it. Use the **official NVIDIA installer** described at https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/install.md . On Windows, its documented PowerShell installation is:

```powershell
irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 | iex
```

Inspect the installer and its dependencies before executing a downloaded script. Open a **new** terminal afterward, then confirm the binary and download the official Q8 model:

```bat
nemo-speech --version && nemo-speech doctor && nemo-speech pull nemotron-3.5
```

The optional server writes diagnostics to `logs/nemotron.log`; the reconciler writes to `logs/reconciler.log`; pipeline, VRAM and fallback details remain in `logs/run.log`. The model can take longer to initialize on its first download. CPU fallback is not automatic for Nemotron: if GPU admission fails, the optional ASR is disabled for that session while Haas remains active.

## Approximate memory plan (decimal GB)

| Component | Planning estimate of device VRAM |
| --- | ---: |
| Haas Vulkan | 0.7–1.1 |
| Nemotron 3.5 Q8_0 CUDA | 0.9–1.5 |
| LFM-JP reconciler Q4_K_M GPU | 0.9–1.4 |
| PLaMo 2B Q4_K_M 12 GPU layers | 0.5–1.1 |
| FireRedVAD, MPV, CUDA/Vulkan, desktop buffers | 0.8–1.5 |
| **Combined indicative range, RTX VSR disabled** | **3.8–6.6** |

These figures are not measured. The RTX 3070 hard policy targets at most **7.5 decimal GB** (soft shedding at 7.0, emergency at 7.3). RTX VSR, simultaneous CUDA allocations and Windows desktop usage can raise peaks. If GPU admission fails, LFM is attempted on CPU; if the optional Nemotron cannot fit, the pipeline runs Haas alone with PLaMo.

After Windows installation and checking out this branch, run `uv sync`, then `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v`, then `Start translate.bat`. The changes have not yet been measured on your RTX 3070; benchmark the same VOD passages against the previous Qwen secondary model before making it the default.

---

## Legacy notes: earlier Haas + Qwen experimental branch

The following material documents the preceding dual-ASR experiment; model names, runtime requirements and download instructions in that historical section may no longer match this Nemotron branch.

# Haas + Qwen dual Japanese ASR with PLaMo translation

Feature branch `feature/haas-dual-asr-reconciliation` uses original 16 kHz mono FireRedVAD speech segments. Haas runs first. The secondary speech.cpp worker receives the same samples, generation and media interval. Both ASRs stay resident after admission. PLaMo receives one Japanese cue and produces the English MPV subtitle.

| Role | Repository and filename | Runtime | License |
|---|---|---|---|
| Japanese primary, FastConformer/TDT | sakasegawa/Haas-ja-v1.0-GGUF / Haas-616M-ja-v1.0-Q8_0.gguf | speech.cpp 0.10.1 Vulkan | Apache-2.0 |
| Japanese secondary, Qwen3-ASR | sakasegawa/Qwen3-ASR-1.7B-GGUF / Qwen3-ASR-1.7B-Q6_K.gguf | independent speech.cpp worker, forced Japanese | Apache-2.0 |
| Japanese selector | bartowski/Qwen_Qwen3.5-0.8B-GGUF / Qwen_Qwen3.5-0.8B-Q4_K_M.gguf | llama.cpp b12318, 2048 context, full GPU offload requested when admitted | Apache-2.0 |
| English translator | pfnet/plamo-3-translate-2b-GGUF / plamo-3-translate-2b-Q4_K_M.gguf | separate PLaMo llama.cpp server | PLaMo Community License |

The speech.cpp catalog pins ASR files and checksums. The two llama.cpp processes use the shared Hugging Face cache. First launch can take several minutes. Run Setup.bat once on Windows to install the pinned executables. The console reports the active ASR and selector after admission. See logs/run.log for fallbacks and logs/reconciler.log for selector startup errors.

## Configuration and fallback

The launcher seeds a missing config.toml from config.dual-asr.example.toml. Existing configurations inherit defaults for missing sections. In [dual_asr], enabled=false selects baseline A. In [reconciler], enabled=false selects B. Enable both for C. Live defaults to selective, with a 2 s secondary timeout and a 4 s selector timeout. VOD defaults to all, with 25 s and 20 s timeouts. Both support off, selective or all. The live playback delay defaults to 20 s. Existing local overrides take precedence.

The selector compares kana, width, spacing and punctuation while preserving numeric decimals, lexical differences and particles. Different kanji readings, short reactions, negation, missing spans and glossary mismatches can trigger selection. A llama.cpp grammar restricts output to A (Haas), B (Qwen), or U (uncertain, use Haas). Python copies the original text and builds the selection record. Selector choices and elapsed request times are logged. VAD confidence describes the audio segment. The workers do not supply calibrated, comparable transcript probabilities. Baseline B keeps usable Haas when both transcripts differ and recovers a validated omission from Qwen. Stage debug logs contain both originals. Successful dual selections bypass subsequent caption spelling rewrites. Baseline caption recovery and PLaMo glossary, context and translation retries remain available.

GPU policy samples total NVIDIA device 0 usage in decimal GB, including desktop and playback allocations. Admission reservations include estimated runtime buffers and future playback/VAD headroom. The target is 6.8 GB. At 7.0 GB optional work is skipped. At 7.3 GB optional processes are released for the session. If total usage still measures at least 7.5 GB after release, playback stops. A missing counter disables optional GPU work. This sampled application policy cannot enforce a WDDM hardware limit or detect every transient spike. Assumes the RTX 3070 is NVIDIA device 0 and speech.cpp Vulkan0. PLaMo first tries its configured layer count, then a partial or CPU placement when admission requires it. The selector requests 99 GPU layers by default and reserves 0.95 GB for its weights and buffers plus 0.65 GB for playback. Startup verifies the remaining headroom and falls back to CPU if needed. Set [reconciler].gpu_layers=99 in older local configs to enable this behavior. An explicit zero keeps CPU selection.

## Paired accuracy evaluation

Use local, rights-cleared clips of identical Hololive audio for all three profiles. Include solo and overlapping speakers, rapid dialogue, names, negation, laughter, silence and music. Use one annotated interval per clip. Prepare a UTF-8 CSV with clip,reference_ja,reference_en,tags columns. Clip paths are relative to the manifest. Use empty references for verified non-speech. Run from the repository root after setup.

```bat
uv run --locked --with psutil==7.0.0 python scripts/benchmark_dual.py clips\manifest.csv results-abc
```

A/B/C JSON reports include source hashes, aligned text, CER, missed/spurious speech, sampled process-tree CPU time and RSS, device-wide mean and peak VRAM, and wall time including startup. Per-run logs retain cue timings, pipeline p95 and maximum queue depths. A failed or disabled optional model invalidates that profile. For each tagged clip manually score preserved names, polarity, speakers, meaning and omissions against the source. Parallel speakers can be missed by both recognizers.

For playback testing, repeat a saved VOD section in MPV, pause for two minutes, seek backward, and compare onset against audible speech. For live, record one session, observe clock discontinuities/reconnections, backlog and expired cue counts. Log GPU-engine time using a Windows GPU trace for Python, both speech workers and both llama servers. Verify owned PIDs exit after MPV close, Ctrl+C and console close. Hardware results, playback offsets and transcript quality require local measurement.

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
cd /d C:\Games && git clone --single-branch --branch feature/haas-dual-asr-reconciliation https://github.com/KarimPai/video-translate-main.git video-translate-dual-asr && cd /d C:\Games\video-translate-dual-asr && call Setup.bat && uv run --locked python -m unittest discover -s tests -v && call "Start translate.bat"
```

Choose video quality **1=360p, 2=480p, 3=720p (default), 4=1080p**. VODs prefer direct HTTPS/DASH sources with lower-quality fallback; live sources prefer HLS.

### Optional 480p RTX Video Super Resolution (experimental)

When selecting **2 (480p)** in `Start translate.bat`, answer **Y** to try NVIDIA RTX VSR 3x enhancement (~1440p output). It is **off by default**; other quality choices use unchanged MPV playback. Enable **Video Super Resolution** and manually choose **High** under **NVIDIA App > System > Video**; MPV cannot change NVIDIA's quality tier.

CLI: `--mpv --video-quality 480 --hwdec auto-safe --rtx-vsr`, or opt in through `[mpv] rtx_vsr = true` in your local config. Windows and MPV D3D11 hardware decoding are required. The filter is applied only after MPV confirms an input video height of 480p or less. It is removed if unavailable, hardware decoding falls back to software, or 30-second health samples find at least 15 dropped or 30 delayed frames. Manual/cache pauses are excluded from performance alerts. The underlying MPV session, native seeking, FireRedVAD, Haas ASR, PLaMo translation, and English subtitles remain unchanged.

**Verify in practice:** MPV accepting the filter does not prove NVIDIA AI enhancement is running. Check the NVIDIA App RTX Video indicator, `logs\\mpv.log`, and `logs\\run.log` for `RTX VSR playback check` or `RTX VSR disabled`. Compare the same 480p VOD at 30/60 fps with VSR on and off, and monitor VRAM via `nvidia-smi`; performance and activation are unverified until tested on your RTX 3070. For CPU-only tests run `uv run --no-sync python -m unittest discover -s tests -p "test_rtx_vsr.py" -v`.


## Accuracy-first FireRedVAD defaults (experimental)

The retest branch now prioritizes **retaining audible Japanese speech over minimum subtitle latency**. The VAD receives the original 16-kHz mono audio, without GTCRN, DeepFilterNet, or diarization.

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
- **Live:** 7 s soft / 8 s hard cap for the existing 20-second live playback delay. This is a timing safeguard, not a claim that shorter segments recognize better; live needs comparative testing.
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
- For live playback, the default delay is **20 seconds** (not a hard subtitle timestamp offset).

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
