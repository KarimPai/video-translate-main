# Video Translate — Dual Haas + Whisper Turbo ASR / Hy-MT2

Local Japanese-to-English subtitles for YouTube livestreams and VODs on Windows.

`YouTube → FFmpeg 16 kHz audio → CPU FireRedVAD → Haas (Vulkan) + Whisper Turbo (CUDA) → evidence-based selection → HY-MT2 7B (llama.cpp) → MPV subtitles`

This branch starts from `feature/hymt-context-improvements` and replaces its *default* Qwen recognition path with the earlier dual Japanese ASR approach. Qwen3-ASR is retained as an optional `asr.backend = "qwen"` setting for comparison. The HY-MT2 translator, Japanese context, glossary, video sync, and Qt GUI are preserved. Dual ASR uses a deterministic agreement/evidence selector rather than loading the older LFM reconciler; it does not stitch together unsupported words from different hypotheses.

## Install and run

Use this checkout on Windows x64 with a current NVIDIA driver and WinGet. Run **Setup.bat**, then **Start GUI.vbs**. Enter a YouTube URL, choose 360p/480p/720p/1080p, and start. **Start translate.bat** is the terminal alternative.

Setup installs uv, Python 3.12, yt-dlp, Deno, FFmpeg, MPV, CPU FireRedVAD, faster-whisper, Qt, and optional NVML diagnostics. Haas also requires your existing speech.cpp Vulkan `speech.exe` on PATH. The pinned llama.cpp CUDA archives include runtime DLLs; a separate CUDA toolkit is not needed.

| Component | Location |
| --- | --- |
| llama.cpp b12318 CUDA 12.4 | `.runtime/llama-b12318-cuda12/` |
| Haas-ja-v1.0 | Reuse an installed speech.cpp model (`asr.haas_model`) |
| Whisper Large-v3 Turbo | Reuse compatible existing HF cache or absolute CTranslate2 directory (`asr.secondary_model`) |
| Hy-MT2-7B Q4_K_M, 4.62 GB | `models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf` |
| Python environment | `.venv/` |

Setup verifies pinned GitHub runtime digests and Hugging Face model revisions/SHA256. Valid local weights are reused. Models, runtime binaries, and local configuration are not committed. Existing weights can be verified separately with `uv run --no-sync python -m auto_translate.models`.

Copy `config.example.toml` to `config.toml` for local overrides. This branch has a new schema; the old Voxtral/PLaMo settings no longer apply. CLI example:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mpv --no-web --context-lines 3
```

Model overrides are `--asr-model`, `--asr-mmproj`, `--mt-model`, and `--llama-server`. The GUI exposes ASR/translation model paths and 0–20 previous dialogue lines; the projector/server paths remain in configuration.

## Accuracy and context

When explicitly enabled, Qwen3-ASR receives optional `asr.context` and metadata-selected canonical names with one spoken alias each, capped at four talents/eight hints. Raw titles and the full nickname dictionary are not sent as recognition hints because they can be echoed as speech; the title still guides translation. Its language tag gates Japanese output; kana-bearing Japanese mislabelled Chinese is retained. Empty, implausible, repeated, context-echoed, and token-truncated results are rejected instead of inventing speech. Quiet speech receives at most 18 dB (8×) gain with peak limiting; the near-silence gate uses 10 ms RMS windows so silent VAD padding does not hide a short reply. The CPU language filter is enabled in the supplied dual-ASR sample, but only loads an existing cache and fails open if unavailable; it never downloads a model without permission.

Hy-MT2-7B uses Tencent's official default, terminology, background-information, and style templates, combined when present, without a system prompt. The default style `简洁直白` asks for plain, concise English (`translate.style = ""` disables it). Sampling follows Tencent's 7B recommendation (top-k 20, top-p 0.6, repetition penalty 1.05) at temperature 0.2 instead of 0.7 for subtitle stability. Relevant glossary terms, the stream title/channel/host/cast/chapter, and the most recent six informative Japanese lines guide translation within a 600-character budget; previous English is excluded by default. Dialogue resets on seeking, switching videos, or more than 15 seconds of audio-timeline silence, not a playback pause. Simple acknowledgements do not displace useful context. Kana spelling variants share declared talent identities, while ambiguous bare Botan/Miko/Matsuri hints still require that talent in context; clear button, festival, shrine, and ship uses remain ordinary words. Recognized registered names take priority in the glossary hints. Invalid English or a missing expected name gets one terminology-preserving, context-free retry; still-invalid output is suppressed rather than rewritten. No deterministic name substitution or pronoun rewrite is applied. Text benchmarks still require human review for meaning and naturalness.

FireRedVAD runs on the original audio on CPU. Its sensitivity defaults are `threshold = 0.24`, `neg_threshold = 0.16`, a 120 ms voiced minimum, and an 80 ms sustained high-confidence exception (`0.85`) for short replies. These durations count smoothed VAD-positive frames, not raw speech length. A 20 ms onset with three-frame smoothing catches briefer starts; 400 ms padding and 550 ms end silence remain unchanged. Lower thresholds also admit more noise, so VAD-negative probing stays disabled and ASR output guards remain enabled. Local `[vad]` settings override these defaults; restart translation after changing them.

## Timing and GPU memory

VAD utterance timestamps determine subtitle timing. VOD seeks invalidate pending generations and use a session-local subtitle cache. Livestream MPV/FFmpeg share a local HLS media timeline. MPV keeps hardware decoding, the requested quality, pause/fullscreen, and VOD seeking; Qt owns the seekbar and MPV's competing OSC is disabled.

Both ASR models load first. HY-MT2 estimates initial offload from free VRAM with a CUDA-workspace reserve, warms both dialogue-sized and short prompts, then trims layers against measured device free memory. It avoids probing full offload on an already crowded GPU, which can page out idle ASR weights under Windows. The default headroom target is 1.0 GB on the intended RTX 3070 8 GiB. Further pressure requests reduce translation offload between cues; CPU fallback is available. Model download size is not a runtime VRAM estimate, and other GPU applications affect headroom.

Optional **Safe Playback** bounds audio read-ahead and inference concurrency without changing video resolution or decoding. **Diagnostics** displays measured GPU, ASR/translation timings, queue/cue counts, and owned process status; unsupported counters remain unknown. Diagnostic exports can contain transcript and video metadata, so review them before sharing.

## Tests and benchmark

```cmd
set PYTHONPATH=src
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m compileall -q src tests scripts
```

The offline suite covers request construction, language/output guards, context/glossary behaviour, failure recovery, generation resets, playback, Qt settings, and cleanup. It does not establish acoustic accuracy or long-session GPU reliability.

Run `uv run --no-sync python scripts/benchmark.py cases.json --output results.json` with reviewed, segmented PCM WAV clips. The manifest is a JSON list; each entry has `audio` (relative to the manifest), `ja` (reference), optional `en` (human reference), optional `title`, and optional `names` (Japanese-to-English mapping). Audio must be mono 16-bit PCM at 16 kHz, with each clip at most 60 seconds. Results contain ASR character error rate, expected-name recall, per-stage latency/real-time factor, and both English texts for human review. This is an opt-in transcript export, unlike the in-memory playback cache.

Validate representative real Hololive clips, short reactions, radio audio, VOD seeks, live clocks, video switching, and closing during inference on the target hardware. Synthetic probes and mocked tests are not proof of real-world accuracy or a guaranteed live delay.

## GUI build and troubleshooting

A preexisting packaged EXE is not rebuilt by source edits. Launch **Start GUI.vbs** to use this checkout, or rebuild the frontend as described in [GUI_SETUP.md](GUI_SETUP.md). Keep the matching backend and `_internal` directory with a packaged executable. No CI publication or remote push is required for local use.

Inspect `logs/run.log`, `logs/asr-server.log`, `logs/mt-server.log`, `logs/mpv.log`, and `logs/gui-startup.log`. Missing native DLLs require the complete matching runtime; driver errors require a current NVIDIA driver. Reduce unrelated GPU workloads or explicitly lower `translate.gpu_layers` if headroom is insufficient.


## Dual-ASR settings and limitations

- `asr.backend = "dual"` defaults to Haas + Whisper Turbo. Use `asr.backend = "qwen"` to run the preserved Qwen backend.
- Haas uses `asr.haas_model = "haas-ja-v1.0"` and `asr.haas_device = "Vulkan0"` via the existing `speech.exe` worker.
- Whisper uses the cached `h2oai/faster-whisper-large-v3-turbo` CTranslate2 model with beam 6 and `int8_float16` by default. Supply an absolute directory when its original HF cache is elsewhere. **Weights are not downloaded automatically by Whisper.**
- `secondary_live_mode = "selective"` checks empty/short Haas lines with Whisper; `secondary_vod_mode = "all"` checks every VOD utterance. Set either to `all`, `selective`, or `off`.
- If Whisper startup fails, Haas continues with a warning and the diagnostics worker status says Haas only. A timeout retires the Whisper child rather than letting a stale reply contaminate new audio.
- The selector uses transcript plausibility, agreement and Whisper's **model-internal** acoustic evidence. It is deliberately conservative on unresolved disagreements; an unsupported correction is not inserted into subtitles.
- `Safe Playback` is on by default in this branch; it serializes **actual ASR and translation inference**, bounds queued audio and preserves MPV's hardware decode. Mixed Vulkan/CUDA workloads must still be tested on Windows.
- At startup HY-MT2 searches the configured model path, known `C:\Games\video-translate*\models` locations, and the HF cache for the **exact verified Q4_K_M model**; it does not duplicate verified weights. Qwen weights are only installed when the optional Qwen backend is chosen.
- The new source branch is not a rebuilt Windows EXE. After pulling, launch the source GUI or rebuild the EXE with the existing spec. GitHub CI can run offline tests, not RTX 3070 acoustic/inference tests.

Quiet Japanese dialogue receives optional bounded 10 ms-frame gain before both ASR decoders (`asr.normalize_quiet_audio = true`); a VAD-negative probe is never amplified. The offline benchmark supports dual mode without inserting metadata as artificial speech context.
