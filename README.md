# Video Translate — Qwen3-ASR + Hy-MT2

Local Japanese-to-English subtitles for YouTube livestreams and VODs on Windows.

`YouTube → FFmpeg 16 kHz audio → CPU FireRedVAD → Qwen3-ASR-1.7B → Hy-MT2-7B → MPV subtitles`

The `experiment/qwen3-asr-hymt` branch uses two application-owned llama.cpp servers on loopback. Audio and inference stay local; YouTube access and installation downloads require internet access. This branch replaces Voxtral/audio.cpp and PLaMo, including their name-repair, caption, chat, and extended-memory heuristics.

## Install and run

Use this checkout on Windows x64 with a current NVIDIA driver and WinGet. Run **Setup.bat**, then **Start GUI.vbs**. Enter a YouTube URL, choose 360p/480p/720p/1080p, and start. **Start translate.bat** is the terminal alternative.

Setup installs uv, Python 3.12, yt-dlp, Deno, FFmpeg, MPV, CPU FireRedVAD, Qt, and optional NVML diagnostics. The pinned llama.cpp CUDA archives include runtime DLLs; a separate CUDA toolkit is not needed.

| Component | Location |
| --- | --- |
| llama.cpp b12318 CUDA 12.4 | `.runtime/llama-b12318-cuda12/` |
| Qwen3-ASR-1.7B Q8_0, 2.17 GB | `models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf` |
| Qwen3-ASR audio projector Q8_0, 356 MB | `models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` |
| Hy-MT2-7B Q4_K_M, 4.62 GB | `models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf` |
| Python environment | `.venv/` |

Setup verifies pinned GitHub runtime digests and Hugging Face model revisions/SHA256. Valid local weights are reused. Models, runtime binaries, and local configuration are not committed. Existing weights can be verified separately with `uv run --no-sync python -m auto_translate.models`.

Copy `config.example.toml` to `config.toml` for local overrides. This branch has a new schema; the old Voxtral/PLaMo settings no longer apply. CLI example:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mpv --no-web --context-lines 3
```

Model overrides are `--asr-model`, `--asr-mmproj`, `--mt-model`, and `--llama-server`. The GUI exposes ASR/translation model paths and 0–20 previous dialogue lines; the projector/server paths remain in configuration.

## Accuracy and context

Qwen3-ASR receives optional `asr.context` and metadata-selected canonical names with one spoken alias each, capped at four talents/eight hints. Raw titles and the full nickname dictionary are not sent as recognition hints because they can be echoed as speech; the title still guides translation. Its language tag gates Japanese output; kana-bearing Japanese mislabelled Chinese is retained. Empty, implausible, repeated, context-echoed, and token-truncated results are rejected instead of inventing speech. Quiet speech receives at most 18 dB (8×) gain with peak limiting; the near-silence gate uses 10 ms RMS windows so silent VAD padding does not hide a short reply. The optional CPU language filter defaults off because the ASR model already identifies language.

Hy-MT2-7B uses Tencent's official default, terminology, background-information, and style templates, combined when present, without a system prompt. The default style `简洁直白` asks for plain, concise English (`translate.style = ""` disables it). Sampling follows Tencent's 7B recommendation (top-k 20, top-p 0.6, repetition penalty 1.05) at temperature 0.2 instead of 0.7 for subtitle stability. Relevant glossary terms, the stream title/channel/host/cast/chapter, and the most recent six informative Japanese lines guide translation within a 600-character budget; previous English is excluded by default. Dialogue resets on seeking, switching videos, or more than 15 seconds of audio-timeline silence, not a playback pause. Simple acknowledgements do not displace useful context. Kana spelling variants share declared talent identities, while ambiguous bare Botan/Miko/Matsuri hints still require that talent in context; clear button, festival, shrine, and ship uses remain ordinary words. Recognized registered names take priority in the glossary hints. Invalid English or a missing expected name gets one terminology-preserving, context-free retry; still-invalid output is suppressed rather than rewritten. No deterministic name substitution or pronoun rewrite is applied. Text benchmarks still require human review for meaning and naturalness.

FireRedVAD runs on the original audio on CPU. Its sensitivity defaults are `threshold = 0.24`, `neg_threshold = 0.16`, a 120 ms voiced minimum, and an 80 ms sustained high-confidence exception (`0.85`) for short replies. These durations count smoothed VAD-positive frames, not raw speech length. A 20 ms onset with three-frame smoothing catches briefer starts; 400 ms padding and 550 ms end silence remain unchanged. Lower thresholds also admit more noise, so VAD-negative probing stays disabled and ASR output guards remain enabled. Local `[vad]` settings override these defaults; restart translation after changing them.

## Timing and GPU memory

VAD utterance timestamps determine subtitle timing. VOD seeks invalidate pending generations and use a session-local subtitle cache. Livestream MPV/FFmpeg share a local HLS media timeline. MPV keeps hardware decoding, the requested quality, pause/fullscreen, and VOD seeking; Qt owns the seekbar and MPV's competing OSC is disabled.

ASR loads first. Hy-MT2 estimates initial offload from free VRAM with a CUDA-workspace reserve, warms both dialogue-sized and short prompts, then trims layers against measured device free memory. It avoids probing full offload on an already crowded GPU, which can page out idle ASR weights under Windows. The default headroom target is 1.0 GB on the intended RTX 3070 8 GiB. Further pressure requests reduce translation offload between cues; CPU fallback is available. Model download size is not a runtime VRAM estimate, and other GPU applications affect headroom.

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
