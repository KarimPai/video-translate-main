# Video Translate — Haas-ja-v1.0 + Gemma 4 E4B

Local Japanese-to-English subtitles for YouTube livestreams and VODs on Windows.

`YouTube → FFmpeg 16 kHz audio → CPU FireRedVAD → Haas-ja-v1.0 (speech.cpp) → Gemma 4 E4B (llama.cpp) → MPV subtitles`

This branch uses speech.cpp for Haas-ja-v1.0 ASR (Vulkan) and llama.cpp for Gemma 4 E4B translation (CUDA). Audio and inference stay local; YouTube access and installation downloads require internet access.

## Install and run

Use this checkout on Windows x64 with a current NVIDIA driver. Install dependencies via `scripts/Setup-Windows.ps1` or the CLI, then launch `VideoTranslate.exe` from `dist/VideoTranslate/`.

| Component | Location |
| --- | --- |
| speech.cpp (Haas-ja-v1.0 built-in) | `.runtime/speech-cpp/` |
| llama.cpp CUDA 12.4 | `.runtime/llama-b12318-cuda12/` |
| Gemma 4 E4B Q4_K_M, 5.15 GB | `models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf` |
| Python environment | `.venv/` |

Copy `config.example.toml` to `config.toml` for local overrides. CLI example:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mpv --no-web --context-lines 3
```

Model overrides are `--asr-model`, `--mt-model`, and `--llama-server`. The GUI exposes ASR/translation model paths and 0–20 previous dialogue lines; server paths remain in configuration.

## Accuracy and context

Haas-ja-v1.0 is a Japanese-only ASR model — no language tagging is needed. Empty, implausible, repeated, and token-truncated results are rejected. Quiet speech receives at most 18 dB (8×) gain with peak limiting; the near-silence gate uses 10 ms RMS windows so silent VAD padding does not hide a short reply.

Gemma 4 E4B translates directly to natural, meaning-first English. Relevant glossary terms, the stream title/channel/host/cast/chapter, and the most recent six informative Japanese lines guide translation within a 600-character budget. Dialogue resets on seeking, switching videos, or more than 15 seconds of audio-timeline silence. Japanese names, nicknames, slang, emotion, and profanity are preserved. Invalid English or a missing expected name gets one context-free retry; still-invalid output is suppressed.

FireRedVAD runs on the original audio on CPU. Its sensitivity defaults are `threshold = 0.24`, `neg_threshold = 0.16`, a 120 ms voiced minimum, and an 80 ms sustained high-confidence exception (`0.85`) for short replies. These durations count smoothed VAD-positive frames, not raw speech length. A 20 ms onset with three-frame smoothing catches briefer starts; 400 ms padding and 550 ms end silence remain unchanged. Lower thresholds also admit more noise, so VAD-negative probing stays disabled and ASR output guards remain enabled. Local `[vad]` settings override these defaults; restart translation after changing them.

## Timing and GPU memory

VAD utterance timestamps determine subtitle timing. VOD seeks invalidate pending generations and use a session-local subtitle cache. Livestream MPV/FFmpeg share a local HLS media timeline. MPV keeps hardware decoding, the requested quality, pause/fullscreen, and VOD seeking; Qt owns the seekbar and MPV's competing OSC is disabled.

Haas (speech.cpp, Vulkan) loads first, then Gemma 4 E4B (llama.cpp, CUDA) estimates initial offload from free VRAM. Gemma 4 E4B Q4_K_M uses ~5 GB VRAM; Haas uses ~0.6 GB. The default headroom target is 1.0 GB on the intended RTX 3070 8 GiB. Further pressure requests reduce translation offload between cues; CPU fallback is available.

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

The packaged EXE is at `dist/VideoTranslate/VideoTranslate.exe`. To rebuild after source edits:

```cmd
uv pip install pyinstaller
uv run --no-sync python -m PyInstaller --noconfirm --clean --onedir --windowed --name VideoTranslate --paths src VideoTranslate.spec
```

Keep the complete `dist/VideoTranslate/` directory with its `_internal` folder. No CI publication or remote push is required for local use.

Inspect `logs/run.log`, `logs/asr-server.log`, `logs/mt-server.log`, `logs/mpv.log`, and `logs/gui-startup.log`. Missing native DLLs require the complete matching runtime; driver errors require a current NVIDIA driver. Reduce unrelated GPU workloads or explicitly lower `translate.gpu_layers` if headroom is insufficient.
