# Video Translate — Voxtral experiment

Local Japanese-to-English subtitles for YouTube livestreams and VODs on Windows 11.

`YouTube → MPV audio → CPU FireRedVAD → Voxtral Mini 4B Q4_K → PLaMo 3 Translate 2B → English subtitles`

This experimental branch uses one Japanese speech recognizer: **Voxtral Mini 4B Realtime 2602 Q4_K**, through the Windows CUDA build of [audio.cpp](https://github.com/0xShug0/audio.cpp). Each complete VAD utterance is transcribed offline. The model stays loaded in an application-owned local server between utterances. Inference uses local loopback requests; audio is never sent to an external inference API.

## Fresh Windows installation

In CMD, clone directly into the independent experimental folder:

```cmd
git clone --single-branch --branch experiment/voxtral-mini-4b-single-asr https://github.com/KarimPai/video-translate-main.git C:\Games\video-translate-voxtral
```

Open that folder and run `Setup.bat`. Setup requires internet access, Windows x64, a current NVIDIA driver, and WinGet. It installs or verifies uv, Python 3.12, yt-dlp, Deno, FFmpeg, MPV, GUI dependencies, and CPU FireRedVAD/PyTorch. No CUDA toolkit installation is required: the pinned native CUDA builds ship their runtime DLLs.

Setup installs these components within this checkout:

| Component | Location |
| --- | --- |
| audio.cpp v0.9.5, CUDA 12.4 | `.runtime/audio-v0.9.5-cuda12/` |
| Voxtral Q4_K, approximately 3.10 GB download | `models/voxtral/voxtral-mini-4b-realtime-2602-q4_k.gguf` |
| llama.cpp b12318, CUDA 12.4 | `.runtime/llama-b12318-cuda12/` |
| PLaMo 3 Translate 2B Q4_K_M | Hugging Face shared model cache |
| Python and GUI dependencies | `.venv/` |

The installer verifies the authoritative GitHub SHA256 values for runtime archives and the pinned Hugging Face LFS SHA256 for Voxtral. A valid existing checkpoint is reused; interrupted Hugging Face downloads can resume. Model weights, runtime binaries, and local configuration are excluded from Git. FireRedVAD's small model and the optional CPU language-identification model are cached on first use.

Double-click **Start GUI.vbs** to launch the console-free GUI. Enter a YouTube URL, choose 360p/480p/720p/1080p, and start translation. **Start translate.bat** provides a terminal alternative. If a configuration is needed, copy `config.example.toml` to `config.toml`; paths are relative to this checkout. The experimental configuration schema is independent: start from the new example when moving from another branch.

## Playback, timing, and names

MPV retains hardware decoding, ordinary video scaling, fullscreen, VOD seeking, and the requested video quality. The native MPV OSC remains disabled when Qt owns playback controls, preserving the seekbar fix. YouTube Live and VOD ingestion keep their existing timing paths.

FireRedVAD retains the existing segmentation baseline and meaningful short Japanese reactions. Audio.cpp currently provides no Voxtral word timestamps, so subtitle timing comes from the original VAD utterance. Seek, stop, restart, and video-switch generation checks prevent an expired result from becoming a new subtitle. A seek invalidates the active response; the persistent server can finish that bounded request before the next request begins. Stop and timeout terminate the application's owned server, and a later session loads it again.

Voxtral's upstream model detects language automatically; the backend has no verified Japanese-language forcing option. The existing CPU spoken-language filter and Japanese text checks protect Japanese-only translation. Low-confidence short reactions are retained. Non-LLM repetition, noise, and hallucination checks reject unusable transcripts without inventing dialogue.

The Hololive glossary and context policies preserve the spoken forms: ラプ → **Lap**, ラプ様 → **Lap-sama**, ラプちゃん → **Lap-chan**, ラプラス → **Laplus**, みこ → **Miko**, みこち → **Mikochi**, わため → **Watame**. Names do not expand into full stage names. Matsuri/festival, Miko/shrine maiden, and grass/laughter require supporting dialogue context. Translation memory and neutral-pronoun safeguards remain enabled.

## RTX 3070 memory and diagnostics

The default targets an 8 GiB RTX 3070: Voxtral uses CUDA Q4_K; FireRedVAD and language identification stay on CPU. PLaMo adapts partial GPU offloading to measured free device memory and can fall back to CPU. The initial headroom target is approximately 0.7–1 GiB where feasible. Model-file size is not a runtime VRAM estimate. Allocation failures produce actionable diagnostics; the recognizer does not silently switch checkpoints or change video resolution.

The optional **Safe Playback** mode bounds audio queues and concurrent inference. The **Diagnostics** dashboard reports available GPU memory, stage timings, queue/cue counters, playback observations, and owned processes. Install NVML telemetry with `uv sync --locked --extra gui --extra diagnostics` (included in Setup). WDDM can make some counters unavailable; device memory includes other applications. Monitor the actual peak and remaining memory during a representative long session before changing offload settings.

The translation memory, relevance-selected terminology, optional YouTube chat context, and diagnostic export remain available. Chat is weak terminology evidence, never model instructions or a replacement for audio. Public live/replay chat can be missing or delayed. Diagnostic exports contain bounded log tails and optional filtered Windows event metadata; review subtitle and video metadata before sharing them.

## Windows executable

The [Windows GUI frontend build](https://github.com/KarimPai/video-translate-main/actions/workflows/windows-gui.yml) workflow builds **VideoTranslate.exe** with PyInstaller, runs the offline regression suite and Windows/Qt lifecycle checks, launches the packaged EXE to verify Qt startup, and uploads `VideoTranslate-Voxtral-Windows-GUI`. Extract the complete artifact, run its `Setup.bat`, then open `VideoTranslate.exe`. Keep the packaged `_internal` directory beside the EXE. The artifact contains the source backend and setup files; it deliberately does not contain model weights or a Python runtime environment. An EXE-only artifact is for updating an already matching complete artifact.

## Tests and physical validation

For the lightweight offline suite:

```cmd
uv sync --locked --only-group ci
uv run --no-sync ruff check --select E9,F63,F7,F82 src tests scripts
uv run --no-sync python -m compileall -q src tests scripts
set PYTHONPATH=src
uv run --no-sync python -m unittest discover -s tests -v
```

Add `--extra gui` to the regular project installation for Qt smoke tests. Offline mocked protocol tests establish request handling, failure behavior, generation resets, filters, and cleanup; they do not establish transcription accuracy or GPU compatibility. `scripts/benchmark_quality.py` compares reviewed, time-aligned Japanese references with actual output and flags English lines for human review.

On the target Windows 11 / RTX 3070 system, verify Japanese audio with known reference text, nickname accuracy, short reactions, 3–15-second utterances, live subtitle delay, forward/backward seeking, video switching, each quality setting, GPU memory peaks, and closing during inference. Approximately 20 seconds of live delay is a target, not a measured guarantee. Exercise missing runtime/model files and allocation failures, then confirm that unrelated processes survive and owned processes exit. This repository's CI cannot establish those physical hardware results.

For setup failures, inspect the console message. For runtime failures, inspect `logs/run.log`, `logs/voxtral-server.log`, `logs/plamo-server.log`, `logs/mpv.log`, and `logs/gui-startup.log`. Missing CUDA DLLs require the complete matching runtime archive; driver errors require an updated NVIDIA driver. See [GUI_SETUP.md](GUI_SETUP.md) for the GUI and packaging details.
