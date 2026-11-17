# Windows GUI setup and executable

This checkout starts Qwen3-ASR-1.7B and HY-MT1.5-7B using separate owned llama.cpp servers. The intended system is Windows x64 with an NVIDIA RTX 3070 8 GiB and a current driver.

## Install and launch

1. Use the local `experiment/qwen3-asr-hymt` checkout.
2. Run `Setup.bat` for dependencies, pinned llama.cpp CUDA runtime, ASR model/projector, and translation model.
3. Double-click `Start GUI.vbs` for the source frontend.
4. Enter a YouTube URL, select resolution, and start translation.

The VBS launcher uses `pythonw.exe`; native helpers suppress external console windows. Setup retains a console for installation errors. Startup failures are recorded in `logs/gui-startup.log`.

## Controls and settings

Pause/resume, VOD seeking, fullscreen, and 360p/480p/720p/1080p selection remain available. Qt owns the seekbar; MPV hardware decoding and scaling remain enabled. The Console tab retains transcript/log output.

Settings exposes the ASR GGUF path, HY-MT GGUF path, and 0–20 context lines (default three). Changes apply at the next start. Seeking or switching videos clears recent dialogue context. The ASR projector and server executable are configured in `config.toml`, not inferred from arbitrary selected model paths.

Safe Playback bounds queued audio and inference concurrency. Diagnostics displays observed GPU memory/activity, ASR/translation timings, playback facts, queue counts, and owned workers. Missing metrics remain unknown. Diagnostic export runs in the background; Windows event metadata is opt-in.

## Configuration

Copy `config.example.toml` to `config.toml` when creating a new installation. Old Voxtral/PLaMo fields are obsolete.

| Setting | Default |
| --- | --- |
| `asr.model_path` | `models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf` |
| `asr.mmproj_path` | `models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` |
| `asr.server_path`, `translate.server_path` | `.runtime/llama-b12318-cuda12/llama-server.exe` |
| `translate.model_path` | `models/hy-mt/HY-MT1.5-7B-Q4_K_M.gguf` |
| `translate.gpu_layers` | `-1` (measured adaptive offload) |
| `translate.context_lines` | `3` Japanese dialogue lines |
| `translate.max_context_chars` | `300` |
| `translate.context_max_gap_s` | `15.0` seconds of audio-timeline silence |
| `vad.adaptive_split_s` | `5.0` (VOD natural-pause split target) |
| `asr.response_timeout_s` | `30` |
| `asr.live_response_timeout_s` | `15` |
| `vad.device` | `cpu` |

Paths are relative to this checkout. Setup records discovered MPV/runtime paths in `.cache/` and verifies model revisions/hashes.

## Build

The preexisting `VideoTranslate.exe` is not automatically updated by this branch. To build a matching frontend, install the GUI extras and PyInstaller, then run from this project:

```cmd
uv sync --locked --extra gui --extra diagnostics
uv pip install pyinstaller
uv run --no-sync python -m PyInstaller --noconfirm --clean --onedir --windowed --name VideoTranslate --paths src --hidden-import auto_translate.runtime_diagnostics --hidden-import auto_translate.gui src/auto_translate/gui_bootstrap.py
```

Keep the complete `dist/VideoTranslate/` directory inside the installed checkout, including `_internal`. The frontend deliberately does not embed model weights or the backend Python environment. A complete distribution also needs matching source/setup files, glossary, configuration, web assets, and dependency lockfile.

## Troubleshooting and validation

Inspect `logs/asr-server.log` and `logs/mt-server.log` for loading/request failures. Verify the complete llama.cpp CUDA runtime, Microsoft Visual C++ x64 Redistributable, and NVIDIA driver. Let HY-MT reduce GPU layers or set a lower fixed count if unrelated workloads consume memory; model download size is not VRAM usage.

Verify real audio, live timing, VOD seeks, video switching, and stopping during model startup/inference. The application should terminate only its owned backend, MPV, and model servers. A successful Qt launch or synthetic model probe does not establish real Hololive accuracy or sustained playback reliability.
