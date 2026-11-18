# Windows GUI setup and executable

This checkout starts Qwen3-ASR-1.7B and Hy-MT2-7B using separate owned llama.cpp servers. The intended system is Windows x64 with an NVIDIA RTX 3070 8 GiB and a current driver.

## Install and launch

1. Use the local `experiment/qwen3-asr-hymt` checkout.
2. Run `Setup.bat` for dependencies, pinned llama.cpp CUDA runtime, ASR model/projector, and translation model.
3. Double-click `Start GUI.vbs` for the source frontend.
4. Enter a YouTube URL, select resolution, and start translation.

The VBS launcher uses `pythonw.exe`; native helpers suppress external console windows. Setup retains a console for installation errors. Startup failures are recorded in `logs/gui-startup.log`.

## Controls and settings

Pause/resume, VOD seeking, fullscreen, and 360p/480p/720p/1080p selection remain available. Qt owns the seekbar; MPV hardware decoding and scaling remain enabled. The Console tab retains transcript/log output.

Settings exposes the ASR GGUF path, Hy-MT2 GGUF path, and 0–20 context lines (default six). Changes apply at the next start. Seeking or switching videos clears recent dialogue context. The ASR projector and server executable are configured in `config.toml`, not inferred from arbitrary selected model paths.

Safe Playback bounds queued audio and inference concurrency. Diagnostics displays observed GPU memory/activity, ASR/translation timings, playback facts, queue counts, and owned workers. Missing metrics remain unknown. Diagnostic export runs in the background; Windows event metadata is opt-in.

## Configuration

Copy `config.example.toml` to `config.toml` when creating a new installation. Old Voxtral/PLaMo fields are obsolete.

| Setting | Default |
| --- | --- |
| `asr.model_path` | `models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf` |
| `asr.mmproj_path` | `models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` |
| `asr.server_path`, `translate.server_path` | `.runtime/llama-b12318-cuda12/llama-server.exe` |
| `translate.model_path` | `models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf` |
| `translate.gpu_layers` | `-1` (measured adaptive offload) |
| `translate.context_lines` | `6` Japanese dialogue lines |
| `translate.max_context_chars` | `600` (includes stream title, channel, host, cast and chapter) |
| `translate.context_max_gap_s` | `15.0` seconds of audio-timeline silence |
| `translate.context_english` | `false` (opt in to show earlier English in context) |
| `translate.style` | `简洁直白` (Hy-MT2 style prompt: plain and concise; `""` disables) |
| `translate.merge_fragments` | `true` (join a mid-sentence cue with the next cue within `merge_max_gap_s` = `0.6`; live holds it until playback is `live_merge_margin_s` = `1.5` from its onset) |
| `mpv.reading_cps`, `mpv.max_subtitle_duration_s` | `15` characters/second, capped at `7.0` seconds (minimum `1.5`) |
| `mpv.en_size`, `mpv.background_opacity` | `32` (relative to a 720-pixel-high screen), `0.3` dark box behind the text (`0` = outline only) |
| `mpv.max_line_chars`, `mpv.max_cue_lines`, `mpv.max_subtitle_rows` | `42`, `2` (longer cues split into timed chunks), `4` wrapped rows |
| `holodex.api_key` | `""` (optional; or `HOLODEX_API_KEY`, adds the tagged collab cast) |
| `asr.recent_context_lines` | `2` previous transcripts in the ASR prompt |
| `vad.adaptive_split_s` | `5.0` (VOD natural-pause split target) |
| `asr.response_timeout_s` | `30` |
| `asr.live_response_timeout_s` | `15` |
| `vad.device` | `cpu` |
| `vad.threshold`, `vad.neg_threshold` | `0.24`, `0.16` (speech onset/continuation probability) |
| `vad.min_speech_ms` | `120` voiced milliseconds |
| `vad.short_reply_ms`, `vad.short_reply_probability` | `80` sustained milliseconds at smoothed probability `0.85` for shorter replies (not an 80 ms raw-audio guarantee) |
| `vad.onset_ms`, `vad.prob_smooth_frames` | `20` ms, `3` ten-millisecond frames |
| `vad.speech_pad_ms`, `vad.min_silence_ms` | `400` ms padding, `550` ms end-of-phrase silence |
| `vad.fallback_s` | `0.0` (no ASR probes of VAD-negative audio) |

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

Inspect `logs/asr-server.log` and `logs/mt-server.log` for loading/request failures. Verify the complete llama.cpp CUDA runtime, Microsoft Visual C++ x64 Redistributable, and NVIDIA driver. Let Hy-MT2 reduce GPU layers or set a lower fixed count if unrelated workloads consume memory; model download size is not VRAM usage.

Verify real audio, live timing, VOD seeks, video switching, and stopping during model startup/inference. The application should terminate only its owned backend, MPV, and model servers. A successful Qt launch or synthetic model probe does not establish real Hololive accuracy or sustained playback reliability.
