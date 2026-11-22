# Video Translate — Japanese → English live/VOD subtitles

Windows-oriented local video translation project with **interchangeable speech recognition and translation models**, English-only subtitles, and an embedded MPV player.

The **`main`** branch now contains the universal implementation previously developed on `video-translate-main`. The older Haas/PLaMo-only code is preserved on `backup/main-before-universal-20261122`.

## Features

- **Sources:** YouTube livestreams and VODs; the CLI also accepts local media files.
- **Playback:** MPV hardware decoding, video-quality choices (360p, 480p, 720p, 1080p), VOD seeking, pause/resume, fullscreen, and subtitle synchronization.
- **GUI:** PySide6 application with YouTube URL entry, ASR/translator selectors, embedded video, controls, and diagnostics/console views.
- **Speech detection:** Silero VAD, with adjustable splitting and buffering.
- **ASR choices:** Haas-ja-v1.0 (speech.cpp/Vulkan), Qwen3-ASR 1.7B (llama.cpp/CUDA), and Whisper Large-v3 Turbo (faster-whisper).
- **Japanese → English translation:** PLaMo 3 Translate 2B, PLaMo 3 Translate 8B, Gemma 4 E4B, or Hy-MT2 (GGUF via llama.cpp).
- **Quality safeguards:** Japanese glossary/nickname support, contextual translation, constrained retries, output filtering, and session subtitle caching.
- **Optional diagnostics:** Runtime logs and NVIDIA GPU memory observations when the required package is installed.

Models are selected independently. Available combinations depend on the installed model files, backend executables, and system memory.

## Requirements

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/).
- **Windows** for the bundled Qt/MPV embedding and process-management implementation.
- [FFmpeg](https://ffmpeg.org/), [yt-dlp](https://github.com/yt-dlp/yt-dlp), and [MPV](https://mpv.io/) installed and available to the application.
- The corresponding **speech.cpp Vulkan** runtime for Haas, **llama.cpp CUDA** `llama-server` for GGUF backends, and/or **faster-whisper** for Whisper.
- Locally available model files matching whichever ASR and translation backends are selected.
- A compatible GPU and sufficient VRAM for GPU-enabled model combinations; offloading and performance are hardware-dependent.

**Models, the `.runtime/` executables, `.venv/`, and built `VideoTranslate.exe` are not shipped in this source repository.** There is no one-click Windows installer in this branch.

## Installation

From PowerShell on Windows:

```powershell
git clone https://github.com/KarimPai/video-translate-main.git
cd video-translate-main
uv sync --extra gui --extra diagnostics --extra asr
```

Before launching:

1. Open `config.toml`. Set model and runtime paths for the selected backends. Some committed paths refer to the original developer's `C:/Games/video-translate-main/` folder and **must be adapted** on another machine.
2. Supply the Haas model and compatible speech.cpp executable if using the default ASR backend.
3. Supply the selected translation model and a compatible CUDA `llama-server` executable (or configure another supported backend).
4. Confirm that FFmpeg, MPV, and yt-dlp are installed and discoverable.

The defaults under `[models]` in `config.toml` are `haas` ASR, `plamo` translation, and `silero` VAD. Backend-specific overrides live under sections such as `[asr.qwen3-asr]`, `[asr.whisper]`, `[translate.plamo]`, and `[translate.plamo8b]`.

## Run

**GUI:**

```powershell
uv run python -m auto_translate.gui_bootstrap
```

Paste a YouTube livestream or VOD URL, select the desired backends/video quality, and start playback. The GUI starts a separate backend process and embeds MPV.

**CLI — list registered backends:**

```powershell
uv run video-translate --list-backends
```

**CLI — translate with MPV playback:**

```powershell
uv run video-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv --video-quality 720
```

**CLI — select models explicitly:**

```powershell
uv run video-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv --asr-backend haas --translate-backend plamo
```

Use `uv run video-translate --help` for playback, live/VOD, model-path, and diagnostics options. The CLI accepts a local media path as its positional source argument.

## GUI executable (optional)

`VideoTranslate.spec` describes a PyInstaller **GUI-only** bundle. It intentionally excludes the ASR, translation, and CUDA inference engines. The packaged GUI still needs a project installation, its `.venv`, external model files, and runtime executables.

For developers with PyInstaller installed:

```powershell
uv run --with pyinstaller python -m PyInstaller --noconfirm VideoTranslate.spec
```

The expected output is under `dist/VideoTranslate/`. Packaging and GPU behavior should be tested on the target Windows machine; a successful build is not a guarantee of a working model environment.

## Tests and diagnostics

Lightweight source checks:

```powershell
uv run --group ci ruff check src tests
uv run --with pytest --group ci python -m pytest -q
```

The repository also contains **integration/runtime tests** that may launch external processes or load models. Do not run the full suite while gaming or when GPU resources are constrained; select mock-based tests if you need a low-impact check. Tests requiring optional dependencies or actual model installations may need additional environment setup.

See the GUI's Console/Diagnostics area and generated `logs/` files when troubleshooting translation or MPV startup failures. Model loads can consume significant RAM, VRAM, and CPU.

## Notes and limitations

- The application is aimed at **Japanese speech → English subtitles**. Translation, nickname recognition, and subtitle timing remain model-dependent; they are not guaranteed error-free.
- Hardware acceleration and all-GPU model combinations are limited by available VRAM. Larger GGUF models may offload only some layers.
- The optional `[name_fusion]` contextual-name feature is experimental and **disabled by default**.
- A missing model, missing external executable, invalid configuration path, or incompatible GPU/runtime can prevent startup.
- No measured quality or performance claim is implied by the existence of a backend or a passing mocked unit test.

## Project layout

- `src/auto_translate/` — CLI, GUI, model registry, pipeline, configuration, and diagnostics.
- `src/auto_translate/asr/` — selectable speech-recognition backends.
- `src/auto_translate/translate/` — selectable translation backends.
- `src/auto_translate/output/` — MPV integration and English subtitles.
- `tests/` — regression and integration tests.
- `config.toml` — model selection, paths, and runtime settings.
- `glossary.toml` — Japanese/English names and terminology.
- `VideoTranslate.spec` — optional GUI-only PyInstaller configuration.
