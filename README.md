# Video Translate — Japanese → English live/VOD subtitles

Windows-oriented local video translation project with **interchangeable speech recognition and translation models**, English-only subtitles, and an embedded MPV player.

[![CI](https://github.com/KarimPai/video-translate-main/actions/workflows/universal-checks.yml/badge.svg?branch=main)](https://github.com/KarimPai/video-translate-main/actions/workflows/universal-checks.yml)

The `main` branch is the active implementation. Models run locally; YouTube playback requires network access.

## Features

- **Sources:** YouTube livestreams and VODs; the CLI also accepts local media files.
- **Playback:** MPV hardware decoding, video-quality choices (360p, 480p, 720p, 1080p), VOD seeking, pause/resume, fullscreen, and subtitle synchronization.
- **GUI:** PySide6 application with YouTube URL entry, ASR/translator selectors, embedded video, controls, and diagnostics/console views.
- **Speech detection:** Silero VAD, with adjustable splitting and buffering.
- **ASR and translation:** Independently selectable local models (see the backend table below).
- **Quality safeguards:** Japanese glossary/nickname support, contextual translation, constrained retries, output filtering, and session subtitle caching.
- **Optional diagnostics:** Runtime logs and NVIDIA GPU memory observations when the required package is installed.

## Supported backends

| Type | Backend ID | Model / runtime |
| --- | --- | --- |
| ASR | `haas` (default) | Haas-ja-v1.0 / speech.cpp (Vulkan) |
| ASR | `qwen3-asr` | Qwen3-ASR 1.7B / llama.cpp (CUDA), matching audio projector required |
| ASR | `whisper` | Whisper Large-v3 Turbo / faster-whisper |
| Translation | `gemma4` (default) | Gemma 4 E4B / llama.cpp |
| Translation | `plamo` | PLaMo 3 Translate 2B / llama.cpp |
| Translation | `plamo8b` | PLaMo 3 Translate 8B / llama.cpp |
| Translation | `hymt` | Hy-MT2 7B / llama.cpp |

Available combinations depend on installed model files, compatible executables, and available memory. The defaults above refer to the repository configuration, not necessarily your private `config.toml`.

```text
Video/audio → Silero VAD → selected ASR → selected translator → English subtitles
                                                               ↓
                                                     MPV / terminal / web
```

This is a single-ASR pipeline: choosing one backend does not automatically run another for reconciliation.

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

1. If you do not already have a local configuration, run `Copy-Item config.example.toml config.toml` in PowerShell (the GUI also creates it if absent). Set model and runtime paths for your machine. Keep `config.toml` private and untracked; do not overwrite your existing settings during updates.
2. Supply the Haas model and compatible speech.cpp executable if using the default ASR backend.
3. For GGUF backends, extract the official [llama.cpp b12719 Windows x64 CUDA 12.4 binary](https://github.com/ggml-org/llama.cpp/releases/download/b12719/llama-b12719-bin-win-cuda-12.4-x64.zip) and its matching [CUDA runtime DLL archive](https://github.com/ggml-org/llama.cpp/releases/download/b12719/cudart-llama-bin-win-cuda-12.4-x64.zip) into `.runtime/llama-b12719-cuda12/`; keep the DLLs beside `llama-server.exe`. This is the configured default. Preserve older versioned runtime folders for rollback by changing `server_path` in `config.toml`.
4. Confirm that FFmpeg, MPV, and yt-dlp are installed and discoverable.

## Configuration

Use [`config.example.toml`](config.example.toml) as the reference for supported settings. Backend-specific overrides belong under `[asr.<backend>]` and `[translate.<backend>]`.

For example, to select Qwen3-ASR and PLaMo 2B in an existing local configuration, update its `[models]` section:

```toml
[models]
asr_backend = "qwen3-asr"
translate_backend = "plamo"
```

- Configure `[asr.qwen3-asr]` with a compatible GGUF model, its matching `mmproj_path`, and server executable. If multiple checkpoints exist, explicitly select the matching model/projector pair. Japanese decoding (`language = "ja"`) is the default; set `language = "auto"` for automatic detection if English dialogue or game audio produces false Japanese text.
- Configure `[translate.plamo]` with the desired PLaMo 2B model and compatible llama.cpp runtime. PLaMo 8B uses `[translate.plamo8b]`.
- Both PLaMo backends use native translation style without a custom style prompt; glossary, context and decoding settings are retained.
- Silero VAD runs on CPU by default. Adaptive thresholding is optional; fixed thresholding is the default.
- `glossary.toml` stores names and terminology. GPU and playback settings can be adjusted in the local configuration.

## Run

**GUI:**

```powershell
uv run python -m auto_translate.gui_bootstrap
```

Paste a YouTube livestream or VOD URL, select the desired backends/video quality, and start playback. The GUI starts a separate backend process and embeds MPV.

**CLI — list registered backends:**

```powershell
uv run video-translate --list-backends
uv run video-translate --list-models
```

**CLI — translate with MPV playback:**

```powershell
uv run video-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv --video-quality 720
```

**CLI — select models explicitly:**

```powershell
uv run video-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv --asr-backend qwen3-asr --translate-backend plamo
```

Use `uv run video-translate --help` for playback, live/VOD, model-path, and diagnostics options. The CLI accepts a local media path as its positional source argument.

## GUI executable (optional)

`VideoTranslate.spec` describes a PyInstaller **GUI-only** bundle. It intentionally excludes the ASR, translation, and CUDA inference engines. The packaged GUI still needs a project installation, its `.venv`, external model files, and runtime executables.

For developers with PyInstaller installed:

```powershell
uv run --with pyinstaller python -m PyInstaller --noconfirm VideoTranslate.spec
```

The expected output is under `dist/VideoTranslate/`. Packaging and GPU behavior should be tested on the target Windows machine; a successful build is not a guarantee of a working model environment.

## Development and tests

Routine CPU-oriented regression checks (PowerShell):

```powershell
Remove-Item Env:VIDEO_TRANSLATE_RUN_REAL_MODELS -ErrorAction SilentlyContinue
uv sync --locked --group ci --extra gui --extra asr
uv run --locked --group ci ruff check src tests
uv run --locked --group ci python -m compileall -q src tests
uv run --locked --group ci --extra gui --extra asr python -m pytest -q
```

GitHub Actions runs lint and compilation checks, plus CPU test suites on Windows and Ubuntu for pushes and pull requests to `main`. The default pytest suite excludes the two real-model test modules; `tests/conftest.py` also blocks real llama-server process launches unless explicitly enabled.

Real-model integration tests are optional and require the appropriate installed models, executables and sufficient GPU memory. Run them intentionally and separately from routine checks:

```powershell
$env:VIDEO_TRANSLATE_RUN_REAL_MODELS = "1"
try {
    uv run --group ci --extra gui --extra asr python -m pytest -q tests/test_model_switching.py tests/test_e2e_runtime.py
} finally {
    Remove-Item Env:VIDEO_TRANSLATE_RUN_REAL_MODELS -ErrorAction SilentlyContinue
}
```

Do not enable real-model tests when GPU resources are constrained. CPU tests do not establish real-world translation quality, playback stability or VRAM requirements.

## Benchmarks and quality evaluation

Benchmarks are separate from routine regression tests and normal playback. They use fixed clips and model settings to compare ASR, translation and segmentation.

- [holoGTA stress-test benchmark](docs/HOLOGTA_BENCHMARK.md): repeatable full-pipeline comparisons.
- [Japanese ASR reference workflow](docs/HOLOGTA_ASR_REFERENCE.md): 30 fixed review windows and human-verified CER scoring.
- [Qwen checkpoint comparison](docs/QWEN_JA_AB.md): standard versus Japanese fine-tuned Qwen ASR.
- [Additional controlled experiments](docs/): vocabulary, segmentation, crosstalk and truncation recovery.

Generated benchmark runs are ignored by Git. Preserve the reference manifest and manually verified transcripts. Cue counts, model agreement and output length alone are not accuracy measurements.

## Diagnostics

Use the GUI Console/Diagnostics views and generated `logs/` files to investigate playback, model startup and translation failures. Model loading can consume significant RAM, VRAM and CPU. Optional NVIDIA GPU observations require the diagnostics dependency.

## Notes and limitations

- The application is aimed at **Japanese speech → English subtitles**. Translation, nickname recognition, and subtitle timing remain model-dependent; they are not guaranteed error-free.
- Hardware acceleration and all-GPU model combinations are limited by available VRAM. Larger GGUF models may offload only some layers.
- The optional `[name_fusion]` contextual-name feature is experimental and **disabled by default**.
- A missing model, missing external executable, invalid configuration path, or incompatible GPU/runtime can prevent startup.
- No measured quality or performance claim is implied by the existence of a backend or a passing mocked unit test.

## Project layout

```text
src/auto_translate/
  asr/                Speech recognition backends
  translate/          Translation backends and shared text checks
  vad/                Speech detection
  output/             MPV, terminal and web subtitle output
  ...                 GUI, pipeline, configuration and runtime management
tests/                Automated regression and optional integration tests
scripts/              Benchmark tools and shared evaluation metrics
benchmarks/reference/ Fixed evaluation-window manifest
benchmarks/runs/      Local generated benchmark results (ignored)
docs/                 Evaluation guides and experiment reports
config.example.toml   Example configuration
glossary.toml         Names and terminology
VideoTranslate.spec   Optional GUI-only packaging
pyproject.toml        Dependencies and tooling
uv.lock               Locked dependencies
```

Local configuration, model files, runtime binaries, virtual environments and logs are not tracked.

## Updating an existing checkout

From the project directory:

```powershell
git pull --ff-only origin main
uv sync --extra gui --extra diagnostics --extra asr
```

If local tracked changes prevent a fast-forward update, review or back them up before resetting. Keep your private configuration, model files and runtime binaries.
