# Japanese Video Translator

Japanese-to-English subtitles for YouTube streams/VODs and local media. The current Windows MPV workflow uses FireRedVAD, Haas Japanese ASR (speech.cpp / Vulkan), and a local translation backend. MPV plays the original source; the translator processes audio independently.

## Windows requirements

- Python 3.12+, `uv`, `yt-dlp`, `ffmpeg`, and `mpv`.
- `speech` from speech.cpp on PATH, with the `haas-ja-v1.0` model and a working Vulkan device.
- A CUDA-capable GPU for FireRedVAD.
- **PLaMo option:** `llama-server` from a GPU-enabled llama.cpp build on PATH. The translator can launch its own local server.
- **HY-MT2 option:** an already running, compatible local OpenAI-style translation endpoint. HY-MT2 remains an available backend in this branch.

Install Python dependencies with `uv sync`. Optional packages for separation, DeepFilterNet, and speaker diarization are declared under `[project.optional-dependencies]` in `pyproject.toml`.

## Start a YouTube VOD

From CMD in the project directory:

```cmd
"Start translate.bat"
```

Paste a YouTube URL. The launcher runs MPV and displays **JP:** and **EN:** lines in CMD. Diagnostic messages go to `logs\run.log`, rather than flooding the terminal.

Equivalent direct command:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mode vod --mpv --no-web
```

For extra diagnostic logging, append `-v`. Use `uv run auto-translate --help` for the active CLI switches. There is no `--engine` switch: change the translation backend in `config.toml`.

## Translation backend

`config.toml` is local and intentionally git-ignored. To use PLaMo 3 Translate:

```toml
[translate]
backend = "plamo"
plamo_base_url = "http://127.0.0.1:8000/v1"
plamo_model = "plamo-3-translate"
plamo_hf_model = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
plamo_gpu_layers = 99
context_lines = 8
glossary = "glossary.toml"
```

If no `[translate]` configuration is supplied, the code's current default is `hymt2`, which requires its local service. The PLaMo backend uses a managed `llama-server` when no server is already running at the configured URL; its log is `logs\plamo-server.log`. Configure `[asr]`, `[vad]`, and `[mpv]` via `src/auto_translate/config.py`.

Translation failures and missing subtitles should be investigated using `logs\run.log` first; `logs\mpv.log` contains MPV warnings and errors. Use `-v` for detailed pipeline, subtitle timing, and audio-clock diagnostics.

## Detection and subtitles

1. FFmpeg decodes Japanese audio, with accelerated VOD ingestion in MPV mode.
2. FireRedVAD detects utterance boundaries. By default, the silence interval is 400 ms, maximum utterance length is 5 seconds, and fallback probing checks speech that VAD may miss.
3. Haas transcribes utterances, with optional Japanese YouTube caption assistance. A conservative filter suppresses obvious non-speech ASR outputs.
4. The configured translator renders English-only subtitles; `glossary.toml` handles proper names and streaming terms.
5. MPV initially waits for up to **5 seconds of translated audio lead**, then plays. Pressing Play manually bypasses the wait. Playback should not repeatedly pause when translation falls behind.

The MPV demux limits are **1 GiB forward + 1 GiB backward**, with 120 seconds of readahead. These are upper limits, not preallocated memory.

A separate **RAM-only session subtitle cache** stores translated cues and processed regions for the currently loaded VOD. Seeking back to a translated region can immediately restore existing English subtitles and reuse matching ASR/translation results. The subtitle cache is discarded when the application closes or a different video is loaded; it is not an on-disk translation archive.

## Optional processing

Vocal separation, DeepFilterNet, and pyannote speaker diarization are **disabled by default** and require their respective optional dependencies. These components remain supported; they are not orphaned modules.

## Tests

Run the retained regression tests from the project folder:

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

The `tests/test_*.py` files are development tests; they do not run during normal VOD playback. Temporary manual MPV probe scripts have been removed.

## Troubleshooting

- **No English subtitle:** Check `logs\run.log` for ASR, translation, and seek errors; captions and ASR may not cover every utterance.
- **A name is misrecognized:** Update `glossary.toml`; name corrections require supporting Japanese/context, not unconditional word replacement.
- **Translation lags:** Review the pipeline queue/backpressure and ASR/MT timings in `logs\run.log`. GPU contention and long translation prompts can reduce throughput.
- **Seeking to uncached content:** MPV may need to fetch video data and the pipeline may have to transcribe/translate new audio.
- **PLaMo startup errors:** Check whether `llama-server` is on PATH and read `logs\plamo-server.log`. Reusing a previously running server is supported.

## Branches

- `main`: saved working baseline.
- `test`: development and cleanup branch. Validate changes here before promoting them to `main`.
