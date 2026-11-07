# Japanese Video Translator

Japanese-to-English subtitles for YouTube streams/VODs and local media. The current Windows MPV workflow uses FireRedVAD, Haas Japanese ASR (speech.cpp / Vulkan), and PLaMo 3 Translate 2B running locally through llama.cpp. MPV plays the original source; the translator processes audio independently.

## Windows requirements

- Python 3.12+, `uv`, `yt-dlp`, `ffmpeg`, and `mpv`.
- `speech` from speech.cpp on PATH, with the `haas-ja-v1.0` model and a working Vulkan device.
- A CUDA-capable GPU for FireRedVAD.
- `llama-server` from a GPU-enabled llama.cpp build on PATH for PLaMo 3 Translate 2B. The translator can start its own server.

Install Python dependencies with `uv sync`. Optional packages for separation and DeepFilterNet are declared under `[project.optional-dependencies]` in `pyproject.toml`.

## Start a YouTube VOD

From CMD in the project directory:

```cmd
"Start translate.bat"
```

Paste a YouTube URL. The launcher runs a **single translator process** with `--simple-terminal`, displays **JP:** and **EN:** lines directly in CMD, and writes diagnostics to `logs\run.log`. There is no second `uv` process or stdout pipe. On startup failure, the launcher prints the exit code and matching error lines.

Equivalent direct command:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mode vod --mpv --no-web
```

For extra diagnostic logging, append `-v`. Use `uv run auto-translate --help` for the active CLI switches. PLaMo 3 Translate 2B is the only translation engine; no backend selection is necessary.

## PLaMo 3 Translate 2B

`config.toml` is local and intentionally git-ignored. PLaMo is enabled by default; optionally configure it with:

```toml
[translate]
plamo_base_url = "http://127.0.0.1:8000/v1"
plamo_model = "plamo-3-translate"
plamo_hf_model = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
plamo_gpu_layers = 99
context_lines = 8
memory_lines = 24  # additional recent subtitle pairs for term and scene memory
glossary = "glossary.toml"
```

If no `[translate]` configuration is supplied, PLaMo 3 Translate 2B Q4_K_M is selected automatically. The translator starts a managed `llama-server` when no server responds at the configured URL; its log is `logs\plamo-server.log`. Older local engine selectors and unused legacy translator options are ignored for compatibility. Configure `[asr]`, `[vad]`, and `[mpv]` via `src/auto_translate/config.py`.

PLaMO keeps the last `context_lines` JP/EN pairs as dialogue and can inspect up to `memory_lines` recent pairs for brief terminology and earlier Japanese scene hints. Exact glossary spellings remain authoritative; learned short-term word hints are tentative. A conservative no-model-pass repair can fix one-edit repetitions of short Katakana words (for example `シャッター` / `シャッタ`) from adjacent subtitles, or from two matching words on either side of a misheard word in one subtitle. It cannot resolve unrelated ASR misrecognitions. Memory is limited to the current video segment and is cleared on seeks; set `memory_lines = 0` to disable the extra memory and repetition repair. Larger memory may add prompt tokens and translation time.

Translation failures and missing subtitles should be investigated using `logs\run.log` first; `logs\mpv.log` contains MPV warnings and errors. Use `-v` for additional pipeline diagnostics.

## Detection and subtitles

1. FFmpeg decodes Japanese audio, with accelerated VOD ingestion in MPV mode.
2. FireRedVAD detects utterance boundaries. By default, the silence interval is 400 ms, maximum utterance length is 5 seconds, and fallback probing checks speech that VAD may miss.
3. Haas transcribes utterances, with optional Japanese YouTube caption assistance. A conservative filter suppresses obvious non-speech ASR outputs.
4. PLaMo 3 Translate 2B renders English-only subtitles; `glossary.toml` handles proper names and streaming terms.
5. MPV initially waits until translation covers **5 seconds ahead of playback**, then plays. Pressing Play manually bypasses the wait. Playback should not repeatedly pause when translation falls behind.

The MPV demux limits are **1 GiB forward + 1 GiB backward**, with 120 seconds of readahead. These are upper limits, not preallocated memory.

A separate **RAM-only session subtitle cache** stores translated cues and processed regions for the currently loaded VOD. Seeking back to a translated region can immediately restore existing English subtitles and reuse matching ASR/translation results. The subtitle cache is discarded when the application closes or a different video is loaded; it is not an on-disk translation archive.

## Optional processing

Vocal separation and DeepFilterNet are **disabled by default** and require their respective optional dependencies.
