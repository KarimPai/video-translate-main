# Japanese Video Translator

Experimental branch `test-reazon-k2-plamo2b` for local Japanese-to-English subtitles. Uses FireRedVAD (CUDA), **ReazonSpeech k2 v2 Japanese ASR** (CPU via sherpa-onnx Zipformer INT8), and **PLaMo 3 Translate 2B GGUF Q4_K_M** (local llama.cpp). MPV plays the original source; the translator processes audio independently. This branch was derived from `main`; `main`, `test`, and `whisper-v3` are unchanged.

## Windows requirements

- Python 3.12+, `uv`, `yt-dlp`, `ffmpeg`, and `mpv`.
- `sherpa-onnx` (installed by `uv sync`) for the ReazonSpeech Zipformer model. No `speech.cpp` / Haas executable or Vulkan ASR is needed.
- A CUDA-capable GPU for FireRedVAD.
- `llama-server` from a GPU-enabled llama.cpp build on PATH for PLaMo 3 Translate 2B. The translator can start its own server.

Install Python dependencies with `uv sync` (this branch's first sync refreshes the local `uv.lock` for the new `sherpa-onnx` dependency). Optional packages for separation and DeepFilterNet are declared under `[project.optional-dependencies]` in `pyproject.toml`.

## Start a YouTube VOD

From CMD in the project directory:

```cmd
"Start translate.bat"
```

Paste a YouTube URL. The launcher runs a **single translator process** with `--simple-terminal`, displays **JP:** and **EN:** lines directly in CMD, and writes diagnostics to `logs\run.log`. There is no second `uv` process or stdout pipe. On startup failure, the launcher prints the exit code and matching error lines.

Equivalent direct command:

```cmd
uv run --no-sync auto-translate "YOUTUBE_URL" --mode auto --mpv --no-web
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
2. FireRedVAD detects utterance boundaries. By default, the silence interval is 400 ms, maximum utterance length is 5 seconds, and VAD-negative fallback probing is disabled to avoid treating game audio as speech.
3. ReazonSpeech k2 v2 transcribes Japanese; YouTube JP captions are disabled by default for independent ASR comparison (`--youtube-captions` opts in). A conservative filter suppresses obvious non-speech ASR outputs.
4. PLaMo 3 Translate 2B renders English-only subtitles; `glossary.toml` handles proper names and streaming terms.
5. MPV initially waits until translation covers **5 seconds ahead of playback**, then plays. Pressing Play manually bypasses the wait. Playback should not repeatedly pause when translation falls behind.

The MPV demux limits are **1 GiB forward + 1 GiB backward**, with 120 seconds of readahead. These are upper limits, not preallocated memory.

A separate **RAM-only session subtitle cache** stores translated cues and processed regions for the currently loaded VOD. Seeking back to a translated region can immediately restore existing English subtitles and reuse matching ASR/translation results. The subtitle cache is discarded when the application closes or a different video is loaded; it is not an on-disk translation archive.

## Optional processing

Vocal separation and DeepFilterNet are **disabled by default** and require their respective optional dependencies.


## ReazonSpeech k2 v2 setup and testing

- Japanese Zipformer RNN-T, 159.34M parameters; official Apache-2.0 checkpoint:
  https://huggingface.co/reazon-research/reazonspeech-k2-v2
- Uses encoder and joiner **INT8**, decoder **FP32** and tokens from the pinned
  official revision. Around 170 MB total download (rounded).
- Downloads on the first run only into models/reazonspeech-k2-v2/ (gitignored).
  Thereafter recognition works offline.
- Uses CPU by default (4 threads); Reazon needs no Vulkan device, keeping
  RTX 3070 GPU memory available for FireRedVAD and PLaMo.
- PLaMo's existing glossary, MPV seek buffer, player and RAM-only subtitle
  cache are inherited from main.
- No extra ASR->translator model: Reazon produces Japanese transcription;
  the already configured local PLaMo translates it to English.
- VOD YouTube caption assistance is **disabled by default** to measure real
  ASR recognition. Use --youtube-captions to re-enable it. Live captions
  remain unavailable in this mode.
- VAD-negative ASR probing is disabled for this experiment to reduce
  fabricated text in non-speech background audio; change [vad] fallback_s
  only if you understand that risk.

Windows CMD (new clone):

    cd /d C:\Games
    git clone --branch test-reazon-k2-plamo2b --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-reazon
    cd /d C:\Games\video-translate-reazon
    uv sync
    "Start translate.bat"

Subsequent updates:

    cd /d C:\Games\video-translate-reazon
    git pull --ff-only origin test-reazon-k2-plamo2b

Optional config.toml:

    [asr]
    model_dir = "models/reazonspeech-k2-v2"
    num_threads = 4
    provider = "cpu"

To run the offline model adapter tests (no model downloads):

    uv run --no-sync python -m unittest discover -s tests -p "test_reazon*.py" -v

The full runtime must be validated on Windows. Reazon's published Japanese
CER scores should not be interpreted as Hololive GTA accuracy.


## Windows ONNX Runtime native-crash mitigation

On Windows, an older onnxruntime.dll such as 1.17.1 in System32 can
override sherpa-onnx's newer bundled runtime, resulting in
"The requested API version [28] is not available ... ORT 1.17.1".
See https://github.com/k2-fsa/sherpa-onnx/issues/3059.

To mitigate, the ASR now runs in a **separate Python subprocess** and
preloads the matching onnxruntime.dll from the sherpa_onnx/lib wheel
directory before importing sherpa's native extension. This isolates its
DLLs from FireRedVAD and other ONNX users in the parent process.

A worker failure is reported to logs/run.log with the process exit code
and diagnostic instructions. First-run model downloads also log to the
same file. The local PLaMo llama.cpp GGUF/cache is unchanged and will
be reused if already downloaded.

If an older DLL is loaded despite preloading, do not delete or replace
Windows system DLLs; inspect the full startup log instead. Run:

    cd /d C:\Games\video-translate-reazon
    git pull --ff-only origin test-reazon-k2-plamo2b
    uv sync
    "Start translate.bat"


## Japanese Unicode output on Windows

The Reazon ASR subprocess sends line-delimited JSON. Windows can default to
cp1252 for redirected Python stdout even after the launcher calls chcp 65001.
The worker now reconfigures its streams to UTF-8, the parent sets
PYTHONIOENCODING=utf-8, and worker JSON is ASCII-escaped as a further fallback.
This fixes the earlier UnicodeEncodeError for Japanese transcripts (e.g. え).
