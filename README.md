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

Translation failures and missing subtitles should be investigated using `logs\run.log` first; `logs\mpv.log` contains MPV warnings and errors. Use `-v` for detailed pipeline, subtitle timing, and audio-clock diagnostics.

### Selective PLaMo quality recheck

The translator uses its existing video-title context, recent dialogue and Hololive
glossary on every request. A short instruction asks PLaMo not to romanize
meaningless vocalizations; obvious repeated Japanese syllables are suppressed
before inference. If a translation contains suspicious phonetic repetition,
untranslated Japanese, a possible missing explicit negation, or an unexpected
`[SKIP]` token, it can receive **one** additional request through the same
PLaMo model. Ordinary subtitles use a single request; no separate reasoning model
or ASR pass is used. Rechecks are throttled and time-limited to avoid persistent
live-stream backlogs. Proper names and meaningful speech are preserved by
conservative validation. These checks cannot prove semantic correctness or
recover words that Haas did not recognize.

Optional local settings:

```toml
[translate]
quality_recheck_enabled = true
quality_recheck_timeout_s = 3.0
quality_recheck_min_interval_s = 10.0
```

Use `quality_recheck_enabled = false` to turn off only the second-pass requests.
`config.toml` is git-ignored and existing local values take precedence.

### FireRedVAD defaults

The recommended default values for both live streams and VODs are:

```toml
[vad]
threshold = 0.35
neg_threshold = 0.25
speech_pad_ms = 200
min_silence_ms = 400
min_speech_ms = 100
max_utterance_s = 6.0
fallback_s = 5.0
```

`max_utterance_s` must remain finite; infinity is unsupported by the current frame conversion. Because `config.toml` is git-ignored, any existing local `[vad]` values still override these defaults. Remove or update old overrides to use the recommendations.

## Detection and subtitles

1. FFmpeg decodes Japanese audio, with accelerated VOD ingestion in MPV mode.
2. FireRedVAD detects utterance boundaries. The defaults use 200 ms speech padding, 400 ms silence, 100 ms minimum speech and a finite 6-second maximum utterance; fallback probing checks speech that VAD may miss.
3. Haas transcribes utterances, with optional Japanese YouTube caption assistance. A conservative filter suppresses obvious non-speech ASR outputs.
4. PLaMo 3 Translate 2B renders English-only subtitles; `glossary.toml` handles proper names and streaming terms.
5. MPV initially waits until translation covers **5 seconds ahead of playback**, then plays. Pressing Play manually bypasses the wait. Playback should not repeatedly pause when translation falls behind.

The MPV demux limits are **1 GiB forward + 1 GiB backward**, with 120 seconds of readahead. These are upper limits, not preallocated memory.

A separate **RAM-only session subtitle cache** stores translated cues and processed regions for the currently loaded VOD. Seeking back to a translated region can immediately restore existing English subtitles and reuse matching ASR/translation results. The subtitle cache is discarded when the application closes or a different video is loaded; it is not an on-disk translation archive.

## Live streams

The existing launcher automatically detects ongoing YouTube live streams. MPV
plays about **8 seconds behind the incoming audio** by default. Set
`[mpv] delay_s = 5.0` through `10.0` in local `config.toml`, or pass
`--delay 8` on the CLI. If the first audio is still processing at 8 seconds,
startup may wait up to 10 seconds but never waits indefinitely for translation.

Live HLS audio and MPV both target the third-to-last available segment. Live
MPV timestamps are mapped relative to the audio-start anchor, unlike the
absolute VOD timeline. The live audio reader retries up to three consecutive
failures and retains a continuous subtitle timeline on reconnect. Live mode
does not use the VOD translation cache or VOD seek buffering.

## Optional processing

Vocal separation and DeepFilterNet are **disabled by default** and require their respective optional dependencies.

## Optional: compare Japanese ASR models

In the `test` branch, double-click `Compare ASR.bat` from the project
folder and press Enter to benchmark the supplied short YouTube sample
(`https://www.youtube.com/watch?v=LE8iGHrakAc`). You can also paste a
different YouTube URL. The launcher downloads the source audio once, converts
its first 120 seconds to 16 kHz mono, then uses speech.cpp Silero VAD to
produce the **same short speech regions** for:

1. Haas-ja-v1.0 Q8 with the app's GPU-decoding setting;
2. Haas-ja-v1.0 Q8 with this clip's bread-name vocabulary hints;
3. Qwen3-ASR 1.7B with Japanese-language selection and **no prompt**;
4. `neosophie/Qwen3-ASR-1.7B-JA` fine-tune, using
   `kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF` Q8_0 and **no prompt**.

The same audio and VAD segments are reused for all four models. The new report is
`logs/asr_benchmark/VIDEO_ID/compare_finetune.txt`. Earlier `compare.txt`
and `compare_clean.txt` results are preserved. The first run may download the
Japanese fine-tune GGUF (~2.18 GB) through the existing `huggingface-hub`
dependency; Hugging Face caches it for subsequent runs. The base Qwen
model and Silero VAD use speech.cpp's normal cache. Model downloads happen
*before* recognition timing; timings still include model startup. The
original prompted Qwen run hallucinated its prompt on six segments, so both
Qwen variants intentionally use no prompt. The third-party speech.cpp
conversion is separate from the original BF16 safetensors and has not been
validated against this specific stream yet. **This does not produce an accuracy score without a manually verified
Japanese reference transcript.** Compare the transcriptions by listening to
each WAV segment in `segments`. The benchmark uses speech.cpp's Silero
segmentation, whereas the normal MPV translator uses FireRedVAD, so inference
times and text may differ from live playback. `Start translate.bat` and the
default Haas backend are unchanged.

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
