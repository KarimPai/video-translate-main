# Japanese Video Translator — Qwen3.5-4B experiment

Isolated `test-qwen35-4b` branch. This experiment replaces the translator
with **Qwen3.5-4B Q4_K_M** through local `llama-server`. The established
FireRedVAD + Haas-ja-v1.0 Vulkan recognition, YouTube caption reconciliation,
Hololive nickname fixes, ARK contextual corrections, glossary and MPV
subtitles are unchanged.

## Requirements

- Windows, Python 3.12+, `uv`, `ffmpeg`, `yt-dlp`, and MPV.
- `speech` / speech.cpp with Haas-ja-v1.0 on Vulkan; CUDA for FireRedVAD.
- **Recent CUDA-enabled llama.cpp `llama-server`** with Qwen3.5 GGUF model
  support, `--jinja` and `--reasoning off`. Older binaries may exit during
  startup; check `logs\qwen-server.log` and upgrade them if necessary.
- Model: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  using the published `Q4_K_M` quantization (~2.74 GB of weights). The
  server downloads the model on first use and caches it through llama.cpp.
  No vision projector is required for text translation.

## Starting the independent branch

Clone this branch to a **different Windows folder** so your existing checkout
of `test` remains untouched:

```cmd
cd /d C:\Games
git clone --branch test-qwen35-4b --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-qwen
cd /d C:\Games\video-translate-qwen
uv sync
"Start translate.bat"
```

The batch file automatically detects live/VOD, runs MPV and displays `JP:`
and `EN:` terminal output. Logs are in `logs\run.log` and
`logs\qwen-server.log`. If MPV does not appear or the model fails to load,
inspect these logs. The original `test` branch is not changed.

## Local settings

`config.toml` is git-ignored. If it does not exist, these defaults apply:

```toml
[translate]
qwen_base_url = "http://127.0.0.1:8001/v1"
qwen_model = "qwen3.5-4b"
qwen_hf_model = "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
qwen_context = 4096
qwen_gpu_layers = 99
subtitle_style = "gist"  # "gist" (short meaning-focused) or "faithful"
context_lines = 5
memory_lines = 12
max_tokens = 128
glossary = "glossary.toml"
quality_recheck_enabled = true
quality_recheck_timeout_s = 3.0
quality_recheck_min_interval_s = 10.0
```

**The dedicated 8001 port prevents accidental reuse of an unrelated local
translation server.** The translator **refuses to reuse an existing server**,
even one with the correct model ID: it cannot guarantee cleanup of a process
it did not start. Close any previous llama-server on 8001 before launching or
choose another free port in local config.toml.

### Automatic model-server shutdown

The translator's managed llama-server **closes automatically** when the MPV
window is closed, the translator exits after a VOD, or Ctrl+C is pressed.
The same Windows Job Object kill-on-close mechanism as MPV also stops its
owned server if the CMD window or Python process closes unexpectedly and
Python's normal cleanup handlers cannot run. The Qwen model is unloaded and
VRAM is freed when the server process terminates.

This does not terminate unrelated llama.cpp processes or a server started
outside this translator. Logs: `logs\\qwen-server.log`.

### Subtitle style

This experimental branch defaults to `subtitle_style = "gist"`: short,
natural, meaning-first English rather than literal transcription. It trims
fillers and repetition without dropping names, numbers, essential facts,
negation, questions, or important gaming terminology. It does not summarize
across separate utterances or change the number of speakers. For the
previous, more faithful style, use `subtitle_style = "faithful"` under
`[translate]` in your local `config.toml`.

The style prompt is appended to the existing Qwen system message, so glossary
and accuracy safeguards still apply. Sampling parameters and output token
limits are unchanged, allowing a cleaner comparison against the previous run.

### Prompts and terminology

- Uses a **Qwen chat system instruction** and a short per-cue user request;
  it does not send model-specific translator metadata.
- Instructs Qwen to output English only, **disable thinking**, preserve names
  and honorifics, and not invent speakers or missing words.
- Sends up to `context_lines` **recent Japanese lines**, never the prior
  generated English translations, to avoid repeating earlier model mistakes.
- Only includes matching **current-cue** entries from the unchanged
  `glossary.toml`, up to ten terms. ARK terms apply only when game context
  is supported by the title or recent Japanese speech.
- Preserves the existing caption reconciliation, Peko-chan name support,
  speech filtering, polarity review and selective one-time recheck.
  Known high-confidence translation inversions are repaired without a model call.

If the stream has no Japanese YouTube captions, recognition still uses Haas.
A translation model cannot recover what ASR never heard correctly. A nickname
resolver must not automatically replace genuine `猫ちゃん` with `ぺこちゃん`.

The model file plus GPU/Vulkan recognition, VAD and MPV compete for 8 GB
VRAM. Start with Q4_K_M, a 4096-token context, and a single server slot.
A larger context may increase memory pressure or latency.

## Original-pitch audio

The old experimental FFmpeg pitch shift has been removed. FireRedVAD,
Sherpa speaker diarization, and Haas all receive original-pitch PCM audio
(resampled only as required). MPV playback remains unmodified.
Both legacy `pitch_semitones` and `pitch_shift_semitones` config entries
are ignored for compatibility with existing local `config.toml` files.

## Sherpa-ONNX speaker-aware subtitles

The isolated Qwen experiment now includes a CPU-only Sherpa-ONNX speaker
segmentation stage **before Haas**. It is enabled in configuration by default,
but Sherpa is an optional Python package. Install it once in this checkout:

```cmd
uv sync --extra diarization
```

On the first run the launcher prints an explicit download message for
pyannote segmentation 3.0 INT8 (1.54 MB) and NeMo TitaNet Small (40.3 MB).
The downloader first uses official maintainer-hosted Hugging Face standalone
ONNX files, retries temporary errors, then falls back to Sherpa's GitHub
release assets. It checks SHA256 before accepting either model, rejects
partial caches, and preserves the original non-diarized audio path if both
sources fail. The files are cached under `%LOCALAPPDATA%\auto-translate\speaker-models`.
Model setup failure falls back to the unchanged unsplit FireRedVAD path.
No separate model runs on the GPU. The CPU cost and extra subtitle delay
must be measured on the target i7-9700K.

```toml
[speaker]
enabled = true
num_threads = 2
min_turn_s = 0.55
similarity_threshold = 0.68
max_speakers = 4
max_turns = 5
```

Set `enabled = false` to restore the original pipeline immediately. The
optional `model_dir = ""` uses the default model cache directory.

The implementation processes completed FireRedVAD utterances through
Sherpa's **offline** diarization API. Clear speaker changes produce separate
audio slices, Haas transcriptions, Qwen translations and MPV subtitle lines.
MPV prioritizes distinct speakers (A:, B:, etc.) among its three lines,
with the newest lines above. A single speaker can still have multiple
stacked lines if there are no other speakers to prioritize.

Speaker IDs now use session-long voice memory, retaining multiple
reference embeddings per speaker. The first voice sample is provisional; a
second independent, matching utterance is needed before a new letter is
allocated. Identity memory persists through VOD seeks and resets on exit.
The registry is strictly capped at max_speakers (default four labels A-D);
unfamiliar or ambiguous voices are left unlabeled, not displayed as E/F or
assigned to the closest voice without adequate evidence. Very short samples
also cannot create new identities. This reduces label fragmentation but
cannot guarantee perfect identity matching or unmix overlapping speech.

## Tests

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

All tests are offline and need no live model unless explicitly specified by
the test. They do not prove Japanese-to-English quality; compare the same
recorded Japanese clip against the `test` branch for accuracy and speed.

## Branch separation

- `main`: original stable baseline.
- `test`: existing Haas + original translator pipeline.
- `test-qwen35-4b`: isolated Qwen-only translator experiment.
