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

## Optional lower-pitch VAD + ASR experiment

The input to FireRedVAD and Haas can be pitched down in the existing
FFmpeg audio-only ingest path. This is **disabled by default**; there is no
published evidence that lowering pitch improves FireRedVAD detection on
high-pitched Japanese voices. In your git-ignored `config.toml`, opt in with:

```toml
[ingest]
pitch_shift_semitones = -2.0  # experiment: compare against 0.0 (default)
```

Allowed values are -4.0 through 0.0 semitones. Start at -2.0 and compare
speech detection, Japanese recognition, subtitle latency and false positives
against 0.0 using the **same VOD**. The FFmpeg chain uses `asetrate`,
`aresample` and `atempo` to approximately maintain audio duration, but may
introduce artifacts, filtering delay or small alignment differences.
Both FireRedVAD and Haas receive the modified audio; Qwen and the MPV video
source do not directly receive the altered sound. This does **not** separate
speakers. The previous `[ingest] pitch_semitones` field is ignored on
purpose: old settings will not unexpectedly enable pitch shifting.

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
