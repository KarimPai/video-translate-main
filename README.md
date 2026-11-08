# Japanese Video Translator — Qwen3.5-4B experiment

Isolated `test-qwen35-4b` branch. Both models are Qwen-only:

- **ASR:** Japanese-fine-tuned Qwen3-ASR-1.7B-JA Q8_0 in speech.cpp on Vulkan.
- **Translation:** Qwen3.5-4B Q4_K_M in the managed llama-server on CUDA.

FireRedVAD, Sherpa voice-turn segmentation, optional YouTube captions, the
Hololive glossary and MPV's English subtitle overlay are retained. There is
no Haas ASR fallback.

## Requirements

- Windows, Python 3.12+, `uv`, `ffmpeg`, `yt-dlp`, and MPV.
- `speech` / speech.cpp 0.8.10+ with Qwen3-ASR and Vulkan support; CUDA for FireRedVAD.
- **Recent CUDA-enabled llama.cpp `llama-server`** with Qwen3.5 GGUF model
  support, `--jinja` and `--reasoning off`. Older binaries may exit during
  startup; check `logs\qwen-server.log` and upgrade them if necessary.
- Model: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  using the published `Q4_K_M` quantization (~2.74 GB of weights). The
  server downloads the model on first use and caches it through llama.cpp.
  No vision projector is required for text translation.
- Japanese ASR model: [kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF](https://huggingface.co/kznkzn/Qwen3-ASR-1.7B-JA-speechcpp-GGUF),
  Q8_0, 2,176,109,280 bytes (2.03 GiB). It is the speech.cpp-specific
  conversion of Neosophie's proper-noun Japanese fine-tune, not a llama.cpp
  GGUF. The application downloads it via the existing huggingface-hub
  dependency on first use, verifies its SHA-256, and caches it.
  First download requires roughly 2.18 GB. To pre-download:
  `uv run --no-sync python -c "from auto_translate.asr.qwen_asr import resolve_qwen_asr_model; from auto_translate.config import AsrConfig; print(resolve_qwen_asr_model(AsrConfig()))"`.
  The full Q8_0 fine-tune was conversion-tested but has no published
  Hololive/stream accuracy benchmark. Expect higher VRAM and potentially
  slower ASR than the former small recognizer.

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
[asr]
device = "Vulkan0"
model_path = ""         # optional verified local .gguf path for offline setup
prompt = ""             # optional SHORT names/terms hint; off to avoid hallucinations

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
- Maintains a **small scene context** from current Japanese ASR/captions.
  Known topics (horse racing, ARK, GTA) and up to three evidenced terms can
  persist beyond the five-line prompt window. Scene memory expires after
  32 utterances without support, updates when a different topic becomes
  established, and resets on VOD seeks. It never invents names, outcomes or
  speech, never uses prior English guesses, and makes no extra model calls.
  Local `[translate] scene_context_enabled = false` disables this feature
  for A/B comparisons; `scene_context_max_idle_lines = 32` adjusts expiry.
- Only includes matching **current-cue** entries from the unchanged
  `glossary.toml`, up to ten terms. ARK terms apply only when game context
  is supported by the title or recent Japanese speech.
- Preserves the existing caption reconciliation, Peko-chan name support,
  speech filtering, polarity review and selective one-time recheck.
  Known high-confidence translation inversions are repaired without a model call.

If the stream has no Japanese YouTube captions, recognition uses Qwen3-ASR-1.7B-JA.
A translation model cannot recover what ASR never heard correctly. A nickname
resolver must not automatically replace genuine `猫ちゃん` with `ぺこちゃん`.

The two Qwen models compete for 8 GB VRAM with FireRedVAD, MPV and the
Windows desktop. Qwen3-ASR-1.7B Q8_0 is a 2.18 GB file, while Qwen3.5-4B
Q4_K_M has about 2.74 GB of weights; actual allocations are larger and
hardware-dependent. On an RTX 3070 8 GB, test concurrent loading and latency
before assuming both models can stay fully GPU-resident. If the ASR worker
cannot load due to VRAM pressure, try `[asr] device = "cpu"` as a diagnostic
rather than silently substituting another ASR model. That will be slower.
The `[asr] prompt` can provide a few confirmed Japanese terms, but is empty
by default to avoid falsely transcribing unspoken prompt names. A larger
translation context may increase memory pressure or latency.

## Original-pitch audio

The old experimental FFmpeg pitch shift has been removed. FireRedVAD,
Sherpa speaker diarization, and Qwen ASR all receive original-pitch PCM audio
(resampled only as required). MPV playback remains unmodified.
Both legacy `pitch_semitones` and `pitch_shift_semitones` config entries
are ignored for compatibility with existing local `config.toml` files.

## Sherpa-ONNX voice-turn-separated subtitles

The optional Sherpa CPU stage runs after FireRedVAD and before Qwen ASR. It
uses pyannote segmentation 3.0 INT8 and NeMo TitaNet Small *internally*
to detect voice changes **within each VAD utterance**. No session-long
speaker memory or A/B/C identities are generated.

Install the additional dependency once:

```cmd
uv sync --extra diarization
```

The first run downloads the two verified ONNX models to
`%LOCALAPPDATA%\\auto-translate\\speaker-models`. Sherpa's embedding model
remains necessary for clustering voices *within a window*, even though
persistent identity matching has been removed. No extra GPU VRAM is used.

```toml
[speaker]
enabled = true
num_threads = 2
min_turn_s = 0.55
max_turns = 5
```

For 3s+ VAD utterances, Sherpa estimates speaker-turn boundaries directly.
Speech between 2 and 3 seconds is symmetrically padded with silence for
segmentation without modifying the actual audio or timeline. Utterances
shorter than 2s remain whole to avoid speculative cuts. When a clean
non-overlapping speaker change is detected, the system picks a nearby
low-energy boundary, ensures all PCM is preserved exactly once, and sends
each slice independently through Qwen ASR and Qwen3.5 translation. MPV retains up to three
English-only rows, newest above, with no speaker letters in MPV or CMD.

**Limitations:** Diarization is not source separation. Simultaneous
overlapping speakers or uncertain boundaries remain unsplit, because
slicing overlapping voices can mix dialogue or clip spoken words. The
2-3s path can increase CPU usage; check live subtitle latency.

Set `enabled = false` to bypass the stage. Legacy
`similarity_threshold` and `max_speakers` entries in local
`config.toml` are ignored for compatibility.

## YouTube metadata errors

The metadata probe no longer passes the deprecated yt-dlp
`--no-call-home` option. If YouTube reports "We're experiencing technical
difficulties" for a live broadcast, the probe retries twice (brief backoff)
and then surfaces YouTube's original error if the video is still inaccessible.
The retry cannot bypass unavailable streams, authentication restrictions,
or a YouTube outage. Verify the same URL in your browser and try another
working VOD before changing ASR or translation settings.

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
- `test-qwen35-4b`: isolated Qwen Japanese ASR + Qwen3.5 translator.
