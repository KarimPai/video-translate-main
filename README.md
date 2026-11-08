# Japanese Video Translator — Qwen3.5-4B experiment

Isolated `test-qwen35-4b` branch:

- **ASR:** Haas-ja-v1.0 Q8_0 in speech.cpp on Vulkan, CPU decoder.
- **Translation:** Qwen3.5-4B Q4_K_M via the managed CUDA llama-server.

FireRedVAD, Sherpa CPU speaker segmentation, YouTube caption correction,
Hololive glossary and English-only MPV subtitles remain unchanged.
There is no secondary ASR fallback; Haas is the only recognizer.

## Requirements

- Windows, Python 3.12+, `uv`, `ffmpeg`, `yt-dlp`, and MPV.
- `speech` / speech.cpp 0.10+ with Haas and Vulkan support; CUDA for FireRedVAD.
- **Recent CUDA-enabled llama.cpp `llama-server`** with Qwen3.5 GGUF model
  support, `--jinja` and `--reasoning off`. Older binaries may exit during
  startup; check `logs\qwen-server.log` and upgrade them if necessary.
- Model: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  using the published `Q4_K_M` quantization (~2.74 GB of weights). The
  server downloads the model on first use and caches it through llama.cpp.
  No vision projector is required for text translation.
- Japanese ASR model: [sakasegawa/Haas-ja-v1.0-GGUF](https://huggingface.co/sakasegawa/Haas-ja-v1.0-GGUF).
  Selected as `haas-ja-v1.0:q8_0` (656 MB). speech.cpp fetches and checks
  it automatically; run `speech info haas-ja-v1.0:q8_0` to prefetch.
  Haas is not a llama.cpp GGUF. The Qwen3-ASR model and Python loader
  are not used.

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
haas_model = "haas-ja-v1.0:q8_0"
haas_device = "Vulkan0"
haas_decode_on_cpu = true  # conserve GPU memory for Qwen3.5
response_timeout_s = 30.0

[translate]
qwen_base_url = "http://127.0.0.1:8001/v1"
qwen_model = "qwen3.5-4b"
qwen_hf_model = "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
qwen_context = 4096
qwen_gpu_layers = 99
subtitle_style = "faithful"  # accuracy-first; "gist" is optional
# Experimental sampling, now explicitly sent to llama.cpp per line:
temperature = 0.1
top_p = 0.8
top_k = 20
min_p = 0.0
repeat_penalty = 1.0
presence_penalty = 0.0
# Optional: sampling_seed = 42  # for repeatable experiments
context_lines = 5
memory_lines = 12
max_tokens = 128
glossary = "glossary.toml"
quality_recheck_enabled = true
# Optional: quality_recheck_temperature = 0.5 (only on flagged lines)
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

The experimental branch now defaults to `subtitle_style = "faithful"`:
keep more spoken details rather than compressing every line to a gist.
`subtitle_style = "gist"` remains available in local `config.toml`.
Neither option changes the speaker-turn logic or the Japanese ASR text.

The default `temperature = 0.1` was selected after reviewing the
50-case Japanese transcript experiment (single seed, no verified source
audio). It favors faithful wording but is not a universal optimum.
An existing, git-ignored `config.toml` with `temperature = 0.35`
continues to override the default until edited.

Sampling controls (`temperature`, `top_p`, `top_k`, `min_p`,
`repeat_penalty`, `presence_penalty`) are explicit per-request values
sent to the managed Qwen3.5 llama.cpp server. The starting values are
**experimental**, not an established optimum. The selective quality
recheck uses the same values unless `quality_recheck_temperature` is
configured; this affects only a flagged second request, never every line.
Thinking remains disabled and the context window remains 4096 tokens.

### Benchmark-driven terminology and duration safeguards

For ARK, the Japanese item-quality rank `至高` is officially `Ascendant`,
so `至高のサドル` means `Ascendant saddle`, not `ultimate saddle`.
The observed ASR homophone `思考のサドル` is repaired only in confirmed
ARK contexts. Other `思考` and `至高` usages are unaffected.

VALORANT aliases such as `バローランド` require independently supported
gaming context. The English spelling `Baro-land` is also corrected in that
context. When the Japanese explicitly says `バロトラウマ`, the separate
game `Barotrauma` is preserved and specific observed garbled names are
repaired, without renaming other games.

For the attested GTA exchange `1分に延ばせ`, the duration is extended
**to one minute**, not **by another minute**. A narrow quality review and
fallback fix protect this difference without rewriting unrelated timings.
The 50-case benchmark now checks these exact names and duration words.

### Compare Qwen sampling temperatures on your PC

Close `Start translate.bat` and any other process using local Qwen port
8001, then double-click `Benchmark sampling.bat`. This launches **only
Qwen3.5-4B translation**, not Haas ASR, FireRed, Sherpa or MPV. It loads
the translator **once**, then compares the same 50 Japanese/reference pairs
using temperature `0.1`, `0.35`, `0.5` and `0.7`. All other sampling
values, the faithful prompt, the glossary, source-only context and the
optional fixed seed are held constant. Quality rechecks and previous English
memory are disabled to isolate the sampling setting. No `config.toml`
changes are made; any pre-existing local settings are respected except
benchmark-only temperature, style, seed and recheck overrides.

The benchmark writes two files under `logs`: detailed English outputs
and per-temperature summary CSV. The **reference character similarity**
is a text-string comparison, **not** semantic translation accuracy.
Terms/number checks only validate explicitly listed strings. The 50
examples are curated from previous Japanese transcript snippets and
edited examples: **not** verified against the original audio, so they
cannot measure ASR accuracy. Review the actual English outputs manually,
especially names, context, negatives and unsupported guesses.

You can supply your own Japanese/reference pairs (one JSON object per line):

```json
{"ja":"30万だろ!?","reference_en":"300,000, right?!","title":"Hololive GTA","must_include":["300,000"],"must_exclude":["30,000"]}
```

Run an additional repeatability comparison with:

```cmd
uv run --no-sync python tools\benchmark_qwen_sampling.py --rounds 2 --seed 42
```

`--temperatures`, `--samples` and `--output` are also supported.
**Do not choose the winner using reference similarity alone.** If a local
`config.toml` has `subtitle_style = "gist"`, change it to `"faithful"`
to adopt the new runtime default.

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
- A short evidence-backed VALORANT/Apex game context now helps with streams
  discussing multiple games. When the current Japanese clearly says
  `バローラント` or `バロラント`, Qwen is guided to VALORANT, not
  Barotrauma; less certain aliases (`バローランド`, `バロランと`) require
  recent strong recognition or the established scene. The final subtitle
  corrects only the *English game name*, leaving the raw Japanese ASR intact.
  No additional ASR or translator inference is performed. Other games and
  unverified names are not automatically changed.
- Explicit Choco-san honorific and conversational idioms such as
  `空気を探り合ってる` are kept natural without inferring missing speech.

If the stream has no Japanese YouTube captions, recognition uses Haas-ja-v1.0 Q8_0.
A translation model cannot recover what ASR never heard correctly. A nickname
resolver must not automatically replace genuine `猫ちゃん` with `ぺこちゃん`.

Haas Q8_0 is a 656 MB file, while Qwen3.5-4B Q4_K_M has about 2.74 GB
of weights; total runtime allocations are higher. Haas uses Vulkan for
recognition with CPU decoding enabled to preserve GPU memory on RTX 3070 8 GB.
Monitor `nvidia-smi` during playback if VRAM remains constrained.
Retired `[asr]` keys such as `model_path`, `device` and `prompt` are ignored
when an older git-ignored `config.toml` exists.

## Original-pitch audio

The old experimental FFmpeg pitch shift has been removed. FireRedVAD,
Sherpa speaker diarization, and Haas ASR all receive original-pitch PCM audio
(resampled only as required). MPV playback remains unmodified.
Both legacy `pitch_semitones` and `pitch_shift_semitones` config entries
are ignored for compatibility with existing local `config.toml` files.

## Sherpa-ONNX voice-turn-separated subtitles

The optional Sherpa CPU stage runs after FireRedVAD and before Haas ASR. It
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
Speech between 1.65 and 3 seconds is symmetrically padded with silence for
segmentation without modifying the actual audio or timeline. Utterances
shorter than 1.65s remain whole to avoid speculative cuts. Clear speaker
boundaries before/after an overlapping pair can still be separated, while
the overlapping part remains one mixed cue (not falsely attributed to one
voice). When a clean
non-overlapping speaker change is detected, the system picks a nearby
low-energy boundary, ensures all PCM is preserved exactly once, and sends
each slice independently through Haas ASR and Qwen3.5 translation. MPV retains up to three
English-only rows, newest above, with no speaker letters in MPV or CMD.

**Limitations:** Diarization is not source separation. Simultaneous
overlapping speakers or uncertain boundaries remain unsplit, because
slicing overlapping voices can mix dialogue or clip spoken words. The
2-3s path can increase CPU usage; check live subtitle latency.

Set `enabled = false` to bypass the stage. Legacy
`similarity_threshold` and `max_speakers` entries in local
`config.toml` are ignored for compatibility.

## Silent-subtitle stall handling

If FFmpeg ends a VOD's audio significantly before its known duration, the
ingest stage re-resolves the signed YouTube audio URL and resumes from the
processed position (up to three attempts). Repeated early EOF is an explicit
error rather than a silent stop while MPV continues to play.

The Haas ASR speech.cpp worker has a 30-second response deadline for each
utterance (`[asr] response_timeout_s = 30.0`). A crashed ingestion, VAD,
Sherpa, ASR or translation thread now reports a fatal stage error rather than
leaving MPV playing without subtitles. If queued audio stops advancing,
diagnostics log stage health after 20 seconds and fail explicitly after 60
seconds of stalled queued work. A legitimately quiet stream with no queued
audio is not treated as a stall.

Inspect `logs\\run.log` for `pipeline stage`, `response timeout`,
`no pipeline output`, `pipeline stalled`, `ASR returned no Japanese`
and `Sherpa unsplit` messages. Excessive live ingest reconnect failures
also produce a fatal error rather than silently stopping subtitles.
This does not separate simultaneous voices; clean changes on either side of
overlap are now preserved when possible.

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
- `test-qwen35-4b`: isolated Haas Q8_0 ASR + Qwen3.5 translator.
