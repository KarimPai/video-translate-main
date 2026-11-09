# PLaMo 3 Translate 8B — YouTube captions-only experiment

> **Experimental branch:** `feature/plamo8b-youtube-captions-only`, forked from the
> November 9 `test-plamo8b-q3` commit. This branch does not use audio
> recognition. The original pipeline remains unchanged on its other branches.

## What is enabled

- Archived YouTube **VOD** with Japanese manual subtitles or auto-captions.
  **The complete Japanese subtitle file is downloaded before MPV starts.**
  The downloader now prefers structured **JSON3**, caching the original
  event timestamps, with VTT fallback when JSON3 is not provided.
- Full-track Japanese caption reconstruction: collapse verified rolling
  duplicates, retain real repeated speech, remove sound-effect labels,
  join strongly evidenced split endings (e.g. 美味しかっ + た and わかん + ない),
  and preserve the earliest reliable cue onset while merging.
- For three-part continuations, joining decisions use the **last original
  fragment's** timestamp rather than the first timestamp of the entire
  merged phrase. Maximum combined duration still applies. Coverage includes
  `貸し / て…やめ / なさい` from GTA streams. The local glossary also
  recognizes the caption variant `ペコーラ` as **Pekora**.
- Local PLaMo 3 Translate **8B Q3_K_M** with the model's official PFML3
  `context`, previous Japanese/English `input`/`output` pairs, and
  matching Hololive glossary terms. The current caption is always last.
- Default model slot **2048 tokens**, generated output **192 tokens**,
  temperature **0**, **four recent translated pairs** and a sanitized
  video title/channel context. A conservative per-prompt budget drops
  oldest history before source text and reserves room for generation.
- Selective terminology/meaning repairs, scene-local history across seeks,
  and a bounded exact completion cache keyed by caption **and** context.
- MPV original-source playback (no FFmpeg ingest) with native full seekbar,
  three most recent English lines, timestamp-based rendering, 12-second
  initial caption prebuffer and five-second rebuffer after seeking. A single worker prioritizes a
  35-second window near the current playback position after seeking. The
  first 25 seconds after starting or seeking also have an early-subtitle
  catch-up guard, preventing MPV from outrunning nearby translations.

No FireRedVAD, Haas ASR, DeepFilterNet, pyannote/Sherpa diarization,
UVR separation, FFmpeg audio process or CUDA ASR runtime starts. Old
audio-pipeline source files are retained for cross-branch reference but
are not used by the caption-only entry point.

## Requirements and first run

- Windows with MPV and yt-dlp available in PATH (or MPV configured as before).
- Recent CUDA-enabled `llama-server` in PATH with Hugging Face `-hf`
  loading, Flash Attention and Q8 KV-cache support.
- Python >=3.12 and `uv`.
- First-time setup: `uv sync`. The old ASR lockfile was removed;
  this creates a new, **minimal** `uv.lock` locally.
- Run `Start translate.bat` and enter an archived YouTube VOD URL,
  or run `uv run --no-sync python -m auto_translate "VIDEO_URL" --mode auto --mpv`.

The Windows launcher prints Japanese and English lines, plus two concise
diagnostic lines:

```text
Caption source: JSON3 (download/cache), 1000 raw cues
Caption reconstruction: 1000 raw -> 850 translated units
```

Counts above are illustrative only. If it prints **VTT**, JSON3 was not
available or the download failed and a cached VTT was used. When diagnosing
unreconstructed `わかん / ない` or `思い / ます` splits, see
`Caption split not joined` and `Japanese captions reconstructed` in
`logs/run.log` for the exact original timestamps. The strong-ending joiner now accepts up to 5.5 seconds between adjacent
JSON3 event onsets only for specific clipped Japanese word endings, while
rejecting long silent gaps and spans over 17 seconds. This is a conservative
heuristic, not audio-based word alignment; longer joins may display the full
English sentence at the start of the combined Japanese phrase. For any
remaining split, inspect the logged original cue starts and ends before
increasing timing thresholds again.

Startup and translation diagnostics go to `logs/run.log`; MPV errors go to
`logs/mpv.log`.

## Known constraints

- **Live streams are deliberately rejected:** freely available YouTube
  caption tracks are not a guaranteed continuous live subtitle API, and
  the current MPV live-edge clock has not been reconciled with VTT times.
  This mode does **not** silently switch to ASR.
- Videos without accessible Japanese caption tracks fail with a message.
  Captioned speech omitted by YouTube cannot be recovered.
- Timing and transcription quality still depend on YouTube; reconstruction
  cannot recover speech missing from captions or guarantee word-level alignment.
- Caption display timing remains unmodified by default. For consistently
  late original Japanese captions, compare against the video's audio and
  optionally set `[mpv] caption_advance_s = 0.5` in local `config.toml`
  (seconds; range 0–3). This changes **display lookup only**, not original
  cue times, seeking, or translation text. Do not use it to hide model backlog.
- PLaMo is a specialized translation model, not an instruction-chat model.
  Its documented PFML3 syntax supports previous translation pairs and a
  structured context element without arbitrary system prompts.
- Extra context can improve consistency but increases prefill work and may
  carry over incorrect earlier word choices. The four-pair default is a
  starting point, **not a measured accuracy or latency improvement**.
- To disable conversation history if it hurts quality or speed, set
  `[translate] context_lines = 0` in local `config.toml`.
  Set `context_title_enabled = false` to omit the video-title context.
  A local `config.toml` overrides branch defaults, so check its existing
  `plamo_context`, `max_tokens`, and `context_lines` values.
- Model loading, MPV playback and live caption availability must be
  verified on the actual Windows/RTX 3070 system. Unit tests alone do not
  establish end-to-end correctness or measured translation accuracy.

## Tests

```powershell
uv run --with pytest python -m pytest -q tests/test_caption_formats.py tests/test_caption_only.py tests/test_plamo_translate.py tests/test_translation_config.py
```

---

## Legacy pipeline documentation (other branches)

# Japanese Video Translator — PLaMo 3 Translate 8B Q3_K_M experiment

This **separate test branch** replaces CAT with **PLaMo 3 Translate 8B Q3_K_M**. It keeps the existing Haas-ja-v1.0 Q8_0 Vulkan ASR (CPU decode), FireRedVAD CUDA, Sherpa-ONNX CPU speaker diarization, YouTube JP caption reconciliation, Hololive glossary, direct MPV video playback, English-only subtitle stacking, and VOD seeking/cache. **Only one translation LLM is configured.**

The `main` and `test-cat-translate-3.3b` branches remain untouched for comparison.

## Model and verified prompting

Official model/quantization: https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF

- Repository: `pfnet/plamo-3-translate-8b-GGUF`
- Exact file: `plamo-3-translate-8b-Q3_K_M.gguf` (**4.31 GB model weights**)
- API alias: `plamo-3-translate-8b`
- Port: `http://127.0.0.1:8005/v1` (dedicated, avoids CAT port 8003)
- Model is downloaded automatically on its first launch using llama.cpp's `-hf` and `-hff` options.

The official model card recommends **PFML3** prompts directly to `/v1/completions` for production. The launcher uses the original GGUF embedded chat template for compatibility, but translation requests go to the raw completions endpoint **without any chat wrapper or extra system prompts**. Every utterance is sent in this exact language-targeted structure:

```text
<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>行ってきます<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>
```

Translation stops at `<|plamo:tag|>` or `<|plamo:bos|>`. The input now explicitly tags Japanese (`lang=ja`) as recommended by the model's PFML3 specification; English remains the target (`lang=en`). This is a low-overhead prompting correction, not a guarantee of better translation accuracy. Only glossary terms **actually present in the Japanese transcript** are optionally supplied using the documented PFML3 glossary syntax. The model never receives previous English subtitles, which prevents accidental continuation of prior dialogue.

## Low-VRAM defaults (RTX 3070 8 GB)

The managed llama-server starts with the following flags:

```bat
llama-server -hf pfnet/plamo-3-translate-8b-GGUF -hff plamo-3-translate-8b-Q3_K_M.gguf -c 1024 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja --no-prefill-assistant --host 127.0.0.1 --port 8005 --alias plamo-3-translate-8b
```

- **Q3_K_M:** 4.31 GB of model weights (not total VRAM).
- **1024-token context:** short utterances, no prompt history.
- **1 inference slot:** no duplicate KV allocations.
- **Q8_0 KV cache:** quantized K and V state.
- **Flash Attention:** `-fa on`.
- **GPU offload:** `-ngl 99`, subject to actual VRAM availability.
- **Output:** up to 128 English tokens, temperature 0.
- **Cleanup:** owned server shuts down when MPV/session ends.

Estimated total pipeline memory before measurement: roughly **6–8 GB**; it may exceed 8 GB depending on CUDA buffers, concurrent programs, Windows and the GPU build. A successful launch and the actual VRAM peak are not guaranteed by the unit tests. If full GPU offload fails, reduce `plamo_gpu_layers` in `config.toml` (e.g. 80) to transfer some computation to CPU at a potential speed penalty. An old `llama-server` build may reject `-ctk`, `-ctv` or `-fa on`; upgrade the binary rather than silently disabling the requested optimizations.

### Optional configuration

No `config.toml` is required. Defaults:

```toml
[translate]
plamo_base_url = "http://127.0.0.1:8005/v1"
plamo_model = "plamo-3-translate-8b"
plamo_hf_repo = "pfnet/plamo-3-translate-8b-GGUF"
plamo_hf_file = "plamo-3-translate-8b-Q3_K_M.gguf"
plamo_context = 1024
plamo_gpu_layers = 99
max_tokens = 128
timeout_s = 25.0
load_timeout_s = 600.0
context_lines = 0
memory_lines = 12
lookahead_s = 2.0
fragment_join_enabled = true
fragment_join_gap_s = 0.25
glossary = "glossary.toml"
```

Retired CAT settings (`cat_*`), old Index/Qwen translation keys and obsolete quality-retry fields are ignored in legacy local configuration files. Runtime speech segmentation and subtitle timings are otherwise unchanged.

## DeepFilterNet3 speech enhancement (enabled by default)

DeepFilterNet3 now processes the **audio used by FireRedVAD and Haas ASR** before speech segmentation; MPV continues playing the original audio. FFmpeg provides 48 kHz mono PCM to the existing enhancement worker, which converts it back to 16 kHz for the ASR pipeline. UVR vocal separation remains disabled and cannot be enabled simultaneously.

On first enabled launch, `Start translate.bat` checks for the `deepfilternet-rs` and `scipy` extras and, if missing, installs them with `uv sync --extra diarization --extra deepfilter`. Normal launches continue using `uv run --no-sync`.

**Seek recovery:** The enhancement input queue is now bounded to 16 messages (roughly 0.7 seconds of PCM plus progress markers). An MPV seek cancels queued audio from the previous generation rather than waiting for all of it to finish, and the new 16 kHz stream restarts at the absolute target timestamp. The context stage also releases held playback-progress markers after its wall-time lookahead expires, so a quiet section cannot indefinitely block the next 5-second translation lead.

The effective configuration comes from `config.toml` if one is present, otherwise defaults. Existing **git-ignored** local config values can override the new default; to enable DeepFilterNet in such a file, use:

```toml
[deepfilter]
enabled = true

[separation]
enabled = false
```

For an A/B comparison, set `[deepfilter] enabled = false`. Speech enhancement can add CPU latency and alter consonants or quiet speech; verify whether it actually reduces game-noise ASR hallucinations before promoting it to `main`. Missing dependencies or enhancement failures are errors rather than silently falling back to unfiltered audio.

## Install / run (Windows)

New isolated checkout:

```bat
cd /d C:\Games
git clone --branch test-plamo8b-q3 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-plamo8b
cd /d C:\Games\video-translate-plamo8b
uv sync --extra diarization --extra deepfilter
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
"Start translate.bat"
```

Alternatively, to reuse the existing `C:\Games\video-translate-cat` checkout **without deleting its local files manually**:

```bat
cd /d C:\Games\video-translate-cat
git fetch origin test-plamo8b-q3:refs/remotes/origin/test-plamo8b-q3
git switch -c test-plamo8b-q3 --track origin/test-plamo8b-q3
uv sync --extra diarization --extra deepfilter
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
"Start translate.bat"
```

The launcher still requests the YouTube URL, loads models, opens MPV, and prints JP/EN to the terminal.

## Selective multi-pass Haas ASR (experimental)

The default is **one Haas Q8_0 decode**. Only incomplete, suspicious or strongly caption-disagreeing Japanese triggers additional attempts. A retry uses a small (60ms) boundary adjustment where safe. A third candidate, when available, uses the **same Haas model without keyword boosting**. There is no second ASR model and no unsupported speech.cpp sampling option. When a bounded retry times out, optional retries are suspended for 15 seconds by default to avoid overloading a still-busy worker.

Candidates are selected with a conservative score: independent, strongly aligned YouTube JP captions (when available), Japanese text plausibility and decoder-artifact checks. **Raw character count and majority voting are not used.** If there is no meaningful evidence of improvement, the original result wins. Caption alignment is optional; reliable local audio is still the primary source.

**Live:** at most 2 passes total and up to 1.5s for optional retry work.
**VOD:** at most 3 passes total and up to 3.0s for optional retry work.
All passes are serial; seek changes cancel further optional retries. The deadline is checked before each retry and passed into speech.cpp's per-request response wait; it does not forcibly interrupt an OS pipe write or an already-running speech.cpp decode. The **first pass** retains its existing response timeout (30s by default). Therefore these limits are not a hard upper bound on total ASR latency.

Optional `config.toml` overrides:

```toml
[asr]
retry_uncertain_enabled = true
retry_max_audio_s = 8.0
retry_max_passes = 3
retry_live_max_passes = 2
retry_live_budget_s = 1.5
retry_budget_s = 3.0
retry_timeout_cooldown_s = 15.0
```

Use `retry_uncertain_enabled = false` for a pure single-pass comparison. Benchmark transcription correctness, `asr avg/max`, dropped subtitles and MPV sync on the **same VOD/live section** before concluding that multi-pass helps. A second deterministic decode does not establish independent confidence, and optional retries may still increase latency or background worker load.

## Engine-noise acknowledgement guard (experimental)

A short, isolated Japanese `うん` can be hallucinated by Haas over car-engine noise and translated to `Yeah`. The ASR stage now suppresses these cues only when the input was a VAD-negative background-audio probe or when a conservative, CPU-only signal check finds strongly stationary low-frequency multi-harmonic audio. A reliable, closely aligned Japanese caption explicitly containing the same acknowledgement overrides the guard.

The guard **does not drop all `うん`**. Normal voiced replies, full spoken sentences, non-stationary audio and confirmed acknowledgements pass through as before. VAD-negative probes containing stationary engine audio may be rejected before decoding. Captions explicitly labeled `[engine noise]` or `[エンジン音]` are treated as sound labels.

Optional opt-out for A/B comparison:

```toml
[asr]
engine_noise_ack_guard = false
```

This guard is conservative, not a full engine/speech classifier. Fast-changing engines, speech mixed with engine sounds and genuinely spoken short replies can remain ambiguous. Validate against the same recorded audio and compare false positives and missed speech; terminal text alone cannot prove that every `うん` was generated by engine noise.

## Context-aware live subtitles (experimental)

The ASR-to-translation stage holds a Japanese cue for **at most 2.0s of media progress or 2.0s of wall time** to inspect upcoming Japanese transcripts, not to add model prompt history. The stage uses a maximum 12s recent-speech evidence window and merges very short unfinished utterances only across tiny time gaps without a detected speaker handoff. It does **not** identify the same speaker across different VAD windows, and it cannot reconstruct words absent from the ASR output.

A progress watermark is capped at the oldest still-untranslated cue so MPV does not mistake a held cue for fully translated audio. Buffer and speaker state reset on VOD seeks. Prior completed VOD translations remain cached and are passed through unchanged. The added buffering can increase first-subtitle latency by up to 2 seconds before the model's existing processing time.

The contextual corrections are restricted to observed confusions (Burger King, hideout, ammo, Botan) with contextual evidence where applicable. Uncertain vehicle words such as 前頭部 are intentionally left untouched until audio verification.

### Validation on the same VOD or live recording

```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1
```

While replaying `q8Kbl-HjkQ8`, compare Japanese and English lines around Burger King, ammo, hideout and names. Inspect `logs/run.log` for `done: ... latency avg ... max ...`, `dropping expired subtitle`, `first subtitle received`, `pipeline backpressure`, and `Joined adjacent Japanese fragment` (the last is debug logging). Test at least one backward seek and one speaker handoff. The unit tests are deterministic code checks, **not** a real GPU, model-quality or live-MPV benchmark; record actual peak VRAM and cue-delay measurements on the target PC.

## Diagnostics

- `logs/plamo8b-server.log`: llama.cpp download, model load, unsupported flags, CUDA and memory failures.
- `logs/run.log`: yt-dlp, FFmpeg ingest, Japanese recognition, translation pipeline and MPV diagnostics.
- Inspect GPU memory while running:

```bat
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

The tests mock the model server and verify PFML3 formatting, exact model/quantization, memory flags, glossary matching, cleanup, and baseline pipeline behavior. They **cannot prove actual Q3_K_M quality or VRAM usage**; compare the same saved VOD transcript and runtime logs with CAT and PLaMo 2B.

This branch removes CAT's translator code, its Sarashina template, CAT-specific response-retry rules, model settings and tests. General conservative speech/post-translation safeguards remain.
