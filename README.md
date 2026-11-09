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

## Install / run (Windows)

New isolated checkout:

```bat
cd /d C:\Games
git clone --branch test-plamo8b-q3 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-plamo8b
cd /d C:\Games\video-translate-plamo8b
uv sync --extra diarization
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
"Start translate.bat"
```

Alternatively, to reuse the existing `C:\Games\video-translate-cat` checkout **without deleting its local files manually**:

```bat
cd /d C:\Games\video-translate-cat
git fetch origin test-plamo8b-q3:refs/remotes/origin/test-plamo8b-q3
git switch -c test-plamo8b-q3 --track origin/test-plamo8b-q3
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
"Start translate.bat"
```

The launcher still requests the YouTube URL, loads models, opens MPV, and prints JP/EN to the terminal.

## Selective multi-pass Haas ASR (experimental)

The default is **one Haas Q8_0 decode**. Only incomplete, suspicious or strongly caption-disagreeing Japanese triggers additional attempts. A retry uses a small (60ms) boundary adjustment where safe. A third candidate, when available, uses the **same Haas model without keyword boosting**. There is no second ASR model and no unsupported speech.cpp sampling option.

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
```

Use `retry_uncertain_enabled = false` for a pure single-pass comparison. Benchmark transcription correctness, `asr avg/max`, dropped subtitles and MPV sync on the **same VOD/live section** before concluding that multi-pass helps. A second deterministic decode does not establish independent confidence, and optional retries may still increase latency or background worker load.

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
