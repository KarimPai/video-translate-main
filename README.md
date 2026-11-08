# Japanese Video Translator — PLaMo 3 Translate 8B Q3_K_M experiment

This **separate test branch** replaces CAT with **PLaMo 3 Translate 8B Q3_K_M**. It keeps the existing Haas-ja-v1.0 Q8_0 Vulkan ASR (CPU decode), FireRedVAD CUDA, Sherpa-ONNX CPU speaker diarization, YouTube JP caption reconciliation, Hololive glossary, direct MPV video playback, English-only subtitle stacking, and VOD seeking/cache. **Only one translation LLM is configured.**

The `main` and `test-cat-translate-3.3b` branches remain untouched for comparison.

## Experimental: verified speaker context and stricter caption repair

This branch `experiment/plamo8b-context-captions` adds three optional-quality
improvements over `test-plamo8b-q3`. The original PLaMo 8B Q3_K_M branch
remains an unchanged comparison baseline.

### Speaker-aware Japanese reference context

- Sherpa still does CPU speaker turn splitting. A second **CPU** embedding
  check, using the already downloaded NeMo model, compares **only immediately
  adjacent voice segments**. Sherpa's local per-window speaker numbers are
  **never** treated as stable speaker identities across windows.
- A short Japanese `context` element is added to the official PLaMo PFML3
  prompt only if the last verified voice matches the current voice,
  their timestamps are consecutive (gap <= 1.25 seconds), and there was no
  confirmed speaker handoff. At most two prior Japanese utterances / 140
  characters are included. No previous English model outputs are injected.
- Missing embeddings, short audio (under 1.5 seconds), interrupted speakers,
  unrelated voices, long gaps, skipped translations, errors and VOD seeks
  reset context. Nothing is displayed or stored as a person's identity.
- On CPU, inference may be slower; the effect on natural language accuracy
  and timing must be measured. Disable without reverting other changes:

```toml
[speaker]
context_enabled = false
```

### Audio-aligned YouTube caption correction

- Haas remains the primary transcript. A reliable VTT match now also allows
  up to **two tiny text insertions/replacements**, if audio-time overlap is
  at least **0.75 s**, utterance coverage at least **90%**, and the complete
  Japanese strings agree at least **90%**.
- No broad caption replacement. Disagreements, changing numbers/negation,
  untrusted sound cues, long rolling VTT captions, and clearly different
  sentences retain the Haas text. Existing accurate corrections and caption
  fallback remain intact.
- This is **not forced alignment to the audio waveform**: the timestamps
  come from YouTube VTT and source PCM, so captions may still be inaccurate
  or unavailable on some streams.

### Filtering laughter and nonverbal audio

- After Haas/YouTube caption reconciliation and proper-name fixes, the
  subtitle pipeline removes **obvious laughter fragments inside otherwise
  meaningful Japanese speech** before PLaMo inference. Example:
  `何それわかんないアハハ` → `何それわかんない`.
- Laughter-only utterances (e.g. `アハハ`, `ふふふ`, `www`) are
  discarded without model inference; isolated short exhale laughs such
  as `フッ` are also suppressed.
- Existing explicit sound-event labels (`[笑い声]`, `[音楽]`,
  `[拍手]`) and VAD-negative noise/tone filtering remain active.
- This is deliberately **not a generic onomatopoeia ban**:
  `来い来い来い`, `待って待って`, `笑った`, `草原`,
  and actual conversational content are preserved. Laughter that is
  only audible but not reflected in the ASR text cannot be reliably
  removed by this text filter.
- This runs before the PLaMo translator and short-term speaker context,
  without loading another GPU model or adding translation requests.

### Switch an existing single-branch clone to the experiment

```bat
cd /d C:\Games\video-translate-plamo8b
git fetch origin experiment/plamo8b-context-captions:refs/remotes/origin/experiment/plamo8b-context-captions
git switch -c experiment/plamo8b-context-captions --track origin/experiment/plamo8b-context-captions
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
"Start translate.bat"
```

To restore the original PLaMo 8B baseline, close MPV and run
`git switch test-plamo8b-q3`.

The model size/quantization, 1024-token context allocation, Flash Attention,
Q8 KV cache, one inference slot, MPV transport and subtitle style are
**unchanged**. Only the variable-length reference text and CPU embedding
work may add processing latency.

---

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

## Diagnostics

- `logs/plamo8b-server.log`: llama.cpp download, model load, unsupported flags, CUDA and memory failures.
- `logs/run.log`: yt-dlp, FFmpeg ingest, Japanese recognition, translation pipeline and MPV diagnostics.
- Inspect GPU memory while running:

```bat
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

The tests mock the model server and verify PFML3 formatting, exact model/quantization, memory flags, glossary matching, cleanup, and baseline pipeline behavior. They **cannot prove actual Q3_K_M quality or VRAM usage**; compare the same saved VOD transcript and runtime logs with CAT and PLaMo 2B.

This branch removes CAT's translator code, its Sarashina template, CAT-specific response-retry rules, model settings and tests. General conservative speech/post-translation safeguards remain.
