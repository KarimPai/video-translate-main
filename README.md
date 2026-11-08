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
<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:msg|>行ってきます<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>
```

Translation stops at `<|plamo:tag|>` or `<|plamo:bos|>`. Only glossary terms **actually present in the Japanese transcript** are optionally supplied using the documented PFML3 glossary syntax. The model never receives previous English subtitles, which prevents accidental continuation of prior dialogue.

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
git fetch origin
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
