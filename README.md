# Japanese Video Translator — Index-Translate-2B experiment

This independent branch contains **one ASR** and **one translation model**:

- **Japanese speech recognition:** Haas-ja-v1.0 **Q8_0**, speech.cpp Vulkan0, CPU decoder.
- **Japanese → English translation:** **Index-Translate-2B Q4_K_M**, local CUDA llama.cpp.
- **Speech segmentation:** FireRedVAD on CUDA, with pause-aware 6–7s boundaries.
- **Speaker-turn splitting:** Sherpa-ONNX on CPU (pyannote segmentation + NeMo TitaNet).
- **Playback:** original YouTube video in MPV, English-only stacked subtitles, synchronized seek support for VOD.

No other translation backends, provider APIs, or model-selection presets. The old translator, its sampling controls, its tests, and its sampling benchmark have been removed **from this branch only**.

## Why Index-Translate-2B

Official model: https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF

The **Q4_K_M** file is approximately **1.31 GB**. It is a GGUF derived from a translation-focused model, not the general-purpose Qwen3.5-4B model. The official client uses **greedy decoding (temperature 0)** and **thinking disabled**.

The translator uses Index's **short, plain Japanese-to-English translation prompt**, with a single brief instruction requesting natural conversational English. Only matching glossary terms are included; long instTrans sections, repeated context, and past English subtitles are not passed to the model.

To prevent **prompt leakage** into subtitles, the llama.cpp request stops at known instruction markers, and output validation truncates any generated `〖Constraint Requirements〗`, `〖约束要求〗`, `〖Source〗`, or similar metadata before the text reaches MPV. A response containing only leaked instructions is treated as a translation failure rather than displayed.

This is a deliberate reliability trade-off: earlier Japanese history and video-title context are temporarily omitted from individual Index requests until the compact prompt is validated against real VTuber audio.

**No vision projector is loaded**; this is text-to-text translation only.

The model may be more conversational, but actual Japanese ASR-based subtitle fidelity and latency must be validated against your existing branch.

## Independent installation (Windows)

Use a **new folder**, separate from the main and Qwen translator checkouts:

```cmd
cd /d C:\Games
git clone --branch test-index-translate-2b --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-index
cd /d C:\Games\video-translate-index
uv sync --extra diarization
"Start translate.bat"
```

Requirements:

- Python 3.12+, uv, ffmpeg, yt-dlp, MPV.
- NVIDIA CUDA-enabled PyTorch for FireRedVAD.
- **Recent CUDA-enabled llama.cpp** with Qwen3.5 architecture support; `llama-server.exe` in PATH.
- `speech.exe` / speech.cpp 0.10+ with Haas and Vulkan support.

The first launch may download the Index Q4_K_M GGUF through llama.cpp and the speech/speaker/VAD model files if not cached. Models stay on your machine after download.

The batch launcher uses automatic YouTube live/VOD detection, original source MPV video, and a translation prebuffer. It runs from the checkout's `src` directory with `uv run --no-sync`. Closing MPV or finishing the VOD closes the owned model processes.

## Translation settings — clean slate

`config.toml` is **git-ignored**. You do not need to create one; defaults are:

```toml
[translate]
index_base_url = "http://127.0.0.1:8002/v1"
index_model = "index-translate-2b"
index_hf_model = "IndexTeam/Index-Translate-2B-GGUF:Q4_K_M"
index_context = 4096
index_gpu_layers = 99
max_tokens = 160
timeout_s = 20.0
load_timeout_s = 300.0
context_lines = 4
memory_lines = 12
glossary = "glossary.toml"
```

**Port 8002** is deliberately separate from older translator experiments. The program refuses to adopt a server already listening on that port, preventing wrong-model reuse and unsafe cleanup.

All old translation fields (model selectors, Qwen or PLaMo references, sampling presets, quality-recheck toggles, and unused context settings) are ignored if you carry over an older `config.toml`. Only the Index fields above take effect.

Index's decoding uses the **official temperature 0** recommendation, not old top-k/top-p/temperature tuning. There is **no Ollama dependency** and **no external translation API**.

## Speech and subtitle settings retained

```toml
[asr]
haas_model = "haas-ja-v1.0:q8_0"
haas_device = "Vulkan0"
haas_decode_on_cpu = true
haas_keywords_enabled = true
haas_keywords_file = "haas_vocabulary.toml"
haas_keywords_alpha = 1.5
haas_keywords_guard = true
response_timeout_s = 30.0

[vad]
threshold = 0.35
neg_threshold = 0.25
min_silence_ms = 400
speech_pad_ms = 200
min_speech_ms = 100
max_utterance_s = 6.0
max_split_extension_s = 1.0
split_pause_min_ms = 80
split_context_ms = 120

[speaker]
enabled = true
num_threads = 2
min_turn_s = 0.55
max_turns = 5
boundary_search_ms = 180
boundary_quiet_ms = 40
boundary_quiet_ratio = 0.45
```

These are existing defaults. Haas guarded vocabulary boosting helps with Hololive names; if unsupported by the installed speech.cpp version, it retries without hints. FireRedVAD prefers natural pauses and conserves PCM during forced cuts. Sherpa detects speaker changes without inventing persistent speaker identities; overlapping voices are not separated into isolated audio sources.

Japanese YouTube captions can still provide cautious correction when aligned and reliable. The existing glossary, Japanese name fixes, GTA/game term corrections, non-verbal noise filtering, VOD subtitle cache and MPV timing are preserved.

## Troubleshooting

- `logs\run.log` — pipeline, speech recognition, translation and synchronization messages.
- `logs\index-server.log` — Index model download, llama.cpp startup, unsupported binaries, and CUDA allocation errors.
- If an older model server occupies **8002**, close it before starting. This branch will not silently use it.
- You can monitor GPU memory with `nvidia-smi` during playback. This branch is expected to use less translator weight memory than the 4B model, but actual runtime allocations depend on context and backend.

## Offline regression tests

```cmd
cd /d C:\Games\video-translate-index
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Tests mock the speech.cpp and llama.cpp processes; the suite does **not** prove translation accuracy or end-to-end GPU integration. For a fair comparison, run the same VOD and compare the Japanese transcript and resulting English subtitles against the original audio.

## Branches

- `main` — stable pipeline.
- `test` — existing test translator pipeline.
- `test-qwen35-4b` — previous Qwen3.5-4B translator experiment.
- **`test-index-translate-2b` — isolated Index-Translate-2B Q4_K_M experiment.**
