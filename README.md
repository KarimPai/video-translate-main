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

The translator uses Index's **short, plain Japanese-to-English translation prompt** for casual English. The current Japanese utterance is the only source translated. Up to **two previous Japanese lines** are provided as explicitly labeled context for meaning and omitted subjects; previous **English translations and the video title are never copied into the prompt**. This avoids passing earlier mistranslations forward.

Only glossary terms appearing in the current utterance are included. High-confidence mistakes (e.g. missing negation, untranslated Japanese, badly phonetic output, or missing priority GTA terminology) trigger **at most one extra Index attempt**, with an independent **4-second HTTP timeout** by default. A second result replaces the first only when it clears the deterministic checks, or the first was empty. This adds latency only to flagged lines; automatic review cannot guarantee the semantics are correct.

Very short unambiguous "オリ曲" / "オリジナル曲" utterances are translated directly to "Original song" instead of being romanized. No new translator or external provider was added.

To prevent **prompt leakage** into subtitles, the llama.cpp request stops at known instruction markers, and output validation truncates any generated `〖Constraint Requirements〗`, `〖约束要求〗`, `〖Source〗`, or similar metadata before the text reaches MPV. A response containing only leaked instructions is treated as a translation failure rather than displayed.

This remains a reliability trade-off: short Japanese context is restored, but scene-title prompting and long instTrans sections stay disabled to minimize instruction leakage and translation of earlier dialogue. Speaker interruptions remain separate ASR/translation requests; Index is told not to complete cut-off speech.

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
context_lines = 2
memory_lines = 12
quality_retry_enabled = true
quality_retry_timeout_s = 8.0
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
interruption_handoffs = true
interruption_min_continuation_s = 0.25
```

These are existing defaults. Haas guarded vocabulary boosting helps with Hololive names; if unsupported by the installed speech.cpp version, it retries without hints. FireRedVAD prefers natural pauses and conserves PCM during forced cuts. Sherpa detects speaker changes without inventing persistent speaker identities. When speaker 2 clearly interrupts speaker 1 and continues speaking, it ends speaker 1's audio at speaker 2's onset; each segment is recognized and translated independently. The original English cue also disappears on a confirmed speaker handoff, including after a cached VOD seek. Normal three-line stacking remains for speech without such a handoff. Set `interruption_handoffs = false` to restore conservative mixed-overlap behavior.

**Limitation:** This is speaker-turn editing, **not voice source separation**. If speaker 1 continues talking audibly while speaker 2 begins, speaker 2's audio may still contain both voices. Near-simultaneous speakers, very short turns and uncertain overlaps remain mixed rather than falsely attributed to one speaker.

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

Tests mock the speech.cpp and llama.cpp processes; the suite does **not** prove translation accuracy or end-to-end GPU integration.

## Optional Index vs Qwen3.5 comparison

A fixed **18-line Japanese GTA sample** is available at `tests/data/gta_translation_eval.jsonl`. The optional `tools/compare_translation_models.py` sends the **same source text to each model**, saves English and response latency, and prints the results side by side. It **does not** run a second model inside the live subtitle pipeline.

Run **one model at a time** on an 8 GB GPU; leave its existing local llama.cpp server active while collecting its results. These commands only attach to already-running model servers—they do not launch or replace them:

```cmd
uv run --no-sync python tools/compare_translation_models.py run --backend index --url http://127.0.0.1:8002/v1 --output logs/index-eval.jsonl
```

Stop Index, then start Qwen in its own independent checkout (using its actual llama.cpp server port; 8001 is an example):

```cmd
uv run --no-sync python tools/compare_translation_models.py run --backend qwen --url http://127.0.0.1:8001/v1 --output logs/qwen-eval.jsonl
uv run --no-sync python tools/compare_translation_models.py compare --index logs/index-eval.jsonl --qwen logs/qwen-eval.jsonl
```

The sample includes suggested readings for manual checking and several lines with potentially garbled ASR; these are **not verified audio transcripts**. The comparison script intentionally reports no automated accuracy score. Review the original audio for ambiguous Japanese. For best fairness, compare the same fixed Japanese text, keep both server settings documented, and do not compare end-to-end latency to the timing of an unrelated live VOD.

## Branches

- `main` — stable pipeline.
- `test` — existing test translator pipeline.
- `test-qwen35-4b` — previous Qwen3.5-4B translator experiment.
- **`test-index-translate-2b` — isolated Index-Translate-2B Q4_K_M experiment.**
