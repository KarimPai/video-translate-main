# Japanese Video Translator — CAT-Translate-3.3B experiment

This separate branch runs one Japanese ASR and one local translation LLM:

- **ASR:** Haas-ja-v1.0 Q8_0 (speech.cpp Vulkan0, CPU decoding).
- **Translation:** **CyberAgent CAT-Translate-3.3B**, **Q4_K_M GGUF**, via managed local CUDA llama.cpp.
- **VAD:** FireRedVAD CUDA, with natural-pause splitting.
- **Diarization:** Sherpa-ONNX CPU speaker turns and interruption-aware cuts.
- **Video:** MPV plays the original YouTube stream; English-only stacked subtitles and native VOD seeking.
- **Other:** Hololive names, conservative glossary corrections, caption reconciliation, game-context fixes, VOD subtitle cache.

CAT is the **sole translator**. There is no Index-Translate implementation, Index-style constraint prompting, Qwen model, PLaMo model, Ollama, external translation API, model chooser, or Index-specific retry workflow in this branch.

## Exact model

**Original model (non-beta):** https://huggingface.co/cyberagent/CAT-Translate-3.3b

**Community Q4_K_M GGUF:** https://huggingface.co/Sundowner123/cat-translate-3.3b-gguf

**Exact GGUF filename:** `cat-translate-3.3b-Q4_K_M.gguf` (2,066,389,824 bytes; about 2.07 GB). The community repository also contains 1.4B and Gemma Q4 files. The launcher specifies **`-hf` plus `-hff`** to prevent accidental loading of a different model.

The server uses `--jinja` to apply the GGUF's embedded **Sarashina 2.2 instruct chat template**. CAT's official inference format is a **single USER message**:

```text
Translate the following Japanese text into English.

行ってきます
```

No previous Japanese/English dialogue, video title, long instructions, leaked constraint sections, Index stop markers, or Qwen thinking parameters are injected. **Temperature is 0** for deterministic comparison. CAT is not guaranteed to understand VTuber slang or broken ASR perfectly; validate on actual audio.

### Clean default CAT settings

The configuration file is optional. Model defaults are:

```toml
[translate]
cat_base_url = "http://127.0.0.1:8003/v1"
cat_model = "cat-translate-3.3b"
cat_hf_repo = "Sundowner123/cat-translate-3.3b-gguf"
cat_hf_file = "cat-translate-3.3b-Q4_K_M.gguf"
cat_context = 2048
cat_gpu_layers = 99
max_tokens = 192
timeout_s = 20.0
load_timeout_s = 300.0
context_lines = 0
memory_lines = 12
glossary = "glossary.toml"
```

`context_lines = 0` ensures the CAT LLM sees only the current speech turn. `memory_lines` supports existing non-LLM ASR/context corrections and short-term repeat consistency, **not model prompt history**.

Old translation keys such as `index_*`, `quality_retry_enabled`, `qwen_*`, `plamo_*` and old model selectors are ignored if they appear in an existing git-ignored `config.toml`.

CAT's server runs on **port 8003**, deliberately separate from Index's port 8002. It will **not adopt or terminate** a previously running server on that port.

### Preserved speech segmentation settings

```toml
[asr]
haas_model = "haas-ja-v1.0:q8_0"
haas_device = "Vulkan0"
haas_decode_on_cpu = true
haas_keywords_enabled = true
haas_keywords_file = "haas_vocabulary.toml"
haas_keywords_alpha = 1.5
haas_keywords_guard = true

[vad]
threshold = 0.35
min_silence_ms = 400
min_speech_ms = 100
max_utterance_s = 6.0
max_split_extension_s = 1.0

[speaker]
enabled = true
min_turn_s = 0.55
interruption_handoffs = true
interruption_min_continuation_s = 0.25
```

Speaker 2 interrupting speaker 1 yields **two separately transcribed and translated turns** at a confident takeover boundary. This does not physically isolate simultaneous voices; overlap remains challenging. Previous subtitles stop lingering after a confirmed handoff, including cached VOD seek playback.

## Install in a separate Windows checkout

```bat
cd /d C:\Games
git clone --branch test-cat-translate-3.3b --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-cat
cd /d C:\Games\video-translate-cat
uv sync --extra diarization
"Start translate.bat"
```

Prerequisites: Windows, Python 3.12+, `uv`, ffmpeg/yt-dlp/MPV, CUDA-enabled Torch for FireRedVAD, working Haas/speech.cpp Vulkan ASR, and a **recent CUDA llama-server.exe** in PATH with `-hf`, `-hff`, and `--jinja` support.

The first CAT startup downloads its GGUF if not cached. It remains local afterwards. Model-server startup/download issues are recorded in **`logs\cat-server.log`**. The main pipeline logs to **`logs\run.log`**. The server is automatically terminated when the session ends.

## Offline regression tests

```bat
cd /d C:\Games\video-translate-cat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Tests include the exact CAT translation prompt, API request, correct non-beta GGUF filename, managed server lifecycle, and absence of previous dialogue in prompts. They mock model/ASR processes and **do not prove runtime translation quality**. Recheck the same Hololive VOD and measure meaning, subtitles, and latency manually.

## Experimental branch isolation

- `main`, `test` — unchanged.
- `test-qwen35-4b` — unchanged.
- `test-index-translate-2b` — previous Index model experiment retained.
- **`test-cat-translate-3.3b` — clean CAT-Translate-3.3B replacement.**
