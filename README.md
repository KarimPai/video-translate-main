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

The server uses `--jinja --chat-template-file src/auto_translate/translate/cat_sarashina.jinja` to apply the official **Sarashina 2.2 chat markers** explicitly. This avoids relying on the community GGUF's embedded template. A single USER message is rendered as the sequence `<|user|>` + source + `</s>` + `<|assistant|>`. CAT's official inference format is:

```text
Translate the following Japanese text into English.

行ってきます
```

No previous Japanese/English dialogue, video title, long instructions, leaked constraint sections, Index stop markers, or Qwen thinking parameters are injected. **Temperature is 0 on the first pass**, with a small repetition penalty. CAT is not guaranteed to understand VTuber slang or broken ASR perfectly; validate against actual audio.

### Real-VOD output compatibility

The community CAT GGUF has been observed returning the Sarashina end-of-sequence marker `</s>` **inside the English text**, even when using the officially specified prompt. The CAT request explicitly stops on `</s>`, and output cleanup also strips it if the server exposes the marker. Extra inference is only used on obviously invalid completions, never as regular prompt context.

**General failure protection:** CAT sometimes replies as a translation assistant (for example, asking the user to provide Japanese), or repeats punctuation for many tokens. The pipeline now flags these concrete failures, retries **once with the same Japanese text and a 4-second timeout**, and suppresses the subtitle if the second result is still invalid. Normal lines have **one inference only**. This does not censor legitimate speech or guarantee that fluent but incorrect translations will be detected. Common exact short reactions `なるほど` and `あかんか` are handled locally so they cannot trigger irrelevant assistant-style replies. The default maximum output is now **128 tokens**, down from 192.

For deterministic diagnostics, compare the same saved Japanese sentences before and after the update. This is a compatibility experiment; whether it improves translation accuracy must be established on real VODs.

Three source-evident GTA transcript errors have narrow, deterministic safeguards: `30秒` wrongly translated as `15 seconds`; a standalone `聞こえるな!` wrongly rendered `Don't listen!`; and a standalone `よろしくお願いします!` incorrectly generating an unrelated "reliable/professional" request. These are **not general translation-quality fixes**—other CAT errors, incomplete Japanese or incorrect Haas transcription can remain.

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
max_tokens = 128
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

### When FFmpeg produces no VOD audio

This is an **ingest failure, not evidence of CAT inference failing**. On a first-audio-open error the program now (1) requests a fresh signed YouTube URL and retries, then (2) tries an AAC/m4a audio rendition if the second attempt also fails. Only after all three attempts does it stop. FFmpeg stderr (including HTTP/codec errors) is now written to `logs\run.log`; temporary signed media URLs are redacted.

To inspect the actual failure without pasting long logs:

```bat
powershell -NoProfile -Command "Select-String -Path 'logs\run.log' -Pattern 'ffmpeg:','VOD audio','ingest session ended','pipeline stage' | Select-Object -Last 24 | ForEach-Object { $_.Line }"
```

If all attempts fail, check yt-dlp format availability with `yt-dlp -F "https://www.youtube.com/watch?v=nEdOU2p6fmU"` and verify FFmpeg is accessible with `ffmpeg -version`. Do **not** paste signed URLs returned by `yt-dlp -g`: they contain temporary access tokens.

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
