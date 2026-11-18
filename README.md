# Haas + Whisper Large-v3 Turbo Japanese dual-ASR

**Improved branch:** `feature/dual-asr-improvements-20261119`, based on the original dual-ASR branch. No ASR, selector, or translator model is changed. See [improvement notes](docs/dual-asr-improvements.md).

## Windows GUI (recommended)

**Double-click `Start GUI.vbs`** in the project folder to open the silent
Windows app. It installs the PySide6 GUI dependency automatically if required.
After the YouTube video loads, the URL controls hide; translations and logs
remain visible under the **Console** tab.

For a compiled frontend, use `dist\\VideoTranslate\\VideoTranslate.exe` from
the **Windows GUI frontend build** workflow artifact (extract it inside the
project folder first). `Setup.bat` is for first-time installation;
`Start translate.bat` remains the optional CLI fallback. See [GUI_SETUP.md](GUI_SETUP.md).


- Haas-ja-v1.0 (Vulkan) remains primary; Whisper Large-v3 Turbo (CTranslate2 `int8_float16` CUDA) replaces Parakeet and runs in a restartable subprocess.
- Original 16 kHz Japanese speech goes to both ASRs. LFM2.5-JP selects A=Haas, B=Whisper Turbo, U=uncertain; PLaMo 3 Translate 2B generates English subtitles.
- Whisper receives **no static Hololive name prompt** (even if an old `config.toml` says `whisper_name_hints = true`). One-sided isolated `ラプラス・ダークネス` claims are suppressed on strong dual-ASR disagreement; corroborated names are kept.
- For Hololive subtitles, `ラプラス`, `ラプラス・ダークネス`, and `ラプラスダークネス` are rendered simply **Laplus** in English. English post-processing also shortens stray `La+ Darknesss` variants when the Japanese contains Laplus; it does not add names based on title alone.
- Spoken nicknames stay distinct: `ラプ` → **Lap**, `ラプ様` → **Lap-sama**, `ラプちゃん` → **Lap-chan**, `ラプラス` → **Laplus**, `みこ` → **Miko**, `みこち` → **Mikochi**, and `わため` → **Watame**. `まつり` needs proper-name context; literal festivals and grass retain their meaning. Context must not invent dialogue, omitted subjects, or gendered pronouns.
- CPU VoxLingua107 still drops only high-confidence English and Indonesian audio before either ASR; it does **not** translate or display either spoken language. Ambiguous code-switching continues as Japanese.
- Physical GPU headroom gating is retained, with no artificial 7.5-GB ceiling. LFM GPU offload falls back to CPU, and a failed secondary gracefully falls back to Haas. PLaMo remains capped at 12 GPU layers on RTX 3070-class cards.
- Download: the public HF CTranslate2 checkpoint `h2oai/faster-whisper-large-v3-turbo` (~1.6 GB) is fetched automatically once. No CrispASR executable or Parakeet GGUF is needed. Logs: `logs/run.log`, `logs/whisper-turbo.log`, `logs/asr-comparison.jsonl`, `logs/reconciler.log`.
- Run `uv sync --locked`, then `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v`, then `Start translate.bat`. Existing `config.toml` may still set 2s live secondary timeout; raise `[dual_asr] live_timeout_s = 5.0` to use Whisper reliably.
- To use CPU when CUDA memory is insufficient: set `secondary_device = "cpu"` and `secondary_compute_type = "int8"` in `[dual_asr]`. This is slower.
- Accuracy and runtime have not yet been verified on the user's RTX 3070; measure the new comparison logs on an actual VOD.

Setup installs Deno and verifies the pinned speech.cpp installer and llama.cpp archives. It downloads the default PLaMo GGUF into the Hugging Face cache. Existing installations can reuse their llama.cpp model cache.

Use `config.example.toml` for supported overrides. The live player tries video stream copy, then NVENC, then software H.264 if the initial HLS segments cannot be written.

## Overhaul improvements (models unchanged)

- **English subtitle guards:** preserve uncensored language and proper names, remove unpaired double quotes, and suppress obviously untranslated Japanese or prompt/template leakage rather than publishing bad English. Cached cues follow the same boundary.
- **PLaMo context continuity:** keep the existing context and glossary system; clear dialogue history only after more than 15 seconds of audio-timeline silence or a seek/generation reset.
- **Subtitle readability:** balanced wrapping and time-sliced long lines, with optional `[mpv]` line length, reading speed, and duration controls. The original GUI, MPV hardware decoder and RTX VSR code remain intact.
- **VAD:** fix speech-onset timestamps after long-form forced cuts. The original dual-ASR thresholds remain unchanged: transplanting the Qwen branch's more permissive defaults produced game-noise regressions.
- **GPU/diagnostics:** report measured peak used/minimum free GPU memory and per-cue subtitle-display lateness. No model VRAM allocations or hard limits are added.

## Context, correction, and playback diagnostics

The pipeline remains **FireRedVAD → Haas + Whisper Large-v3 Turbo → LFM2.5-JP → PLaMo 3 Translate 2B → English subtitles**. These controls extend the existing GUI and backend; they do not identify speakers or add diarization models.

| Feature | Configuration and behavior |
| --- | --- |
| Translation memory | `[translate] extended_memory = true`, `memory_lines = 16`; retain at most 20 recent Japanese/English pairs. `context_lines = 3` keeps adjacent dialogue while relevance selection supplies bounded older terminology/context. Disable extended memory to retain only ordinary adjacent context. Seeking or starting another video resets memory; later dialogue does not revise displayed subtitles. |
| Smart disagreement correction | `[dual_asr] smart_correction = true`; `correction_vod_retries = 1`, `correction_live_retries = 0`; separate `correction_vod_budget_s = 8.0` and `correction_live_budget_s = 2.0`. Retry only a localized weak audio span when the recognizers meaningfully disagree. No extra model loads. Retries must respect cancellation and playback deadlines; uncertainty can result in abstention. |
| YouTube chat context | `[chat] enabled = false`; optional **Use YouTube Chat for Context** control. `window_s = 30.0`, `max_messages = 100`, `poll_interval_s = 5.0` bound asynchronous work. Only sanitized candidate-present terminology can become weak evidence; raw messages are never model instructions or spoken subtitles. |
| Safe Playback | `[safe_playback] enabled = false`; default `max_inference_concurrency = 1`, `queue_size = 4`, `audio_chunks = 32`, `max_audio_lead_s = 20.0` bound inference and prefetch when enabled. This preserves MPV hardware decoding, selected resolution, subtitle timing, and the independent RTX VSR setting. |
| Diagnostics | `[diagnostics] enabled = true`, `sample_interval_s = 2.0`, `max_bytes = 2000000`, `backups = 3`. The GUI displays GPU observations, stage timings, subtitle queues/counters, playback facts, and owned workers. Unsupported metrics show as unavailable. |

Install optional NVML telemetry with `uv sync --locked --extra diagnostics` (included by `Setup.bat`). Missing NVML falls back to available NVIDIA memory observations; translation continues when monitoring is unavailable. Memory figures describe the whole GPU's dedicated device memory, not just this application's allocation; WDDM may make some metrics unavailable. GPU pressure uses measured free memory, without an artificial total-VRAM ceiling. Safe Playback coordinates application inference; it cannot serialize MPV's driver work or guarantee freedom from driver hangs.

Public live/replay chat is best-effort and may be disabled, members-only, delayed, or inaccessible. YouTube's public chat protocol may change. Replay uses available video offsets. Live PCM receipt can only estimate a source-to-wall-clock anchor: encoder/network delay remains uncertain, so this is labeled `receipt_estimate`. `[chat] live_time_uncertainty_s = 5.0` (range 0–120 seconds) delays uncertain messages by a conservative configured allowance. It cannot prove or fully bound the actual upstream delay. The pipeline continues without chat when retrieval or usable timestamp alignment is unavailable. Chat never overrides strong speech evidence, supplies unheard words, or retroactively repairs subtitles.

The optional diagnostics export collects bounded application log tails and can include narrowly filtered Windows display/watchdog/WER event metadata when explicitly selected. It does not collect cookies, chat identities, media, model caches, or unrelated files. Application logs can contain subtitle text and video metadata: review an export before sharing it. Structured diagnostics and GUI logs rotate during use; inherited helper-process logs rotate between process starts and can grow during a long session. An application cannot reliably write logs after the whole operating system freezes.

## Validation on Windows 11 / RTX 3070

Offline regression tests use deterministic mocks for memory bounds, conflicting names, retries, chat failures, GPU monitoring failures, seek generations, model failures, and cleanup. They do not establish that NVIDIA WATCHDOG / `0x141` freezes are resolved. The Windows GUI build checks frontend behavior and packages the executable; actual playback still needs the physical system.

1. Compare a Japanese VOD with memory enabled/disabled. Include repeated Lap/Laplus, Matsuri/festival, Miko/shrine-maiden, literal grass/laughter, incomplete sentences, and overlap; inspect disagreement reasons and translation consistency.
2. Seek forward/back while ASR or translation is active, then switch videos. Confirm no old subtitles, names, or context leak into the new playback generation.
3. Test Safe Playback on/off at each requested resolution with RTX VSR independently on/off. Confirm hardware decoder status and NVIDIA's VSR activity indicator, AV sync, subtitle latency, and stable GPU memory over a long session.
4. Test missing/disabled chat and a public livestream with verifiable timestamps; speech must remain authoritative and translation must continue when chat disappears.
5. Interrupt Whisper/LFM/PLaMo, stop during model startup, and close the GUI during translation. Confirm useful failure diagnostics, bounded recovery, no orphan application-owned processes, and successful restart.
6. If Windows freezes again, record the playback/settings and the last dashboard/log observations. After reboot, export the selected Windows event metadata. The root cause remains unconfirmed until hardware evidence supports it.

[Historical changes and experiments](CHANGELOG.md)
