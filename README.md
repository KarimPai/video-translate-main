# Haas + Whisper Large-v3 Turbo Japanese dual-ASR (2026-11-14)

## Reliability and transcript tools

The review improvements keep disabled/off ASR modes disabled, apply Haas CPU
fallback to failed restarts, retain watchdogs for canceled Whisper jobs, recover
failed transcript selectors, and make seek/shutdown queues cancellation-aware.
PLaMo uses a shared `translate.cue_timeout_s` budget across sentences and retries;
the pipeline distinguishes bounded in-flight work from a stalled stage. Optional
ASR models now warm in the background after MPV startup.

The **Transcript** tab supports search, a needs-review filter, source-ASR
diagnostics, click-to-replay, manual corrections with revision history, and
Japanese/English/bilingual SRT, WebVTT or ASS export. Corrections stay in the
review session until explicitly exported; they do not silently alter translation
context or the running player's subtitle cache. Import corrected files using the
same language setting and original timestamps.

Explicit CLI export (no transcript file is created unless requested):

```sh
uv sync --locked
uv run --no-sync auto-translate VIDEO_URL --export subtitles.srt --export-language bilingual
```

**Live drafts** are opt-in (`--preview`, the GUI checkbox, or `[preview] enabled =
true`). They reuse idle Whisper capacity, pass through the spoken-language gate,
and show provisional Japanese separately in the GUI/browser. Repeated partial
hypotheses establish local agreement. Drafts never enter translation history,
final subtitle exports, or the cache. Final work takes priority; a busy or
unavailable secondary simply produces no draft. Keep this disabled when measuring
baseline performance on an 8 GB GPU.

The Transcript tab's **Performance report** action produces HTML and JSON from
comparison logs; command-line equivalent:

```sh
uv run --no-sync auto-translate --quality-report logs/asr-comparison.jsonl
```

See [PERFORMANCE.md](PERFORMANCE.md) for annotated Japanese CER, latency/fallback
metrics, beam 1/3/6 and sequential/concurrent benchmarks, and measured preset
recommendations. Missing measurements remain unavailable; there are no assumed
RTX 3070 performance gains.

**Refine VOD timing** is a separate, optional Japanese forced-alignment pass using
WhisperX. Stop translation first, use the matching complete local media file, and
save to a new destination. CPU is the default. The first explicit refinement can
download an alignment model and long media needs additional host RAM:

```sh
uv sync --locked --extra refinement
uv run --no-sync auto-translate local-vod.mp4 --refine-subtitles subtitles.srt --export subtitles-aligned.srt
```

The input for refinement must be a bilingual export with Japanese on its first
line (English may be empty for uncertain cues). Export multiline Japanese
separately instead of guessing language boundaries. Alignment refines cue timing;
it does not correct recognition/translation
content or produce English word timings. Model accuracy, preview latency, and
GPU contention still need representative audio and physical hardware validation.

Feature references: [pysubs2](https://github.com/tkarabela/pysubs2),
[Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit) (workflow inspiration),
[JiWER](https://github.com/jitsi/jiwer),
[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) (streaming policy
inspiration), and [WhisperX](https://github.com/m-bain/whisperX). The application
does not vendor their implementations; optional libraries keep their own terms.

**Current branch:** `feature/haas-whisper-turbo-dual-asr-20261114`, based on the Japanese-only language-filter branch.

## Windows GUI (recommended)

For local EXE updates, close the GUI and run `scripts/Update-GUI.ps1` from
PowerShell. It pulls source changes, reuses cached dependencies, verifies a
staged EXE against the source commit/hash, then overwrites and launches the exact
installed EXE. The window title now shows its build commit. It keeps the current
EXE if the build or verification fails and does not download a GUI artifact.

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
- CPU VoxLingua107 still drops only high-confidence English and Indonesian audio before either ASR; it does **not** translate or display either spoken language. Ambiguous code-switching continues as Japanese.
- Physical GPU headroom gating is retained, with no artificial 7.5-GB ceiling. LFM GPU offload falls back to CPU, and a failed secondary gracefully falls back to Haas. PLaMo remains capped at 12 GPU layers on RTX 3070-class cards.
- Download: the public HF CTranslate2 checkpoint `h2oai/faster-whisper-large-v3-turbo` (~1.6 GB) is fetched automatically once. No CrispASR executable or Parakeet GGUF is needed. Logs: `logs/run.log`, `logs/whisper-turbo.log`, `logs/asr-comparison.jsonl`, `logs/reconciler.log`.
- Run `uv sync --locked`, then `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v`, then `Start translate.bat`. Existing `config.toml` may still set 2s live secondary timeout; raise `[dual_asr] live_timeout_s = 5.0` to use Whisper reliably.
- To use CPU when CUDA memory is insufficient: set `secondary_device = "cpu"` and `secondary_compute_type = "int8"` in `[dual_asr]`. This is slower.
- Accuracy and runtime have not yet been verified on the user's RTX 3070; measure the new comparison logs on an actual VOD.

Setup installs Deno and verifies the pinned speech.cpp installer and llama.cpp archives. It downloads the default PLaMo GGUF into the Hugging Face cache. Existing installations can reuse their llama.cpp model cache.

Use `config.example.toml` for supported overrides. The CLI appends application messages to `logs/run.log`. The live player tries video stream copy, then NVENC, then software H.264 if the initial HLS segments cannot be written.

[Historical changes and experiments](CHANGELOG.md)
