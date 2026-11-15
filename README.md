# Haas + Whisper Large-v3 Turbo Japanese dual-ASR (2026-11-14)

**Current branch:** `feature/haas-whisper-turbo-dual-asr-20261114`, based on the Japanese-only language-filter branch.

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
- CPU VoxLingua107 still drops only high-confidence English and Indonesian audio before either ASR; it does **not** translate or display either spoken language. Ambiguous code-switching continues as Japanese.
- Physical GPU headroom gating is retained, with no artificial 7.5-GB ceiling. LFM GPU offload falls back to CPU, and a failed secondary gracefully falls back to Haas. PLaMo remains capped at 12 GPU layers on RTX 3070-class cards.
- Download: the public HF CTranslate2 checkpoint `h2oai/faster-whisper-large-v3-turbo` (~1.6 GB) is fetched automatically once. No CrispASR executable or Parakeet GGUF is needed. Logs: `logs/run.log`, `logs/whisper-turbo.log`, `logs/asr-comparison.jsonl`, `logs/reconciler.log`.
- Run `uv sync --locked`, then `uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v`, then `Start translate.bat`. Existing `config.toml` may still set 2s live secondary timeout; raise `[dual_asr] live_timeout_s = 5.0` to use Whisper reliably.
- To use CPU when CUDA memory is insufficient: set `secondary_device = "cpu"` and `secondary_compute_type = "int8"` in `[dual_asr]`. This is slower.
- Accuracy and runtime have not yet been verified on the user's RTX 3070; measure the new comparison logs on an actual VOD.

Setup installs Deno and verifies the pinned speech.cpp installer and llama.cpp archives. It downloads the default PLaMo GGUF into the Hugging Face cache. Existing installations can reuse their llama.cpp model cache.

Use `config.example.toml` for supported overrides. The CLI appends application messages to `logs/run.log`. The live player tries video stream copy, then NVENC, then software H.264 if the initial HLS segments cannot be written.

[Historical changes and experiments](CHANGELOG.md)
