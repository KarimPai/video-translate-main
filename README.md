# Kotoba-Whisper v2.2 + PLaMo 3 Translate 2B

**Experimental branch:** `experiment/kotoba-whisper-v2.2-plamo3`

Based on `whisper-v3` at commit `aea3f7e` (Nov 10, 2026). Main and
whisper-v3 are not changed by this experiment.

## Pipeline

YouTube live / 7-hour VOD -> FFmpeg 16-kHz mono -> FireRedVAD (CUDA)
-> **Kotoba-Whisper v2.2** (Japanese transcription only, CTranslate2)
-> **PLaMo 3 Translate 2B Q4_K_M** (local llama.cpp Japanese -> English)
-> MPV (English-only 3-row captions, seek support) and terminal JP/EN.

**Kotoba selection:**

- Original official model: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2
- Community CTranslate2 conversion: https://huggingface.co/RoachLin/kotoba-whisper-v2.2-faster
- Default `[whisper] model = "RoachLin/kotoba-whisper-v2.2-faster"`
- Uses faster-whisper with `int8_float16`, beam 5, Japanese transcribe,
  `condition_on_previous_text = false` and no initial prompts/hotwords.
- The CTranslate2 conversion contains the core v2.2 speech model. It **does not**
  implement the official Transformers `KotobaWhisperPipeline`'s external
  punctuation and diarization stages. The project uses its own FireRedVAD
  and optional Sherpa segmentation; launcher disables optional Sherpa/GTCRN.
- The converted model's FP32 download is approximately **3.03 GB** on first
  use, then quantized at inference. This is a community conversion, not an
  official Kotoba-published faster-whisper release. Validate transcript
  quality against the official checkpoint before treating it as equivalent.
- No dependency on the Anime Whisper Transformers sequence-score path
  (previously linked to a CUDA scatter/gather assertion). Decoder evidence
  comes from faster-whisper's segment log-probability/no-speech fields.

## Current PLaMo 3 configuration

This branch reuses the most recent `whisper-v3` PLaMo 3 integration and
Hololive/glossary improvements. It does not change the translator.

| Parameter | Value |
|---|---|
| Model | `pfnet/plamo-3-translate-2b-GGUF:Q4_K_M` |
| Runtime | local llama.cpp `llama-server` |
| Context | `4096` tokens |
| GPU offload | `-ngl 16` layers |
| History | 8 dialogue turns; 24 memory lines |
| Temperature | `0.0` |
| Response length | 120 tokens |
| Mode | `source_lang="ja"`, `target_lang="en"` |
| Additional context | streaming/game dialogue, source-grounded hints |
| Terminology | matching Hololive/game glossary entries sent in `chat_template_kwargs.glossary` |

The PLaMo GGUF chat-template behavior changed on Nov 8, 2026: specifying
`target_lang="en"` explicitly is important and already implemented here.
PLaMo model: https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF

**Existing server warning:** A responding `llama-server` at
`127.0.0.1:8000` is reused. That means a previously started server can
use different offload/context settings or a cached older GGUF; stop it
before testing this branch if you need a fresh `-ngl 16 -c 4096` instance
with the latest available GGUF chat template.

## Windows setup / first run

Requirements: Windows, uv, FFmpeg, yt-dlp, MPV and GPU-compatible
`llama-server` executable on PATH, plus NVIDIA CUDA libraries installed
by `uv sync` and a working GPU driver.

Clone in a **new folder** so the Whisper large-v3 baseline stays intact:

```cmd
cd /d C:\Games && git clone --branch experiment/kotoba-whisper-v2.2-plamo3 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-kotoba && cd /d C:\Games\video-translate-kotoba && uv sync --no-progress && "Start translate.bat"
```

Future updates in that folder:

```cmd
cd /d C:\Games\video-translate-kotoba && git pull --ff-only origin experiment/kotoba-whisper-v2.2-plamo3 && uv sync --no-progress && "Start translate.bat"
```

Launcher options: audio baseline (optional GTCRN/Sherpa off), MPV
software decoding (`--hwdec no`), 16 PLaMo GPU layers, 10-second live
delay and 5-second translation startup lead. Keeps native seek, 1080p-first
video selection, 75-second bounded VOD audio processing, and fix for
manual pause falsely tripping the 60-second watchdog.

## Compare Japanese transcription accuracy

For a fair evaluation, use the same Hololive VOD and the same exact
speech segments on this branch and `whisper-v3`. Kotoba's core model is
distilled for Japanese, so it may be faster, but **it has not been
established as more accurate on Hololive**. Inspect:

- Original Japanese transcript, missing/gibberish/imagined text and
  silent or musical passages producing stock video outros
- Names such as Watame and exchanges such as 交換, avoiding spurious 強姦
- ASR and PLaMo latency, subtitle lag, dropped frames and GPU memory
- Seek to ~5h/6h50m and manually pause for several minutes

The latest official Kotoba card lists v2.2 as the Japanese ASR checkpoint
and its optional official postprocessing pipeline. No official comparison
with this specific Hololive source has been performed.

## Offline regression tests

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_kotoba_plamo.py" -v
uv run --no-sync python -m unittest discover -s tests -p "test_whisper_plamo.py" -v
uv run --no-sync python -m unittest discover -s tests -p "test_vod_pacing.py" -v
uv run --no-sync python -m unittest discover -s tests -p "test_whisper_long_vod.py" -v
```

These tests inject fake models and do not require downloads. They do
not prove actual CTranslate2 loading, VRAM compatibility, model decoding,
or smooth MPV playback; test on Windows before considering a merge.

## Troubleshooting

- First run may download ~3.03 GB of Kotoba CT2 weights.
- The `WhisperModel` constructor resolves the community model repo from
  Hugging Face and caches it; repeated launches normally reuse cached weights.
- Application errors: `logs\run.log`; MPV: `logs\mpv.log`;
  PLaMo server: `logs\plamo-server.log`.
- If the ASR conversion fails to load, confirm the repo contains
  `model.bin`, `config.json`, `tokenizer.json` and `preprocessor_config.json`
  and verify the CTranslate2 runtime. The official v2.2 Transformers
  model is a separate, non-interchangeable file format.
