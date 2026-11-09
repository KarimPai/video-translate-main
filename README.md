# Japanese Video Translator — PLaMo 3 Translate 8B Q3_K_M

Experimental branch: `test-plamo8b-q3`. The `main` and YouTube captions-only branches
remain separate.

## Runtime architecture

- **Audio enhancement:** streaming **GTCRN** (`gtcrn_simple.onnx`) through
  `sherpa-onnx`, using **CPU** (one thread by default). This completely
  replaces DeepFilterNet3; there is no DeepFilterNet option or dependency.
- **Audio path:** FFmpeg 16 kHz mono PCM → GTCRN → FireRedVAD → Sherpa
  speaker-turn splitting → Haas-ja-v1.0 Q8_0 ASR (speech.cpp Vulkan;
  decoder on CPU) → local PLaMo 3 Translate 8B Q3_K_M → English subtitles.
- MPV plays the original, unfiltered YouTube video/audio. Enhancement affects
  recognition only, never the audio heard in MPV.
- Sherpa uses pyannote segmentation 3.0 INT8 ONNX and TitaNet Small on CPU
  for speaker-turn boundaries inside VAD utterances; it does not separate
  overlapping voices or identify speaker identities throughout the video.
- Japanese YouTube captions are optionally downloaded as an independent
  recognition/context check. The project can transcribe without them.
- A limited VAD-rejected-audio fallback still sends audible idle windows to
  Haas; no second ASR model is loaded.
- GTCRN has a bounded input queue; MPV seeks cancel stale audio, initialize
  a new denoiser state, and reset timestamps for the new generation.

GTCRN is designed to reduce noise (e.g. game engines and environmental sounds).
It can also suppress quiet teammate voices. The more permissive VAD settings
below are **experiments**, not evidence of better word error rate on Hololive
recordings. Compare quiet speech recall and game-noise hallucinations on a
fixed VOD segment.

## FireRedVAD defaults

| Parameter | Value |
|---|---:|
| Speech threshold | 0.30 |
| Negative/silence threshold | 0.20 |
| Minimum speech | 100 ms |
| Minimum silence | 400 ms |
| Speech padding | 200 ms |
| Soft maximum utterance | 7 s |
| Maximum extension | 1 s |
| Split-pause minimum | 80 ms |
| Split context | 120 ms |
| VAD-rejected fallback interval | 5 s |
| Fallback minimum length | 1.5 s |
| Fallback minimum RMS | −50 dBFS |

## Local settings

The optional `config.toml` overrides branch defaults:

```toml
[gtcrn]
enabled = true
num_threads = 1
# model_path = "C:/path/to/gtcrn_simple.onnx"  # optional manual override

[vad]
threshold = 0.30
neg_threshold = 0.20
min_silence_ms = 400
speech_pad_ms = 200
min_speech_ms = 100
fallback_s = 5.0
fallback_min_s = 1.5
fallback_min_db = -50.0

[separation]
enabled = false

[translate]
plamo_model = "plamo-3-translate-8b"
plamo_hf_repo = "pfnet/plamo-3-translate-8b-GGUF"
plamo_hf_file = "plamo-3-translate-8b-Q3_K_M.gguf"
plamo_context = 1024
plamo_gpu_layers = 99
```

Legacy `[deepfilter]` configuration values are intentionally ignored; use
`[gtcrn]` to enable or disable enhancement. GTCRN and UVR separation are
mutually exclusive, to avoid stacking suppression models.

The official Sherpa `gtcrn_simple.onnx` (535,638 bytes, ~523 KiB) downloads
on first enabled run to `%LOCALAPPDATA%\\auto-translate\\gtcrn` on Windows.
The model is cached and reused between streams and seeks. An optional
`model_path` points to an already downloaded compatible GTCRN ONNX file.

## Installing and launching on Windows

Requires Python >= 3.12, `uv`, MPV, FFmpeg, yt-dlp, a compatible speech.cpp
Haas installation, and a recent CUDA `llama-server` for PLaMo.

The `Start translate.bat` launcher checks that `sherpa-onnx` has the streaming
GTCRN API (`OnlineSpeechDenoiser`). If not available, or if the branch lockfile
is absent, it runs `uv sync --extra diarization` to install the supported
version. You can also run this manually:

```bat
cd /d C:\Games\video-translate-plamo8b
uv sync --extra diarization
"Start translate.bat"
```

GTCRN requires `sherpa-onnx>=1.12.30`. It reuses the package already used
for CPU speaker diarization, with no extra CUDA model or PLaMo GPU allocations.

PLaMo model:
`pfnet/plamo-3-translate-8b-GGUF` →
`plamo-3-translate-8b-Q3_K_M.gguf`, approximately 4.31 GB weight file.
The local llama-server uses Flash Attention, quantized KV state, and
`plamo_gpu_layers=99` by default, subject to available RTX 3070 VRAM.

## Regression tests

```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Tests use fake GTCRN and Sherpa objects and verify configuration, audio
format, progress markers, EOF, and seek cancellation. They do **not** prove
GTCRN improves recognition accuracy or real-time performance; test on actual
GTA clips and monitor `logs\\run.log` and GPU memory after pulling.
