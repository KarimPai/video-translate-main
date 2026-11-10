# Whisper Large-v3 Direct Japanese Audio-to-English Translation (AST)

Branch: whisper-v3. This is a **single-model** offline translation pipeline. Only CTranslate2/faster-whisper multilingual Whisper Large-v3 generates the English subtitles. No separate Japanese ASR stage, external LLM, model-server, or GGUF runtime.

## Pipeline

YouTube live/VOD or local media → FFmpeg 16kHz mono → optional GTCRN (CPU) → FireRedVAD (CUDA) → optional Sherpa speaker turns (CPU) → **Whisper Large-v3 (Japanese audio → English)** → MPV English-only subtitles/terminal.

Whisper uses language=ja, task=translate, condition_on_previous_text=False and vad_filter=False. Old config.toml values requesting transcription are ignored. Previous [translate] settings are ignored. Large-v3 Turbo is not recommended because its model is not trained for speech translation.

Whisper defaults to int8_float16 to reserve GPU memory for MPV's hardware video decoder on the RTX 3070; FP16 remains available as an opt-in quality experiment with an int8_float16 fallback on model-load VRAM exhaustion. Text-only decoding remains enabled because FireRedVAD already provides timestamps. Wholly rejected low-confidence utterances can only be recovered when two independent decodes agree exactly and both the VAD and Whisper scores strongly indicate real speech. This is a guarded improvement, not a guarantee of translation correctness.

Direct AST cannot also supply a verbatim Japanese transcript without a second model pass. The command prompt therefore shows EN: lines only.

## Update and start (Windows)

    cd /d C:\Games\video-translate-whisper
    git fetch origin && git switch whisper-v3 && git pull --ff-only origin whisper-v3
    uv sync --extra diarization
    "Start translate.bat"

Launcher defaults: GTCRN CPU speech enhancement and Sherpa CPU speaker-turn segmentation **enabled** (`--audio-profile both`), --hwdec auto-safe (use GPU-assisted decoding when available; fallback to software), --whisper-compute int8_float16 (prior smooth-playback baseline; opt into FP16 with --whisper-compute float16), --delay 10; no translator server is started. The previous FP16-by-default trial was rolled back after a report of a prolonged video freeze with audio continuing. This is an isolation test; causality is not yet proven. Stop old standalone servers manually if still running to free VRAM.

## GTCRN enhancement and Sherpa speaker turns

The launcher now runs with `--audio-profile both`: GTCRN processes **16 kHz mono audio on CPU before FireRedVAD**; Sherpa-ONNX identifies possible **speaker-turn boundaries within VAD utterances on CPU** before Whisper. Both use `sherpa-onnx`, installed by the launcher using `uv sync --extra diarization` when the package is missing. A separate native import preflight reports a missing DLL/runtime before starting playback. On first use, the official GTCRN and Sherpa ONNX weights download to a local cache; later sessions reuse them. GTCRN does not modify the original MPV soundtrack. Sherpa does **not** assign stable speaker IDs or isolate voices into separate audio tracks. UVR vocal separation remains off.

CPU processing and additional speaker splits can increase subtitle latency compared with the earlier baseline. To return temporarily to the previous audio-processing behavior, change `--audio-profile both` to `--audio-profile baseline` in the launcher. This does not alter MPV quality selection, HTTPS/DASH preference, or Whisper precision.

## YouTube quality selector

`Start translate.bat` now asks for a video quality after the URL: **360p, 480p, 720p, or 1080p**. Press Enter for **720p**. The choice is applied before MPV opens the video, and yt-dlp chooses the best available stream **not exceeding** that height (for example, 480p if 720p is unavailable). A mid-playback quality change is intentionally not offered because reloading the source may disturb native seeking and subtitle timing.

You can also run the CLI with `--video-quality 720`. The launcher choice takes precedence over custom video format selectors in `config.toml`; when the CLI flag is absent, the existing selectors remain unchanged. Finished VODs still prefer HTTPS/DASH and fall back to HLS; live streams still prefer HLS. Whisper audio inference, GTCRN, VAD, and the independent audio ingest format are unaffected.

## Key defaults and silence protections

    [whisper]
    model = "large-v3"
    compute_type = "int8_float16"
    beam_size = 5
    temperature = 0.0
    no_speech_retry_block_threshold = 0.85
    rescue_stable_weak_speech = true

    [vad]
    threshold = 0.50
    min_speech_ms = 250
    min_silence_ms = 500
    speech_pad_ms = 150
    fallback_s = 0.0

    [ingest]
    vod_max_audio_lead_s = 75.0

FireRedVAD gates incoming speech before translation. VAD-negative probes, silence and non-finite audio are skipped. Whisper rejects low-confidence/repetitive segments and preserves no-speech probabilities even for rejected segments. High-confidence silence rejection blocks another warmer decode (default 0.85), while uncertain quiet speech may still retry. Retries decode the **same audio window**, not previous audio context. Standalone generic video outros during gameplay are suppressed. No second Silero VAD or cross-window condition_on_previous_text feedback is enabled.

## MPV / VOD performance

MPV keeps `--hwdec=auto-safe` and uses a separate format selector for each source mode: **completed VODs prioritize 1080p-or-lower H.264 video plus audio over direct HTTPS (DASH-style separate tracks)**, falling back to a direct non-H.264 pairing, then HLS renditions if direct formats are unavailable. Live streams retain the previous HLS-based format and live-edge options. The independent FFmpeg VOD audio reader now likewise prefers HTTPS audio but retains a best-audio fallback. Selection fallback occurs during initial yt-dlp format selection; it is **not** an automatic runtime switch after a network disconnect.

This aims to bypass the HLS video-transport errors observed during a prolonged playback freeze. Hardware decoding, Whisper inference, FireRedVAD, subtitles, VOD cache limits, and seek semantics are unchanged. An older local `config.toml` can override the new `[mpv].vod_ytdlp_format` and `[ingest].vod_ytdlp_format` settings if needed.

Preserved: playback-relative audio read-ahead cap of 75s, MPV demux cache 256MiB forward/64MiB backward, 30s readahead, session subtitle cache indexed by playback second, native seeking, English subtitle stacking (3 lines), 10s launcher delay, original source timestamps, pause-aware watchdog and real worker error reporting. FFmpeg audio ingest yields when safely ahead and resumes as playback catches up.

## Offline regression checks

    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_ast.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_silence.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

These tests use mock models. Actual AST translation fidelity, live mpv behavior, and total VRAM use require a real Windows GPU run.

The application logs `MPV format selection: VOD direct HTTPS/DASH first, HLS fallback` at startup when applicable. This describes the **preference**, not proof of the actual stream selected. Check the full MPV log for final demux/transport details if the freeze recurs.

The MPV VOD health line now logs the actually active video decoder (`hwdec_active`), audio playback timestamp, AV sync, decoder/frame drops, and estimated video filter FPS every 30 seconds. This helps distinguish a video-only freeze from an audio/network/decoder stall. It does not restart playback automatically or change seek behavior. If video freezes while audio continues, save the full logs before restarting, and note the wall-clock timestamp.

To inspect unexpected exits:

    powershell -NoProfile -Command "Get-Content 'logs\run.log' -Tail 60"
    powershell -NoProfile -Command "Get-Content 'logs\mpv.log' -Tail 40"
    nvidia-smi
