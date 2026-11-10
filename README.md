# Whisper Large-v3 Direct Japanese Audio-to-English Translation (AST)

Branch: whisper-v3. This is a **single-model** offline translation pipeline. Only CTranslate2/faster-whisper multilingual Whisper Large-v3 generates the English subtitles. No separate Japanese ASR stage, external LLM, model-server, or GGUF runtime.

## Pipeline

YouTube live/VOD or local media → FFmpeg 16kHz mono → optional GTCRN (CPU) → FireRedVAD (CUDA) → optional Sherpa speaker turns (CPU) → **Whisper Large-v3 (Japanese audio → English)** → MPV English-only subtitles/terminal.

Whisper uses language=ja, task=translate, condition_on_previous_text=False and vad_filter=False. Old config.toml values requesting transcription are ignored. Previous [translate] settings are ignored. Large-v3 Turbo is not recommended because its model is not trained for speech translation.

Whisper uses FP16 quality mode on the RTX 3070 (with int8_float16 fallback on model-load VRAM exhaustion), and text-only decoding because FireRedVAD already provides timestamps. Wholly rejected low-confidence utterances can only be recovered when two independent decodes agree exactly and both the VAD and Whisper scores strongly indicate real speech. This is a guarded improvement, not a guarantee of translation correctness.

Direct AST cannot also supply a verbatim Japanese transcript without a second model pass. The command prompt therefore shows EN: lines only.

## Update and start (Windows)

    cd /d C:\Games\video-translate-whisper
    git fetch origin && git switch whisper-v3 && git pull --ff-only origin whisper-v3
    uv sync
    "Start translate.bat"

Launcher defaults: optional GTCRN/Sherpa disabled, --hwdec auto-safe (use GPU-assisted decoding when available; fallback to software), --whisper-compute float16 (fallback to int8_float16 only if FP16 model load is out of VRAM), --delay 10; no translator server is started. Disable the FP16 trial with --whisper-compute int8_float16 in the launcher. Stop old standalone servers manually if still running to free VRAM.

## Key defaults and silence protections

    [whisper]
    model = "large-v3"
    compute_type = "float16"
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

MPV uses `--hwdec=auto-safe` to test whether hardware video decoding reduces occasional hitches during HLS playback or seeking. This may use additional VRAM; revert to `--hwdec no` if stability is worse. Whisper inference, translation, VAD, cache limits, and MPV seek behavior are unchanged.

Preserved: playback-relative audio read-ahead cap of 75s, MPV demux cache 256MiB forward/64MiB backward, 30s readahead, session subtitle cache indexed by playback second, native seeking, English subtitle stacking (3 lines), 10s launcher delay, original source timestamps, pause-aware watchdog and real worker error reporting. FFmpeg audio ingest yields when safely ahead and resumes as playback catches up.

## Offline regression checks

    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_ast.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_silence.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

These tests use mock models. Actual AST translation fidelity, live mpv behavior, and total VRAM use require a real Windows GPU run.

To inspect unexpected exits:

    powershell -NoProfile -Command "Get-Content 'logs\run.log' -Tail 60"
    powershell -NoProfile -Command "Get-Content 'logs\mpv.log' -Tail 40"
    nvidia-smi
