# Granite Speech 5 — direct Japanese audio-to-English subtitles

Experimental branch: experiment/granite-speech-5-direct

A standalone, local Japanese speech translator built on IBM Granite Speech
5.0 2B Q8_0. There is no intermediate Japanese ASR or second translation
LLM. The model reads Japanese audio and writes English directly.

## Pipeline

    YouTube VOD/live or local file
        -> FFmpeg: 16 kHz mono audio
        -> GTCRN denoiser (CPU, enabled)
        -> FireRedVAD (CUDA): bounded utterances, quiet-speech probes
        -> Sherpa speaker-boundary detection (CPU, optional)
        -> Granite Speech 5.0 2B Q8_0 (llama-server, audio->English)
        -> MPV English subtitles / terminal EN-only / optional browser

No Haas, PLaMo, Kotoba, Whisper, Japanese caption matching, legacy
translation prompts, glossary, or separate translation model are used.

## Requirements (Windows 11 + NVIDIA)

- NVIDIA GPU; RTX 3070 8 GB is the target.
- Python 3.12+, uv: https://docs.astral.sh/uv/
- FFmpeg, yt-dlp and MPV on PATH (unchanged from the old project).
- A recent llama.cpp multimodal build providing llama-server.exe on PATH.
  Older llama.cpp releases without Granite Speech 5 audio support will fail.
- Internet on first launch to fetch the official IBM GGUF and multimodal
  projector; cached weights then run offline (YouTube ingestion still needs internet).

The original FireRedVAD, GTCRN and Sherpa models may require their own first-use
downloads. The 8-bit Granite GGUF is chosen for accuracy rather than minimum
VRAM. Model card: https://huggingface.co/ibm-granite/granite-speech-5.0-2b

## Run

    git fetch origin
    git switch -c experiment/granite-speech-5-direct --track origin/experiment/granite-speech-5-direct
    uv sync --extra diarization

Launch Start translate.bat and paste a YouTube live or VOD URL.

The launcher runs:

    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal

The console shows only EN: lines. Logs:
- logs/run.log: application, VAD, MPV and inference diagnostics
- logs/granite5-server.log: model download, llama.cpp startup and runtime errors

Granite server binds to 127.0.0.1:9797, owned and terminated by this app.
Do not run another server on that port.

## Optional config.toml

    [granite]
    model_id = "ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0"
    base_url = "http://127.0.0.1:9797"
    gpu_layers = 99
    context_tokens = 4096
    request_timeout_s = 45.0
    startup_timeout_s = 1200.0
    prompt = "translate the speech to English with proper punctuation and capitalization."
    # llama_server_path = "C:\\tools\\llama.cpp\\llama-server.exe"

Other sections such as [vad], [gtcrn], [speaker], [mpv] still work.
Existing [asr] and [translate] sections are ignored.

VAD defaults are 0.30 speech, 0.20 negative threshold, 400 ms silence,
200 ms pad, 100 ms minimum speech, 7 s utterance (max 8 s). Audio is
segmented locally before Granite inference. VAD-negative audio probes can
reach Granite when they contain energy but lack a confident speech onset.

## Experiment boundaries

- English-only direct output; Japanese transcription is not generated.
- VOD rewind/seek cache and MPV speaker handoff behavior remain.
- No YouTube captions or separate text translation / glossary corrections.
- First run downloads weights and starts the server.
- RTX 3070 throughput and quality require Windows runtime benchmarking.
- API: POST /v1/audio/transcriptions with multipart fields model, prompt, file.

## Tests

    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

Model-independent audio/MPV tests and Granite regression tests are retained.
Mocked tests cannot prove a particular llama.cpp binary supports this model.
