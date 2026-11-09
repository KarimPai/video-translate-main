# Granite Speech 5 — direct Japanese audio-to-English subtitles

Experimental branch: experiment/granite-speech-5-direct

A standalone, local Japanese speech translator built on IBM Granite Speech
5.0 2B Q8_0. There is no intermediate Japanese ASR or second translation
LLM. The model reads Japanese audio and writes English directly.

## Pipeline

    YouTube VOD/live or local file
        -> FFmpeg: 16 kHz mono audio
        -> FireRedVAD (CUDA): bounded utterances, no noise probes by default
        -> Granite Speech 5.0 2B Q8_0 (llama-server, audio->English)
           with IBM-format keyword biasing for common Hololive names
        -> MPV English subtitles / terminal EN-only / optional browser

GTCRN denoising and Sherpa speaker segmentation are OFF by default, but can
be enabled individually. There is no Haas, PLaMo, Kotoba, Whisper, Japanese
caption matching, second translation model, or separate glossary processor.

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
    uv sync

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
    keywords = ["Watame", "Towa", "Suisei", "Pekora", "Miko", "Subaru"]
    # llama_server_path = "C:\\tools\\llama.cpp\\llama-server.exe"

When keywords is nonempty, the model uses IBM's documented AST keyword
template: "translate the speech to English. Keywords: Watame, Towa, ...".
This replaces the punctuation prompt, rather than concatenating unsupported
instructions. Set `keywords = []` to use the `prompt` setting instead.
Keep the keyword list limited to names relevant to the stream; irrelevant
names can bias model output. Keyword biasing is not a guarantee of accuracy.

Default optional settings (also usable in an existing local config.toml):

    [gtcrn]
    enabled = false

    [speaker]
    enabled = false

    [vad]
    fallback_s = 0.0

Set `[gtcrn].enabled = true` or `[speaker].enabled = true` to opt back
in. Either feature requires `uv sync --extra diarization` for sherpa-onnx.
If an existing git-ignored config.toml sets enabled=true, that local
override still takes precedence over the new defaults.
Existing [asr] and [translate] sections are ignored.

VAD defaults are 0.30 speech, 0.20 negative threshold, 400 ms silence,
200 ms pad, 100 ms minimum speech, 7 s utterance (max 8 s). Audio is
segmented locally before Granite inference. VAD-negative probes are off by
default because non-speech music/game sounds can lead to fabricated words.
For quiet voices missed by VAD, optionally set [vad] fallback_s = 5.0.

## Experiment boundaries

- English-only direct output; Japanese transcription is not generated.
- VOD rewind/seek cache and MPV speaker handoff behavior remain.
- No YouTube captions or separate text translation / glossary corrections.
- First run downloads weights and starts the server.
- RTX 3070 throughput and quality require Windows runtime benchmarking;
  these changes are not proof of higher translation accuracy.
- API: POST /v1/audio/transcriptions with multipart fields model, prompt, file.

## Tests

    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

Model-independent audio/MPV tests and Granite regression tests are retained.
Mocked tests cannot prove a particular llama.cpp binary supports this model.
