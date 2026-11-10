# Whisper Large-v3 Japanese ASR + PLaMo 3 Translate 2B

Branch: **whisper-v3**. Experimental offline Japanese-to-English MPV subtitles
for YouTube live streams, VODs, and local video/audio files.

## Active speech and translation pipeline

YouTube audio -> FFmpeg 16 kHz mono -> optional GTCRN (CPU)
-> FireRedVAD (CUDA) -> optional Sherpa speaker-turn segmentation (CPU)
-> **Whisper Large-v3 (CUDA; JAPANESE TRANSCRIPTION ONLY)**
-> **PLaMo 3 Translate 2B (GGUF Q4_K_M; JAPANESE to ENGLISH)**
-> MPV English subtitles and terminal JP/EN.

Whisper uses task="transcribe" and language="ja". Whisper's task="translate"
is *never used* by the launcher or active five-worker pipeline. PLaMo is the
sole English translator. Both models run locally, without Ollama.

MPV native playback, VOD seeking, session subtitle reuse, three-line English
stacking and the branch's 10-second playback delay are retained. The
Japanese transcript and final English translation are logged separately.
The earlier direct-translation test helper is retained for regression
compatibility but is not connected to the active run() function.

## RTX 3070 8 GB defaults

    [whisper]
    model = "large-v3"
    device = "cuda"
    compute_type = "int8_float16"
    language = "ja"
    task = "transcribe"
    beam_size = 5
    temperature = 0.0
    condition_on_previous_text = false

    [translate]
    plamo_hf_model = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    plamo_context = 4096
    plamo_gpu_layers = 16
    context_lines = 8
    memory_lines = 24
    fragment_context_lines = 2
    fragment_max_gap_s = 1.8
    protect_numbers = true
    protect_names = true

The default Start translate.bat uses --audio-profile baseline (disables
optional GTCRN and Sherpa speaker-turn processing), --hwdec no (software
video decoding), and --plamo-gpu-layers 16 to reduce GPU memory pressure.
This is a temporary diagnostic/stability profile, not a proven solution
to MPV crashes. To change these defaults manually, edit the .bat options.

An old local config.toml containing [whisper] task="translate" is ignored.
Japanese transcription is enforced. Other explicitly set local settings
(e.g. beam_size) still override the defaults.

Whisper and llama.cpp/PLaMo VRAM usage has not yet been measured together on
an RTX 3070: the total may exceed 8 GB depending on CUDA allocations,
running applications, and the model server. PLaMo context reduction to 4096
applies when **this project starts its own** llama-server. If another
llama-server is already running at port 8000, it is reused with its existing
settings and context size; shut down that server if you require 4096.

To reduce load while benchmarking, disable optional CPU processors in
config.toml with [gtcrn] enabled=false and [speaker] enabled=false, or use
--audio-profile baseline. Neither affects Whisper's role as Japanese ASR.

## Run on Windows

Install uv and a GPU-enabled llama-server executable available on PATH.

    cd /d C:\Games\video-translate-whisper
    git fetch origin
    git switch whisper-v3
    git pull --ff-only origin whisper-v3
    uv sync --extra diarization
    "Start translate.bat"

If creating a new directory instead:

    cd /d C:\Games
    git clone --branch whisper-v3 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-whisper
    cd /d C:\Games\video-translate-whisper
    uv sync --extra diarization
    "Start translate.bat"

On first launch, faster-whisper downloads Large-v3 if it is not cached.
PLaMo uses the same pfnet GGUF reference as the existing main/Reazon project;
llama.cpp normally reuses the model already stored in its shared cache.
Do not copy or redownload a working model unnecessarily.

The .bat launcher writes stderr and Python diagnostics to logs/run.log.
The llama.cpp server writes to logs/plamo-server.log.

## Fidelity and subtitle synchronization

- Whisper Latin-only recognition output is discarded before PLaMo translation.
  Forced Japanese decoding does not prevent code-switching on its own; mixed
  Japanese/English speech containing Japanese script remains supported.
- Whisper transcribes Japanese only and does not use English history as an
  initial_prompt or hotword list, minimizing language-mode contamination.
- Short speech fragments can use preceding Japanese cues as optional PLaMo
  context without re-transcribing or retiming earlier subtitles.
- PLaMo uses the 315-term Hololive/game glossary. Explicit Japanese numbers
  can be corrected after translation if PLaMo changes their value, without
  guessing what Whisper heard.
- Both history and nearby-fragment context reset on a seek and on generation
  changes. Speaker handoffs reset near-fragment look-behind.
- Cached VOD subtitles with both JA and EN can be reused without another
  Whisper/PLaMo call. English subtitles stay aligned with original audio
  timestamps; no artificial hard offset is added.
- Suspicious ASR clips can be retried on the SAME current audio segment
  with a different temperature. No past English or expanded audio window is
  mixed into the transcript.

## Tests

    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_plamo.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

These tests use fake models and do not benchmark actual GPU inference, total
VRAM, true Hololive transcription accuracy, model-loading compatibility,
or MPV playback. Test the complete runtime on Windows before treating this
branch as a stable replacement for main.


## Diagnosing sudden MPV closure

On this branch, an application/pipeline failure may shut down a still-running
mpv.exe deliberately. New logs distinguish this from an mpv.exe process crash.

After a closure:

    powershell -NoProfile -Command "Get-Content 'logs\run.log' -Tail 60"
    powershell -NoProfile -Command "Get-Content 'logs\mpv.log' -Tail 40"
    powershell -NoProfile -Command "Get-Content 'logs\plamo-server.log' -Tail 30"
    nvidia-smi

A true mpv.exe crash shows a nonzero player exit code in logs/run.log. If it
states "MPV was still running when application cleanup began", the pipeline
initiated closure instead. GPU snapshots report used/free/total MiB where
nvidia-smi is available. No Windows crash cause is assumed without logs.

If another llama-server process is still responding at port 8000, the
application reuses it and **cannot** enforce the lower -ngl/4096 settings.
Stop that server before launching a fresh low-VRAM test, otherwise an old
high-VRAM PLaMo process can invalidate the stability profile.
