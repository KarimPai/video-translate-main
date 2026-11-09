# Whisper Large-v3 — Direct Japanese Audio-to-English Subtitles

Git branch: **whisper-v3**

One local speech model (faster-whisper / CTranslate2) replaces Granite Speech
and its llama.cpp server. No Japanese ASR + translation-LLM cascade.

## Audio / subtitle pipeline

YouTube live/VOD or local file -> FFmpeg 16 kHz mono PCM -> FireRedVAD CUDA
-> Whisper Large-v3 direct Japanese-to-English -> MPV English-only subtitles.

The seek cache, native MPV playback, 3-line subtitle stacking and 10-second
live delay are retained. GTCRN denoising (CPU) and Sherpa speaker-turn
segmentation (CPU) are ENABLED by default for this experiment; both can
be disabled independently. No separate translation model or default keyword bias.

## Recommended defaults (RTX 3070 8 GB)

    [whisper]
    model = "large-v3"
    device = "cuda"
    compute_type = "int8_float16"
    language = "ja"
    task = "translate"
    beam_size = 5
    temperature = 0.0
    no_speech_threshold = 0.6
    log_prob_threshold = -1.0
    compression_ratio_threshold = 2.4
    condition_on_previous_text = false
    repetition_penalty = 1.05
    no_repeat_ngram_size = 3
    max_new_tokens = 160
    reject_low_confidence = true
    min_avg_logprob = -1.20
    max_compression_ratio = 2.4
    noisy_segment_no_speech_prob = 0.60
    noisy_segment_min_avg_logprob = -0.50
    suppress_repeated_signoffs = true
    signoff_cooldown_s = 1200.0

The additional filters reject decoding segments with poor model confidence,
high repetition/compression or a high no-speech score combined with weak
decoding confidence. N-gram/repetition limits and a 160-token ceiling reduce
runaway "well well" or "yes yes" loops. Generic "thank you for watching"
outros are allowed the first time but suppressed if repeated within 20 minutes
of video time. This last rule is heuristic and can hide a real repeated
outro: disable with [whisper] suppress_repeated_signoffs = false.
All drops are logged in logs/run.log; no separate ASR or LLM is loaded.

Large-v3 (not Turbo) is explicitly trained for native speech translation.
The implementation enforces Japanese input and English translation, disables
the second Silero VAD pass, and supplies no name-biasing prompt. The model
stays loaded for the entire playback session. The RTX 3070 may require
closing other GPU-intensive applications to free sufficient VRAM.

## Update an existing Granite experiment checkout

    cd /d C:\Games\video-translate-granite5
    git fetch origin
    git switch --track origin/whisper-v3
    uv sync --extra diarization
    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v

If the local whisper-v3 branch already exists: git switch whisper-v3,
then git pull --ff-only origin whisper-v3, then uv sync --extra diarization.
The directory name does not control the model: the directory can stay named
video-translate-granite5, or you can rename it after closing running programs.

Launch "Start translate.bat", paste a YouTube URL.
The batch file invokes:

    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal

The launcher checks faster-whisper and sherpa-onnx, and installs the
diarization extra if either is absent.
First use downloads the large-v3 weights; cached model inference is offline.
YouTube playback still requires internet. Windows inference requires the
NVIDIA driver, CUDA 12 cuBLAS, and cuDNN 9 for CTranslate2.

## Audio enhancement experiment (enabled by default)

    [gtcrn]
    enabled = true
    num_threads = 1

    [speaker]
    enabled = true
    num_threads = 2

    [vad]
    threshold = 0.50
    neg_threshold = 0.30
    min_silence_ms = 500
    min_speech_ms = 250
    onset_ms = 80
    speech_pad_ms = 150
    prob_smooth_frames = 5
    fallback_s = 0.0

The previous [granite], [asr], [translate], and [captions] blocks in any
existing git-ignored config.toml are ignored, not executed. Explicit [vad],
[gtcrn], and [speaker] overrides still take precedence over defaults.
To compare with the previous untreated-audio baseline, create config.toml
containing [gtcrn] enabled = false and [speaker] enabled = false (using
normal multiline TOML tables).

GTCRN enhances the audio fed to VAD and Whisper; MPV playback retains
the original source audio. Sherpa detects likely speaker changes and may
split longer utterances, but does not label or identify Hololive members.
Neither component is guaranteed to improve accuracy: denoising may degrade
clean voices, and excessive diarization cuts may harm translation context.
Watch logs/run.log for "GTCRN ready" and terminal output for
"Speaker diarization: Sherpa CPU ready." First use downloads their models.

## Logs / tests

- logs/run.log — application / MPV / model failures and performance statistics
- The terminal prints only EN: subtitles by default.
- No llama-server.exe, GGUF, or outgoing speech inference HTTP requests.
- Unit tests mock model calls and do not download Whisper weights.
- Real translation accuracy, latency and GPU memory need a local playback test.
