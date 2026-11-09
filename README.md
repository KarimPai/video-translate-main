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
be disabled independently. No separate translation model. A short, metadata-selected Hololive name hint
and conservative concise-English formatting are enabled by default. Global
member-list prompting is deliberately not used.

## Recommended defaults (RTX 3070 8 GB)

    [whisper]
    model = "large-v3"
    device = "cuda"
    compute_type = "int8_float16"
    language = "ja"
    task = "translate"
    beam_size = 6
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

## Context, recovery, and ablation controls

Use `--audio-profile baseline|gtcrn|speaker|both` to compare preprocessing
on identical video sections; `--diagnostics` logs every decoded utterance.
Default: both GTCRN and Sherpa enabled.

The current context implementation uses an optional 1.2 s rolling
speech tail for *uncertain decode retries*. It requires exact overlap between
the earlier accepted English and the contextual re-decode before it can
publish a new line; otherwise it retains the original decode.
`use_text_context = false` deliberately stays off by default because
unverified prior English text can propagate hallucinations. Enable only for
an A/B test. Optional [whisper] hotwords = ["Okayu", "Botan"] require the
actual stream to contain those names; do not enable names globally.


## Context, retry, and decoder options

Additional optional [whisper] settings:

    [whisper]
    beam_size = 6
    retry_uncertain = true
    retry_on_outro = true
    retry_temperature = 0.2
    audio_context_enabled = true
    audio_context_s = 1.2
    audio_context_max_gap_s = 2.0
    retry_max_audio_s = 12.0
    use_text_context = false
    text_context_lines = 2
    text_context_chars = 120
    text_context_max_gap_s = 5.0
    hotwords = []
    hololive_names_enabled = true
    hololive_name_limit = 6
    concise_english = true
    style_prompt = "Short, plain English subtitles. No added commentary."
    remove_fillers = true
    collapse_repetitions = true
    diagnostics = false

Audio overlap is used only on suspicious or generic-outro retries, not every
subtitle. The engine never shifts utterance timestamps, and rejects a retry
that cannot remove prior overlapping dialogue confidently. It is not a full
SimulStreaming local-agreement implementation. Most clean lines are decoded
once with beam size 6; suspicious lines can require a second inference.

English context is deliberately off by default to avoid self-reinforcing
hallucinations. Try it experimentally using a local config.toml:

    [whisper]
    use_text_context = true
    hotwords = ["Okayu", "Botan"]

Only put a name in hotwords when it is clearly relevant to that specific
stream. A list of names can bias Whisper toward utterances nobody said.

## Compare audio preprocessing using one command per run

From C:\Games\video-translate-whisper-v3:

    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal --audio-profile baseline --diagnostics
    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal --audio-profile gtcrn --diagnostics
    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal --audio-profile speaker --diagnostics
    uv run --no-sync python -m auto_translate "<URL>" --mode auto --mpv --delay 10 --no-web --simple-terminal --audio-profile both --diagnostics

- baseline: GTCRN OFF, Sherpa OFF
- gtcrn: GTCRN ON, Sherpa OFF
- speaker: GTCRN OFF, Sherpa ON
- both: GTCRN ON, Sherpa ON (launcher default)

Use the same five-minute video section, one run at a time. Rename or
copy logs/run.log between runs. The diagnostics include source timestamps,
utterance length, RMS and peak amplitude, VAD speech confidence, GTCRN and
Sherpa state, speaker-change/split metadata, Whisper confidence/compression,
retry reason and processing time.

Windows launcher uses the last selected profile from config.toml defaults
(both enabled) unless overridden from the command line. No change to MPV
seeking or English subtitle stacking.

## Verification boundaries

Tests in tests/test_context_retry.py and the existing suite mock inference
and do not measure the RTX 3070's true throughput. A second inference and
beam size 6 can increase latency; use logs/run.log to compare before/after.
Model confidence and exact string overlap are heuristics, not proofs that
an English translation is correct.


## Hololive names and concise-English subtitles

The built-in official Hololive roster includes 81 talents, with Japanese
spellings, English names, alumni, and newer generations. Reference:
https://hololive.hololivepro.com/en/talents

On YouTube VOD and live links, the launcher already retrieves a title and
channel through yt-dlp. The project selects up to six relevant Hololive names
using that metadata and sends only those names as optional Whisper hotwords.
It does NOT force the entire roster into every model call; on an unrelated
video, the automatic list is empty. No YouTube captions are downloaded.

For a collaboration where names are absent from the video's title, use a
local config.toml file to supply only the names you expect to hear:

    [whisper]
    hotwords = ["Watame", "Okayu", "Botan", "Towa"]
    hololive_names_enabled = true
    hololive_name_limit = 6

User-supplied hotwords have priority over auto-selected names. Check selected
automatic hints in logs/run.log:

    findstr /i /c:"Hololive name hints:" logs\run.log

Whisper is not an instruction-following LLM. The concise style prompt is only
a weak text-decoder hint and may be ignored or occasionally echoed. The
separate deterministic text filter removes leading "um", "uh", "well",
annotation-only laughter, and 3+ repeated small interjections. It preserves
negations, profanity, names, and normal meaningful words. It never
summarizes, embellishes, or uses a second model. It cannot reliably shorten
a long but genuinely spoken sentence.

Switch both features off for comparison, without changing your audio model:

    [whisper]
    hololive_names_enabled = false
    concise_english = false

To disable only text cleanup but still provide the style hint:

    [whisper]
    concise_english = true
    remove_fillers = false
    collapse_repetitions = false

A common misrecognition such as "Watanabe" for "Watame" is not blindly
rewritten; Watanabe can be a real surname. The glossary biases the decoder
only if there is evidence the name is relevant to the video.

Test without GPU:

    uv run --no-sync python -m unittest discover -s tests -p "test_hololive_style.py" -v

Or run the complete test suite:

    uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -q
