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
- PLaMo uses the Hololive/game glossary, now expanded with てぇてぇ/てーてー,
  wireless versus wired mouse, 玉突き事故, Ferris wheels and other GTA phrases.
  Evidence-gated translation hints cover suspected homophones ONLY where the
  current or recent speech supports them (including item-exchange confusion).
  They never rewrite the Japanese ASR transcript. Explicit Japanese numbers
  can be corrected after translation if PLaMo changes their value, without
  guessing what Whisper heard.
- Both history and nearby-fragment context reset on a seek and on generation
  changes. Speaker handoffs reset near-fragment look-behind.
- Cached VOD subtitles with both JA and EN can be reused without another
  Whisper/PLaMo call. English subtitles stay aligned with original audio
  timestamps; no artificial hard offset is added.
- Suspicious ASR clips can be retried on the SAME current audio segment
  with a different temperature. No past English or expanded audio window is
  mixed into the transcript. Whisper large-v3 remains CTranslate2 (beam 5);
  the separate Anime Whisper Transformers sequence-scoring experiment is
  NOT imported here. This avoids the earlier CUDA beam-index crash.
- The JP terminal transcript is kept verbatim from Whisper: older short-term
  word-memory repair no longer silently changes the recognized Japanese.
- Isolated laughter-only cues are skipped by default. Speech mixed with
  laughter is retained. A second identical stock JP outro within 120 seconds
  is skipped; its first occurrence is preserved. Filters reset on seek.
  To disable the display filters, add this valid TOML to config.toml:

  ```toml
  [whisper]
  suppress_standalone_laughter = false
  suppress_repeated_japanese_outros = false
  ```

## Whisper silence-hallucination protection

FireRedVAD already gates audio **before** Whisper, with a 500 ms
minimum silence split and 150 ms padding. VAD-negative noise probes are
disabled, and Whisper receives independent short utterances: adding
faster-whisper's built-in Silero `vad_filter=True` on top would double-gate
speech and risk removing quiet Japanese responses. The Whisper decoder now
**always** sets `condition_on_previous_text=False`, even if an older
local `config.toml` requested otherwise.

Whisper's `no_speech_prob` is **not** a standalone silence detector. The
first pass's no-speech evidence is preserved even when every segment was
rejected by the existing confidence checks. A fully rejected utterance
with `no_speech_prob >= 0.85` does not get a warmer retry, which otherwise
could invent words on silent/game-audio sections. Ambiguous quiet speech
with lower no-speech scores **remains retryable**. To preserve more borderline
soft voices, raise this threshold (do not lower the VAD threshold and all
decoder thresholds at once):

```toml
[whisper]
no_speech_retry_block_threshold = 0.95
```

This does not suppress all hallucinations, especially confident Japanese
hallucinations after false-positive VAD triggers. Validate on representative
30–60 second clips containing real quiet Japanese, laughter, music, silence,
and speech after the gap. Inspect `diagnostics = true` for Whisper confidence
and VAD timing before changing defaults. Do not use phrase blocklists:
a genuine spoken thanks should not be discarded solely for its wording.

## Seven-hour VOD playback optimization

The former MPV config allowed up to 1 GiB forward and 1 GiB backward demux
cache with 120s readahead. This branch now uses **256 MiB forward**, **64 MiB
backward**, and **30s readahead**; mpv's native seeking and original VOD
timestamps remain unchanged. MPV may use additional memory for decoders.

**Important performance fix:** the original MPV subtitle renderer scanned
`_lines` on every 100ms display tick even for VODs, despite an existing
per-second indexed `SessionSubtitleCache`. With unpaced ingest, many hours
of subtitles could accumulate ahead of playback, so rendering became
progressively more expensive. Cached VODs now render only the cues indexed
at the current time. Live subtitles still use the original rolling list.

Long-running sync diagnostics are limited to the last 4,096 displayed cues;
first-visible diagnostic keys are capped at 8,192. The VOD subtitle cache
itself stays available across the entire video for fast translation reuse
and backwards seeking. These changes target long-session CPU/RAM growth but
are not yet verified on your 7-hour recording.

### Manual pause crash fix

A November 10 long-VOD run paused at about 5:48:06 with subtitles
approximately 74 seconds ahead. The audio read-ahead reader correctly
yielded at its 75-second cap, but the pipeline's 60-second "stalled with
queued work" watchdog then terminated the process with one queued speaker
item. This was a false-positive pipeline crash, not an MPV crash.

The watchdog now checks cached player manual-pause status and the active
audio read-ahead wait state. It suspends only its inactivity timeout while
the VOD is deliberately idle, resetting its timer during the pause and
restoring normal checks after playback resumes. Worker exceptions remain
fatal and MPV itself is still monitored. Seek resets invalidate the waiting
state so a stale generation cannot suppress real failures. Live-stream
watchdog behavior is unchanged.

The player polls manual-pause state once a second on its existing MPV
scheduler thread; the pipeline watchdog never makes extra MPV IPC calls.

### Further mitigation for occasional hitches and slower subtitles

Unpaced FFmpeg previously decoded and queued as much of the entire VOD as
the worker pipeline would consume. On very long videos this could keep
Whisper large-v3, FireRedVAD, PLaMo, FFmpeg and MPV competing for system
resources even while the translated subtitles were already far ahead.
When MPV is active for a VOD, FFmpeg now yields once its audio has advanced
75 seconds beyond the measured MPV playback clock and resumes as the video
catches up. Five-second initial translation buffering and native seeks remain
unchanged; every seek resets the audio reader's source timestamp. Live streams,
standalone transcription, and fMP4 fallback are not throttled.

Tune in `config.toml` (only when needed):

```toml
[ingest]
vod_max_audio_lead_s = 75.0  # 0 disables playback-relative pacing
```

Because `--cache=yes` can override the readahead setting with an effectively
unbounded default `cache-secs`, this branch explicitly sets `--cache-secs=30`
and `--demuxer-hysteresis-secs=10` for predictable network buffering.

The application logs a `VOD playback health` line every 30 seconds:
`translation_lead` (translated timeline minus current video time),
`dropped_frames`, `paused_for_cache`, and `cache_duration`.
A negative translation lead identifies the ASR/MT pipeline falling behind;
growing dropped frames with no cache pause points toward video decoding/
resource contention; `paused_for_cache=True` suggests network buffering.
These metrics do not by themselves prove a root cause, but let the next
long-VOD trace distinguish where the hitch originates.

The launcher keeps software decode (`--hwdec no`) to reserve GPU VRAM.
If dropped frames remain high while VRAM has headroom, test `--hwdec auto-safe`
separately rather than enabling it blindly on an 8 GB RTX 3070.

If playback is still choppy, check whether MPV reports dropped frames,
`paused-for-cache` or cache underruns before changing ASR or subtitles;
the launcher's `--hwdec no` also means video decoding is done on CPU.

## Tests

    uv run --no-sync python -m unittest discover -s tests -p "test_vod_pacing.py" -v
    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_long_vod.py" -v
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
