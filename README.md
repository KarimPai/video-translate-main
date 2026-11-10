# Whisper Large-v3 Direct Japanese Audio-to-English Translation (AST)

This is a **single-model** offline translation pipeline. Only CTranslate2/faster-whisper multilingual Whisper Large-v3 generates the English subtitles. No separate Japanese ASR stage, external LLM, model-server, or GGUF runtime.

## Pipeline

YouTube live/VOD or local media → FFmpeg 16kHz mono → FireRedVAD (CUDA) → **Whisper Large-v3 (Japanese audio → English)** → MPV English-only subtitles/terminal.

Whisper uses language=ja, task=translate, condition_on_previous_text=False and vad_filter=False. Old config.toml values requesting transcription are ignored. Previous [translate] settings are ignored. Large-v3 Turbo is not recommended because its model is not trained for speech translation.

Whisper defaults to int8_float16 to reserve GPU memory for MPV's hardware video decoder on the RTX 3070; FP16 remains available as an opt-in quality experiment with an int8_float16 fallback on model-load VRAM exhaustion. Text-only decoding remains enabled because FireRedVAD already provides timestamps. Wholly rejected low-confidence utterances can only be recovered when two independent decodes agree exactly and both the VAD and Whisper scores strongly indicate real speech. This is a guarded improvement, not a guarantee of translation correctness.

Direct AST cannot also supply a verbatim Japanese transcript without a second model pass. The command prompt therefore shows EN: lines only.

## Update and start (Windows)

    cd /d C:\Games\video-translate-whisper
    git fetch origin && git switch feature/whisper-anti-hallucination && git pull --ff-only origin feature/whisper-anti-hallucination
    uv sync
    "Start translate.bat"

The launcher passes original 16 kHz mono PCM to FireRedVAD and Whisper. It uses MPV safe hardware decoding, a 10-second playback delay and CUDA int8_float16. Obsolete speaker and enhancement config sections are ignored.

## YouTube quality selector

`Start translate.bat` now displays a numbered video-quality menu after the URL: **1 = 360p, 2 = 480p, 3 = 720p, 4 = 1080p**. Enter **3**, or press Enter to use the default **720p**. The choice is applied before MPV opens the video, and yt-dlp chooses the best available stream **not exceeding** that height (for example, 480p if 720p is unavailable). A mid-playback quality change is intentionally not offered because reloading the source may disturb native seeking and subtitle timing.

You can also run the CLI with `--video-quality 720`. The launcher choice takes precedence over custom video format selectors in `config.toml`; when the CLI flag is absent, the existing selectors remain unchanged. Finished VODs still prefer HTTPS/DASH and fall back to HLS; live streams still prefer HLS. Whisper audio inference, VAD, and the independent audio ingest format are unaffected.

## Key defaults and silence protections

    [whisper]
    model = "large-v3"
    compute_type = "int8_float16"
    beam_size = 5
    temperature = 0.0
    no_speech_retry_block_threshold = 0.85
    verify_stock_outros = true
    rescue_stable_weak_speech = true
    remove_fillers = true
    collapse_repetitions = true

    [vad]
    threshold = 0.50
    min_speech_ms = 250
    min_silence_ms = 500
    speech_pad_ms = 150
    fallback_s = 0.0

    [ingest]
    vod_max_audio_lead_s = 75.0

The segmenter retains pre-onset audio in addition to the onset confirmation window. The usual 250 ms speech minimum also accepts short runs with at least 120 ms of consecutive smoothed speech probability at or above 0.85. This can recover brief replies and can admit more speech-like game audio.

Whisper uses scalar temperature 0, beam 5, no-speech 0.6, log probability -1.0 and compression ratio 2.4. The verifier may decode the same PCM once at temperature 0.2 without prompts or names, up to 12 seconds per retry. It checks unusually long output, multiword loops and combinations of weak VAD and decoder evidence. A higher score alone cannot replace an existing translation with different words. Filler and nonverbal-only laughter cleanup is enabled by default even with the optional concise style prompt disabled; meaningful repetition and genuine signoffs remain eligible. To restore unfiltered text, set both `[whisper].remove_fillers = false` and `[whisper].collapse_repetitions = false` in local config.toml.

## Targeted generic-outro verification

When Whisper outputs a complete generic English signoff such as "Thank you for watching", the pipeline makes one additional unprompted Japanese transcription pass **on the same speech segment using the already-loaded Whisper Large-v3**. It only publishes the signoff when the Japanese text corroborates both watching and thanks. Ordinary thanks, gameplay dialogue, and all other English AST output follow the existing path. No second model is downloaded, and MPV/seek synchronization and the English-only overlay are unchanged.

This is an anti-hallucination heuristic, **not independent acoustic proof**: both tasks use the same Whisper model, so correlated errors can still occur. It may also suppress a genuine signoff when the Japanese check fails to recognize it. Set `[whisper].verify_stock_outros = false` in a local `config.toml` to disable this targeted check. Filtered outputs are recorded as `Suppressed uncorroborated Whisper stock outro` in `logs/run.log`.

Run the focused mock-model regression:

    uv run --no-sync python -m unittest discover -s tests -p "test_whisper_outro_guard.py" -v


Run the commands below from the feature checkout. The synthetic replay uses a fixed 38-second PCM fixture and scripted FireRed probability inputs. It exercises actual segmentation logic without speech recognition. The matrix includes the checkout default plus every combination of 80/300 ms padding, 400/500 ms silence and 12/16/20 s hard caps. Each soft cap is one second below its hard cap.

    uv run --no-sync python tools/benchmark_whisper.py --synthetic --output synthetic.jsonl

For a GPU comparison, create a JSON array in `clips.json` with one entry per local 16 kHz mono 16-bit WAV. Use separate clips of game audio/music without speech and clips containing one annotated utterance each. Include longer exchanges to inspect segmentation and cue timing. Example entries:

    [
      {"wav":"effects.wav","kind":"non-speech"},
      {"wav":"reply.wav","kind":"utterance","ja":"行かない","en":"I'm not going."},
      {"wav":"conversation.wav","names":["Watame"]}
    ]

Create the baseline worktree once and run the same script and manifest in each environment. Use the same model files, package versions and GPU load. The baseline command uses the original launcher's speaker splitter. Model loading and the optional Japanese diagnostic are outside the reported processing interval.

    git worktree add --detach ..\whisper-before 05665ed85e96a6016f239dff6dd587c13611bf6e
    uv sync --project ..\whisper-before --extra diarization
    uv run --project ..\whisper-before --no-sync python tools/benchmark_whisper.py --checkout ..\whisper-before --baseline --manifest clips.json --japanese --output before.jsonl
    uv run --no-sync python tools/benchmark_whisper.py --manifest clips.json --japanese --output after.jsonl

Compare records by PCM hash. Each includes segment boundaries, English cues, processing seconds and device-wide VRAM sampled after decoding. The per-cue latency estimate includes audio collected after the cue endpoint and decoder time. MPV display timing, pipeline backlog and peak VRAM require a playback run. Japanese CER uses the concatenated diagnostic transcript and a human reference. Single-utterance omission counts detect empty output only. Review nonempty translations against the audio and English references for wrong negation, names, omissions and additions.

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
