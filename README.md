# Whisper Large-v3 Direct Japanese Audio-to-English Translation (AST)

This is a **single-model** offline translation pipeline. Only CTranslate2/faster-whisper multilingual Whisper Large-v3 generates the English subtitles. No separate Japanese ASR stage, external LLM, model-server, or GGUF runtime.

## Pipeline

YouTube live/VOD or local media → FFmpeg 16kHz mono → FireRedVAD (CUDA) → **Whisper Large-v3 (Japanese audio → English)** → MPV English-only subtitles/terminal.

Whisper uses language=ja, task=translate, condition_on_previous_text=False and vad_filter=False. Old config.toml values requesting transcription are ignored. Previous [translate] settings are ignored. Large-v3 Turbo is not recommended because its model is not trained for speech translation.

Whisper defaults to int8_float16 to reserve GPU memory for MPV's hardware video decoder on the RTX 3070; FP16 remains available as an opt-in quality experiment with an int8_float16 fallback on model-load VRAM exhaustion. Text-only decoding remains enabled because FireRedVAD already provides timestamps. Wholly rejected low-confidence utterances can only be recovered when two independent decodes agree exactly and both the VAD and Whisper scores strongly indicate real speech. This is a guarded improvement, not a guarantee of translation correctness.

Direct AST cannot also supply a verbatim Japanese transcript without a second model pass. The command prompt therefore shows EN: lines only.

## Update and start (Windows)

    cd /d C:\Games\video-translate-whisper
    git fetch origin && git switch -c feature/whisper-rtx-vsr-480p --track origin/feature/whisper-rtx-vsr-480p && git pull --ff-only
    uv sync
    "Start translate.bat"

The launcher passes original 16 kHz mono PCM to FireRedVAD and Whisper. It uses MPV safe hardware decoding, a 10-second playback delay and CUDA int8_float16. Obsolete speaker and enhancement config sections are ignored.

## Gender-neutral and no-assumed-owner subtitles (direct Whisper AST)

Whisper already produces English without a Japanese transcript. A blanket `their → your` or `he → they` replacement would invent facts or break English grammar, so this feature does **not** port the Haas/PLaMo prompt or pretend an LLM is available.

With `[whisper] neutral_subtitles = true` (default), the pipeline detects English claims about roles/ownership, gendered pronouns and generic "guys/dudes", plus certain isolated first-person reactions. For these cues only, it can run **one unprompted Japanese transcription pass** on the same utterance using the existing Whisper model, up to `neutral_verify_max_audio_s = 14.0` seconds. This covers complete VOD clauses instead of only very short sentences, but increases latency when triggered.

- Japanese `最初のお客さん` with English `Their first customer!` becomes **`First customer!`**.
- For a complete sentence, `Their first customer arrived.` becomes **`The first customer arrived.`**.
- Japanese explicitly stating `お前の`, `俺の`, `彼女の` or a named person's ownership preserves the corresponding English possessor. Never replace `their` with `your` without evidence.
- Sentences like `But the police killed him!` may become `But the police killed someone!`; `Those guys` becomes `Those people` where Japanese provides no gender or named owner. Explicitly gendered source words and named owners remain untouched. Ambiguous contractions and unrecognized object constructions are left alone rather than breaking grammar.
- `分からない` / `見つけた` / `終わった` may lead to `Not sure.` / `Found it.` / `Finished.` only when Japanese evidence supports the concise wording. Explicit or strongly implied first person is retained.
- Missing or failed Japanese verification leaves English **unchanged**. The Japanese pass is still Whisper (not independent proof), and latency increases only on the selected cues.

This does not load PLaMo or a second model, change FireRedVAD, touch the anti-hallucination guards, alter MPV/seek sync, or disable optional RTX VSR. `neutral_subtitles = false` in your local `config.toml` disables it.

Focused offline test: `uv run --no-sync python -m unittest discover -s tests -p "test_whisper_neutral_subtitles.py" -v`.

## Background-only quiet speech recovery (VOD, experimental)

The previous Whisper pipeline always discarded VAD-negative fallback probes, so changing the VAD threshold could not recover low-level speech *between* foreground turns. VOD now gets full 4-second low-probability windows only where FireRedVAD reports a sustained probability ≥0.20 for 150ms (below the main 0.43 speech gate). Rejected silence/gunshot spikes do not reach Whisper. Live stays on the original no-probe profile.

For each candidate window, already-loaded Whisper performs a Japanese transcription, then **two unprompted English AST passes**. Only when Japanese contains a speech-like phrase, the two English passes agree exactly (ignoring punctuation), and both meet strict confidence/silence/loop/phantom checks does the subtitle appear. This avoids publishing unverified music/effects speech and does not add models. The background cue retains its original timestamps and never enters later Whisper prompt history. Each candidate can cost three inferences; VOD playback or seek latency may increase.

This is an attempt to recover **quiet speech outside other speakers' detected speech spans**, not true source separation. Speech occurring *simultaneously* under a louder speaker remains mixed and Whisper can still omit it. Full overlap handling requires a separately benchmarked multi-speaker separation or source-audio-track solution; simply adding speaker labels would not recover unheard words.

Disable with `[vad] background_probe_s = 0.0` (removes probe generation) or `[whisper] background_probe_enabled = false` (skips probe inference). Noise probing remains filtered and threshold-controlled. Focused test: `uv run --no-sync python -m unittest discover -s tests -p "test_whisper_background_probe.py" -v`.

## YouTube quality selector

`Start translate.bat` now displays a numbered video-quality menu after the URL: **1 = 360p, 2 = 480p, 3 = 720p, 4 = 1080p**. Enter **3**, or press Enter to use the default **720p**. The choice is applied before MPV opens the video, and yt-dlp chooses the best available stream **not exceeding** that height (for example, 480p if 720p is unavailable). A mid-playback quality change is intentionally not offered because reloading the source may disturb native seeking and subtitle timing.

You can also run the CLI with `--video-quality 720`. The launcher choice takes precedence over custom video format selectors in `config.toml`; when the CLI flag is absent, the existing selectors remain unchanged. Finished VODs still prefer HTTPS/DASH and fall back to HLS; live streams still prefer HLS. Whisper audio inference, VAD, and the independent audio ingest format are unaffected.

## Accuracy-oriented Japanese speech recall (experimental)

This branch preserves the direct Whisper Large-v3 JA-audio → English AST model and anti-hallucination checks. The changes favor missed-speech recall and longer complete VOD sentences over minimum throughput, without adding a translator, audio enhancement, diarization or another model.

**FireRedVAD default changes:** threshold 0.50→0.43, negative threshold 0.30→0.28, minimum speech 250→180 ms, onset 80→70 ms, padding 150→300 ms, quiet-point minimum 80→100 ms, split context 120→160 ms. The high-confidence 120 ms reply exception remains. Silence remains 500 ms. **VOD-only low-VAD probes are now narrowly screened** (sustained ≥0.20 scores, 150ms; JA verification plus two agreeing high-confidence EN passes). Live/background-overlap remains unchanged. Lower VAD thresholds can nevertheless admit additional game audio; compare against the earlier branch before treating it as better.

**Context by playback mode:** VOD uses 12 s soft / 14 s hard utterance caps so Japanese clauses have more context, while live retains 7 s soft / 8 s hard caps to avoid delayed captions. Both retain timestamp-contiguous forced splitting, native MPV seeking and progress handoffs. The effective profile is logged as `Whisper speech-recall VAD`. Local `config.toml` overrides are preserved, including `live_max_utterance_s` and `live_max_split_extension_s`.

**Decoder:** beam 5→6 (beam search at temperature 0), output token allowance 160→192, and uncertain-utterance retry limit 12→15 s (only where existing strict retry conditions warrant). GPU compute precision remains `int8_float16` and previous text conditioning stays OFF; no broad temperature fallback or speculative previous-caption prompt is enabled.

**Evidence-gated empty-window rescue:** If the *initial* Whisper pass returns no segments (not a rejected segment), FireRedVAD confidence ≥0.88, original decoder no-speech probability is either unavailable (normal for zero segments) or ≤0.45, and the utterance lasts 0.45–8 seconds, one targeted JA transcription on the same audio checks for recognized Japanese speech. If it returns plausible Japanese text, a final unprompted English AST attempt at the existing 0.2 retry temperature must pass strict logprob, no-speech, loop, compression, pacing and stock-phantom guards. Total budget at most three model calls for this rescue (AST→JA→AST). When no evidence or low confidence, publish **nothing** rather than invent speech. Japanese and English come from the *same* model, so correlated hallucinations remain possible.

Tunable `[whisper]` keys: `recover_empty_speech = true`, `recover_min_vad_confidence = 0.88`, `recover_max_no_speech_prob = 0.45`, `recover_max_audio_s = 8.0`. Set `recover_empty_speech = false` to compare recall without the rescue. Extra inference is limited to selected silent/neutrality/phantom checks; live delay and VOD pacing may still be affected by the increased VAD context. Do not treat this as an observed CER/WER improvement until annotated HoloGTA clips are tested.

Focused regression: `uv run --no-sync python -m unittest discover -s tests -p "test_whisper_accuracy_recall.py" -v` or run the entire suite.

## Key defaults and silence protections

    [whisper]
    model = "large-v3"
    compute_type = "int8_float16"
    beam_size = 6
    max_new_tokens = 192
    recover_empty_speech = true
    temperature = 0.0
    no_speech_retry_block_threshold = 0.85
    verify_stock_outros = true
    filter_phantom_phrases = true
    rescue_stable_weak_speech = true
    remove_fillers = true
    collapse_repetitions = true

    [vad]
    threshold = 0.43
    min_speech_ms = 180
    min_silence_ms = 500
    speech_pad_ms = 300
    fallback_s = 0.0

    [ingest]
    vod_max_audio_lead_s = 75.0

The segmenter retains pre-onset audio in addition to the onset confirmation window. The usual 250 ms speech minimum also accepts short runs with at least 120 ms of consecutive smoothed speech probability at or above 0.85. This can recover brief replies and can admit more speech-like game audio.

Whisper uses scalar temperature 0, beam 6, no-speech 0.6, log probability -1.0 and compression ratio 2.4. The verifier may decode the same PCM once at temperature 0.2 without prompts or names, up to 15 seconds per uncertainty retry. It checks unusually long output, multiword loops and combinations of weak VAD and decoder evidence. A higher score alone cannot replace an existing translation with different words. Filler and nonverbal-only laughter cleanup is enabled by default even with the optional concise style prompt disabled; meaningful repetition and genuine signoffs remain eligible. To restore unfiltered text, set both `[whisper].remove_fillers = false` and `[whisper].collapse_repetitions = false` in local config.toml.

## Targeted generic-outro verification

When Whisper outputs a complete generic English signoff such as "Thank you for watching", the pipeline makes one additional unprompted Japanese transcription pass **on the same speech segment using the already-loaded Whisper Large-v3**. It only publishes the signoff when the Japanese text corroborates both watching and thanks. The same full-cue guard now covers stock subtitle-credit phrases (for example, `Subtitles by Amara.org`) and channel subscription requests (for example, `Please subscribe to my channel`). Patterns match complete cues, not normal dialogue mentioning these words. Wholly implausible speech duration and combined low-VAD/high-no-speech evidence can suppress a matched cue; strong acoustic evidence and Japanese corroboration preserve likely real speech. This duration heuristic uses the VAD utterance, **not** Whisper word timestamps, so it does not assume English and Japanese have identical word cadence. Ordinary thanks, gameplay dialogue, and all other English AST output follow the existing path. No second model is downloaded, and MPV/seek synchronization and the English-only overlay are unchanged.

This is an anti-hallucination heuristic, **not independent acoustic proof**: both tasks use the same Whisper model, so correlated errors can still occur. It may also suppress a genuine signoff when the Japanese check fails to recognize it. Set `[whisper].verify_stock_outros = false` to disable signoff verification, or `[whisper].filter_phantom_phrases = false` to disable the separate subtitle-credit/subscription guard. Filtered cues are recorded as `Suppressed Whisper phantom phrase` or `Suppressed uncorroborated Whisper phantom phrase` in `logs/run.log`.

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

## Optional RTX VSR 480p playback experiment

In `Start translate.bat`, select **2 (480p)**, then answer **Y** to enable RTX Video Super Resolution. The default is **N** (off). Other numbered quality choices keep ordinary MPV playback. CLI users can opt in with `--video-quality 480 --rtx-vsr --mpv --hwdec auto-safe`, or set `[mpv] rtx_vsr = true` in a local config. The runtime still refuses to apply RTX VSR unless the decoded source is at most 480p.

**Set quality yourself:** In **NVIDIA App > System > Video**, enable **Video Super Resolution** and choose **Manual / High (highest available quality)**. The MPV `d3d11vpp` filter cannot set NVIDIA's quality tier. When enabled on Windows, the player requests Direct3D 11 with `--vo=gpu-next --gpu-api=d3d11 --gpu-context=d3d11`, preserving the existing `--hwdec=auto-safe`. After `file-loaded`, it adds the labeled `d3d11vpp=scale=3:scaling-mode=nvidia` filter (480p to approximately 1440p). It does not change MPV seeking, English subtitles, FireRedVAD, Whisper decoding or audio ingestion.

**Safety/fallback:** RTX VSR is disabled by default and skipped on non-Windows systems, with software-only MPV decoding, unknown video height or videos above 480p. The player removes only its own labeled filter if MPV rejects it, it disappears, the decoder switches to software or the 30-second frame-delivery checks detect at least 15 dropped frames or 30 delayed frames. It ignores manual and cache pauses, and preserves the current video session rather than restarting MPV. When disabled, the D3D11 renderer remains but normal non-AI scaling resumes; the next launch with VSR off uses the original MPV options.

**Verify on your machine:** MPV accepting a filter **does not prove NVIDIA actually ran AI enhancement**. Check NVIDIA App's RTX Video active indicator and `logs/mpv.log`. Compare the same 480p 30/60 fps VOD for several minutes with VSR off and on, then try live content. Inspect `logs/run.log` entries `RTX VSR playback check`, `RTX VSR disabled`, and `VOD playback health`, specifically `hwdec_active`, drop counts, VO delay, GPU memory and translation lead. Prefer turning VSR off if the RTX 3070 struggles with simultaneous Whisper inference.

CPU-only checks (mocked counts and capability checks):

    uv run --no-sync python -m unittest discover -s tests -p "test_rtx_vsr.py" -v

No local RTX/Windows performance or NVIDIA quality setting can be validated by these unit tests.
