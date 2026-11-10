# Haas Japanese ASR + PLaMo 3 Translate 2B — 2026-11-11 Retest

**Experimental branch:** `feature/haas-plamo-retest-20261111`, forked from the latest Haas/PLaMo `main` baseline. This is not the VoicePing/Whisper AST pipeline. `main` is unchanged.

## Active pipeline

`YouTube live/VOD or local source → FFmpeg original 16 kHz mono audio → FireRedVAD (CUDA) → Haas-ja-v1.0 (Vulkan speech.cpp) → PLaMo 3 Translate 2B Q4_K_M (local llama.cpp) → English-only MPV captions`.

- MPV plays the original video with the native timeline and seekbar. Source ingest is audio-only for direct-video playback.
- Terminal prints `JP:` and `EN:` lines; MPV overlays only English, up to three stacked lines.
- Japanese YouTube VOD captions are optional evidence and can assist Haas misrecognition.
- `glossary.toml` and the original title/channel context remain active for proper names and terms. Existing conservative Watame/Watanabe correction is preserved.
- No VoicePing, Whisper, separate LLM beyond PLaMo, Sherpa diarization, forced aligner or new audio classifier.
- Vocal separation and DeepFilterNet are disabled by default; the original speech signal is used.

## First-time Windows setup (separate from main and VoicePing)

Requires Python 3.12+, uv, FFmpeg, yt-dlp, mpv, a working Vulkan `speech` CLI from speech.cpp, Haas-ja-v1.0 model, and a GPU-enabled `llama-server` for PLaMo.

```bat
cd /d C:\Games && git clone -b feature/haas-plamo-retest-20261111 --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-haas-plamo-retest && cd /d C:\Games\video-translate-haas-plamo-retest && uv sync && uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v && "Start translate.bat"
```

Choose video quality **1=360p, 2=480p, 3=720p (default), 4=1080p**. VODs prefer direct HTTPS/DASH sources with lower-quality fallback; live sources prefer HLS.

## New FireRedVAD defaults

```toml
[vad]
threshold = 0.45
neg_threshold = 0.30
min_speech_ms = 200
min_silence_ms = 500
speech_pad_ms = 200
onset_ms = 100
prob_smooth_frames = 5
short_reply_ms = 120
short_reply_probability = 0.85
max_utterance_s = 11.0
max_split_extension_s = 1.0
split_pause_min_ms = 80
split_context_ms = 120
fallback_s = 0.0
```

The VAD uses smoothed probability, sustained speech onset, a confidence-gated short-reply exception, and phrase splitting near natural pauses with a 12-second hard cap. VAD-negative probes are **off** to reduce hallucinations over game music/effects. These experimental thresholds may miss soft-spoken Japanese and delay subtitles by hundreds of milliseconds; compare against the old 0.35/100/400/80ms baseline.

## PLaMo context and safeguards

```toml
[translate]
plamo_hf_model = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
plamo_context = 4096
plamo_gpu_layers = 99
max_tokens = 120
context_lines = 6
memory_lines = 12
glossary = "glossary.toml"
remove_fillers = true
collapse_repetitions = true
suppress_runaway_loops = true
max_identical_cues = 2
```

PLaMo's dedicated translation template still uses the Japanese transcript, relevant glossary matches, video metadata, and short recent JP/EN pairs. Unlike VoicePing ASR, PLaMo is the **translation** model; its existing translation template is retained. Memory has been bounded from 24 to 12 and dialogue from 8 to 6 to reduce prompt feedback and latency.

New post-processing filters obvious filler/laughter-only English, long repeated sentence loops, recognizable template echoes, and the third or later identical nearby multiword subtitle. It **does not paraphrase** the Japanese text or arbitrarily change names/negation. Long-loop rejection discards the suspicious cue entirely rather than guessing which parts were spoken.

## Playback and reliability

- **MPV VOD starts after approximately five seconds of translated audio lead**, and later seeks rebuild this buffer while retaining the original source's seek timeline.
- MPV VOD audio read-ahead is capped at **75 seconds** relative to video position. During intentional manual pause, the reader eventually parks instead of processing a whole VOD in advance. Set `[ingest] vod_max_audio_lead_s = 0` to disable.
- Worker failures are reported to the main runner instead of silently hanging. The queue watchdog excludes intentional VOD pause and audio read-ahead parking.
- Session-only subtitles and processed-region caching are preserved, including reuse after backward seeking.
- MPV safe hardware decode and the existing native OSC seekbar are preserved.
- For live playback, the configured original delay remains **9 seconds** (not a hard subtitle timestamp offset).

## Scope and limitations

This is a **port of compatible VoicePing/Whisper project findings**, not a model swap. Haas has no Whisper-style `no_speech_prob` or comparable decoder metrics, so Whisper retry thresholds were not copied. Qwen-specific natural-language prompts were not copied because they were emitted as subtitles in a prior VoicePing test. The optional language/audio-event detector ideas have not been integrated: FireRedVAD identifies speech but does not verify Japanese source language.

The strict VAD, 12-second segments and memory reductions are **new experiments**, not yet measured for Haas accuracy or latency. Existing local untracked `config.toml` overrides these defaults, so first retest with a clean checkout or inspect local overrides. Live testing and native Windows GPU/Vulkan testing are still required.

Run offline regression tests:

```bat
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
```

Check `logs\run.log` for application errors and `logs\mpv.log` for player warnings. Benchmark with the same VOD/time ranges as the earlier Haas/PLaMo baseline to compare recognition completeness, laughter hallucinations, names, negation, lead after seeks, 95th-percentile subtitle latency, and peak VRAM.
