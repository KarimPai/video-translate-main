# holoGTA local stress-test benchmark

This benchmark uses the same downloaded clip for every ASR/translation run.
It only changes model selection (and optionally context length). The production
pipeline and your local config.toml are not modified.

## Source clip

Original video: https://www.youtube.com/watch?v=vAsOpdXJNCY

Suggested segment: 01:25:00–01:45:00 (20 minutes from the original video).
Download once via yt-dlp; audio-only is sufficient for ASR and translation.

## Run baseline in PowerShell 7

From the existing repository directory:

    cd C:\Games\video-translate-main
    git pull --ff-only origin main

    # Replace with the ACTUAL finished file (.webm or .m4a).
    $clip = "C:\Path\To\hologta-stress-test.webm"
    Test-Path $clip
    uv run python .\scripts\benchmark_video.py "$clip" --asr haas --mt gemma4

Run ASR combinations sequentially, with unchanged translation AND VAD.
Haas applies a special 0.28 onset threshold when config.toml has the 0.24
default, so explicitly use 0.28 for every backend in a controlled comparison.
The 14-second VAD cap is retained for context-rich recognition:

    uv run python .\scripts\benchmark_video.py "$clip" --asr haas qwen3-asr whisper --mt gemma4 --vad-max-utterance-s 14 --vad-threshold 0.28

Only include backends whose models and runtimes are configured. To compare
translation models after selecting an ASR using human-checked Japanese audio:

    uv run python .\scripts\benchmark_video.py "$clip" --asr haas --mt gemma4 plamo

To compare context separately, keep ASR and translator unchanged:

    uv run python .\scripts\benchmark_video.py "$clip" --asr haas --mt gemma4 --context-lines 8

The first Haas/Gemma 4 baseline (126 cues over 20 minutes) produced 41
cues at 14–16 seconds. In a repeat with the same effective 0.28 VAD threshold,
14s produced 125 cues; 10s produced 137 cues. The 10s experiment shortened
translated output but recovered 4.6% less Japanese text. Neither is proven
more accurate without checked audio. Some 10s VAD utterances are merged by
the translator into subtitles lasting up to 16s (translate.merge_max_s).
Test whether shorter VOD segmentation helps with a human reference.
The following is an EXPERIMENT, not a new default. It does not modify config.toml:

    uv run python .\scripts\benchmark_video.py "$clip" --asr haas --mt gemma4 --vad-max-utterance-s 10

Compare it against the unchanged baseline. Do not conclude that shorter
cues are more accurate without listening to matched speech.

The runner uses the existing CLI in local-file VOD mode, with fast processing,
no web viewer, safe playback limits, and no shared diagnostics log. It does
not launch MPV, isolating transcription and translation from player problems.

## Files created

A unique timestamped directory within benchmarks/runs/ contains:

- manifest.json: source, config path, selected models, clip size
- 01-ASR-MT.log: full CLI output for debugging
- 01-ASR-MT.jsonl: structured Japanese/English cues, timestamps, inference times
- summary.json and summary.csv: status, cue counts, 10s/14s+ cues, long
  English text counts, stage times, wall time
- <run>-review.csv: especially long subtitles to check against actual audio

Cue count, Japanese characters, and fluent English are not accuracy scores.
More cues could mean more hallucinations. Fewer cues may mean missed speech.
Log parsing relies on the timestamped terminal output; investigate any
unexpected zero-cue result.

## Score quality manually against a reference

Listen to a representative subset, including overlapping voices, short
replies, music, engines, NPC dialogue and silence. Evaluate:

1. Corrected Japanese transcript versus model transcript (character error rate).
2. Missed spoken Japanese without any subtitle.
3. Hallucinations during non-Japanese or no-speech periods.
4. Names and nicknames (Miko, Mikochi, Lap and other spoken forms).
5. English meaning, fluency and dialogue continuity after verifying Japanese.
6. Subtitle timing, tested separately in MPV or ViiTor playback.

ViiTor's exact cloud ASR and translation models are undisclosed. Compare
the identical audible section of the original YouTube video, not a different
part of the stream.

## Controlled optimization stages

Stage A: keep existing VAD and translator fixed; compare the ASR choices.
Stage B: change only VAD behavior if errors occur near speech boundaries.
A possible controlled experiment is min_silence_ms 400 versus 500; neither
should be declared universally superior without results.
Stage C: with the preferred ASR and VAD, compare translation models.
Stage D: compare translation context length, default versus 8 lines.

Try selective ASR overlap only if the evaluation shows omitted words at
forced VAD boundaries. Global repeated recognition can increase hallucinations,
compute load and GPU instability.

## Limits and safety

The 20-minute test must run on your Windows PC with installed model files.
There is no GPU access to the user PC from this repository.
Run one combination at a time. Keep config.toml private and untracked.
