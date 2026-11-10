# VoicePing Qwen3-ASR 1.7B — Direct Japanese-to-English AST (Experimental)

**Runtime pipeline (single translation model):**
YouTube live/VOD or local media → FFmpeg original 16 kHz mono PCM → FireRedVAD (CUDA) → **[VoicePing Qwen3-ASR JA–EN](https://huggingface.co/voiceping-ai/qwen3-asr-ja-en-speech-translation)** → English-only MPV/terminal subtitles.

VoicePing is fine-tuned for speech translation. It produces English text directly from Japanese speech using `language="English"` (the **target**, not source, language). No separate Japanese transcription, PLaMo, Whisper inference, forced aligner, speech enhancement, speaker diarization, or quantization is used in the active pipeline.

## First-time setup on Windows (new isolated checkout recommended)

```bat
cd /d C:\Games && git clone -b feature/voiceping-qwen3-ast --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-voiceping && cd /d C:\Games\video-translate-voiceping && uv sync && "Start translate.bat"
```

Or switch an existing clean checkout of this repository:

```bat
cd /d C:\Games\video-translate-whisper && git fetch origin && git switch --track origin/feature/voiceping-qwen3-ast && uv sync && "Start translate.bat"
```

The model downloads from Hugging Face on first load; budget approximately **4.1 GB** for model files (additional package download/cache space required). Model GPU memory and end-to-end latency have **not** been measured on RTX 3070 8GB. FP32 is not recommended.

## RTX 3070 inference baseline

```toml
[ast]
model = "voiceping-ai/qwen3-asr-ja-en-speech-translation"
device = "cuda:0"
dtype = "bfloat16"
max_inference_batch_size = 1
max_new_tokens = 192
context_enabled = false  # recommended: no prompt echo or name bias
japanese_source_only = false  # legacy key, not a reliable language gate
concise_context = false  # legacy key, instructions are not sent to Qwen
hololive_names_enabled = true
hololive_name_limit = 6
verify_standalone_names = true
name_hints = []  # e.g. ["Watame", "Jack"] for this specific stream
repeat_short_cue_limit = 2
remove_fillers = true
collapse_repetitions = true
diagnostics = false

[vad]
threshold = 0.35
neg_threshold = 0.30
min_speech_ms = 100
min_silence_ms = 400
speech_pad_ms = 80
max_utterance_s = 11.0
max_split_extension_s = 1.0
fallback_s = 0.0
```

These are the **branch defaults** if no `config.toml` is supplied. Existing custom `[vad]` values in a local untracked config override the defaults. The soft speech split is at 11 seconds and the hard deadline is 12 seconds; the model supports up to 30 seconds.

The initial test uses **BF16, batch size 1, default decoder, 192 maximum new tokens, and no quantization or forced aligner**. `qwen-asr` uses Transformers/PyTorch CUDA. Windows native PyTorch is used instead of Linux-only vLLM. Only one model is loaded in GPU memory for translation.

## Critical fix: prompt echo and repeated translations

VoicePing printed its native context instructions as an English subtitle. The
prompt included wording that was not spoken. Natural-language instructions
are therefore disabled for this ASR translation fine-tune. The project sends
**no instructions** to `transcribe(context=...)` at any setting.
`language="English"` still selects Japanese-audio → English-text.

- Default: `context_enabled = false` (no application-generated prompt).
- Optional name bias: `[ast] context_enabled = true` sends only comma-separated
  names selected from stream metadata or manual `name_hints`, such as
  `Watame, Okayu`. This is disabled by default because names were being
  hallucinated over laughter.
- `japanese_source_only` and `concise_context` are accepted for legacy config
  compatibility but no longer inject instructions. Strict Japanese-language
  gating would require tested acoustic language identification.
- A new generation-loop guard detects long phrases repeated three times,
  including multi-sentence cycles. It drops the entire suspect output rather
  than printing potentially fabricated text. Short "No, no, no!" remains.
- English filler and adjacent-cue suppression remain enabled.

Optional vocabulary bias:

```toml
[ast]
context_enabled = true
name_hints = ["Watame", "Jack"]
```

A standalone name promoted by the optional context is still verified with an
unprompted second inference on the same audio. This is heuristic, not proof
of accurate transcription, and has an additional latency cost.
## Standalone-name hallucination guard

If laughter or game effects are misrecognized as a single Hololive name such
as `Watame.`, the pipeline now runs **one verification decode of the same
audio without text context**. It only publishes the standalone name if both
decodes produce exactly the same words (ignoring punctuation/case). Names
inside meaningful sentences are not re-decoded. The guard uses the same
VoicePing model, not a second ASR or translator. It can also suppress a true
name when unprompted recognition fails. **It is a conservative heuristic, not
a speech-versus-laughter detector**, and may miss hallucinations where the two
decodes agree. The default is `verify_standalone_names = true`; set false
in `[ast]` to turn it off.

If you still see names replacing laughter, compare your original VOD with
`[ast] hololive_names_enabled = false` (automatic title/channel hints off)
and optionally `context_enabled = false` (all Qwen context off). This helps
determine whether the model's intrinsic recognition or context bias is the cause.
Never globally delete every occurrence of Watame: some are legitimate speech.

Run `uv run --no-sync python -m unittest discover -s tests -p "test_voiceping_ast.py" -v`
for mock regression coverage. Actual GPU/stream testing is still required.

## Retained features

- Original MPV playback and click/drag seeking; VOD seek generations discard stale subtitles.
- YouTube quality selector: **1** 360p, **2** 480p, **3** 720p (default), **4** 1080p.
- FFmpeg audio-only ingestion, original 16kHz mono, FireRedVAD CUDA.
- English-only, three-line MPV subtitles and 10-second launcher delay.
- VOD subtitle cache, audio lead limit, manual-pause watchdog, and MPV safe hardware decoding.
- Prints `EN:` translation output in the command window; Japanese source transcript is not generated.

The original Whisper decoder remains in the repository only as *inactive legacy code for its historical tests*. The launcher, CLI and pipeline **never instantiate Whisper or a separate MT model** on this branch. `faster-whisper` is no longer a runtime dependency.

## Conservative output safeguards

FireRedVAD-negative probes, nonfinite and silent PCM are not passed to VoicePing. The pipeline suppresses obvious multiword repetition loops and Japanese-only text, and keeps the existing nonverbal filler cleanup. **Whisper-specific log-probability, no-speech and retry thresholds are not applied**, because VoicePing does not expose comparable metrics in this inference API. These changes do not prove a lower hallucination rate.

## Checks and measurements

```bat
uv run --no-sync python -m unittest discover -s tests -p "test_voiceping_ast.py" -v
```

On your actual GPU, test 10–20 minutes of the same Hololive content in both VoicePing and Whisper branches. Measure: peak VRAM (`nvidia-smi`), time from audio to English subtitle, silent/music hallucinations, missing dialogue, name recognition, and negation errors. Model download, Windows dependency compatibility and live translation speed **cannot be verified by mock tests**.

The official inference docs are at [VoicePing model card](https://huggingface.co/voiceping-ai/qwen3-asr-ja-en-speech-translation) and [Qwen3-ASR repo](https://github.com/QwenLM/Qwen3-ASR).
