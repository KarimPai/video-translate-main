# Anime Whisper + PLaMo 3 Translate 2B

Branch: `experiment/anime-whisper-plamo2b` — independent accuracy experiment.
The stable `main`, `whisper-v3`, and other non-Granite branches remain unchanged.

## Pipeline

YouTube live/VOD -> FFmpeg mono 16 kHz -> FireRedVAD (CPU by default)
-> **litagin/anime-whisper** (Japanese ASR via Transformers/PyTorch CUDA)
-> PLaMo 3 Translate 2B Q4_K_M (local llama.cpp, Japanese to English)
-> MPV with English-only subtitles and JP/EN terminal output.

The existing 3-row subtitle stacking, VOD seek cache, 1080p-first playback,
PLaMo glossary, and synchronization architecture come from `whisper-v3`.
Optional GTCRN denoising and Sherpa speaker splitting are OFF in the launcher,
to keep the ASR accuracy comparison controlled.

## Model behavior and accuracy

- Original HF model: https://huggingface.co/litagin/anime-whisper
- Fine-tuned from Kotoba-Whisper v2.0 on anime-style dialogue. Approximately
  3.03 GB checkpoint download on first run (shared Hugging Face cache).
- **Never use an initial prompt.** The model developer explicitly warns that
  any initial prompt severely degrades accuracy and increases hallucinations.
  The AnimeWhisperASR adapter ignores prompt and hotwords arguments even when
  passed by the caller; Japanese is selected with `language="japanese"` and
  `task="transcribe"` as recommended by the author.
- Default decoding: beam size 5, deterministic temperature 0,
  no_repeat_ngram_size 5, repetition_penalty 1.0. Adaptive short-clip
  token budget is enabled (1s ~64 tokens, 3s ~122, 8s capped at 160),
  reducing the chance of generating unsupported extra sentences.
- Accuracy-first retry: the primary decode uses beam 5 and optionally returns
  token transition scores. When scores indicate an uncertain cue, or a stock
  Japanese outro appears, the same audio is decoded again unprimed with beam 8
  at temperature 0. The alternate hypothesis is accepted ONLY if it improves
  the average token score by >=0.18, retains numbers, is Japanese, and has
  sufficient text agreement (or replaces a stock outro). If scores are
  unavailable, the first hypothesis is preserved. Token scores are heuristic,
  not calibrated hallucination probabilities. Extra decoder scores may use
  additional GPU memory (disable with [whisper] collect_token_scores=false).
- The existing FireRedVAD silence gate, Japanese script gate, and local PLaMo
  translator remain active.
- FireRedVAD runs on CPU by default on the Anime Whisper branch. This is the
  same FireRed streaming model and segmentation, but isolated from Whisper's
  CUDA context during large VOD seeks. To force the old GPU mode, set
  `[vad] device = "cuda"` in local config.toml. CPU VAD may be slower.
- FireRedVAD segmentation is tuned for fuller Japanese phrase boundaries:
  threshold 0.50, speech padding 250ms, minimum silence 550ms,
  minimum speech 180ms, 8s soft cap plus up to 1s extension.
- Standalone laughter is hidden by default, while laughter that accompanies
  actual dialogue is preserved. A repeated *exact* video outro is skipped
  only on its second occurrence within 120s. These are presentation filters,
  not proof of hallucination; disable either using config.toml if desired.
- Translation-only contextual hints handle observed ASR confusions:
  遊星/有線 and 無銭/無線 *in mouse context*, 弾付き事故/玉突き事故 with
  traffic evidence, 覡断車/観覧車 within a Ferris-wheel sentence, and
  ハレムーン/ハネムーン when a wedding is explicitly mentioned. No broad
  homophone substitutions, no invented proper names, and no ASR priming.
- The glossary now covers てーてー/てぇてぇ (wholesome affectionate interaction),
  wireless/wired mice, traffic pile-ups, amusement parks and Ferris wheels.
- The Japanese ASR transcript remains unchanged in terminal and subtitle
  output; PLaMo gets suggestions as non-authoritative translation context.
- The model transcribes sighs, laughter, and expressive speech by design;
  this may produce more nonverbal output, which is not inherently hallucination. PLaMo remains the only
  English translation engine and the ASR never generates explanatory narration.

## Windows / RTX 3070 8 GB

Prerequisites: Windows, Python 3.12+, uv, NVIDIA driver, MPV, FFmpeg, yt-dlp,
CUDA-enabled llama-server on PATH. First start also downloads PyTorch CUDA
packages and the ~3 GB Anime Whisper checkpoint.

Single-line Windows CMD command to clone, install and launch:

```cmd
cd /d C:\Games && git clone --branch experiment/anime-whisper-plamo2b --single-branch https://github.com/KarimPai/video-translate-main.git video-translate-anime && cd /d C:\Games\video-translate-anime && uv sync --no-progress && "Start translate.bat"
```

To update an existing clone:

```cmd
cd /d C:\Games\video-translate-anime && git pull --ff-only origin experiment/anime-whisper-plamo2b && uv sync --no-progress && "Start translate.bat"
```

For GPU memory headroom, the launcher keeps MPV software video decoding
(`--hwdec no`) and PLaMo GPU offload at 16 layers. The running llama-server
at 127.0.0.1:8000 may be reused, including its *existing* offload settings.
Stop a previous server if you need to enforce the test profile.

The launcher prints only JP/EN translations. Full logs:
`logs\run.log`, `logs\mpv.log`, `logs\plamo-server.log`.

## Testing

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_anime_whisper.py" -v
```

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_whisper_plamo.py" -v
```

```cmd
uv run --no-sync python -m unittest discover -s tests -p "test_anime_accuracy.py" -v
```

Optional local overrides to disable output filters for a full verbatim comparison:

```toml
[whisper]
suppress_standalone_laughter = false
suppress_repeated_japanese_outros = false
adaptive_max_tokens = false
retry_uncertain = false
collect_token_scores = false
```

The previous long-VOD seek crash (CUDA error 710) was reported from the VAD
stage, but CUDA errors are asynchronous and the true originating kernel is
not established. This branch isolates FireRedVAD on CPU, and avoids calling
`torch.cuda.empty_cache()` during shutdown, which previously masked the first
error. A sticky CUDA device assertion requires a fresh process; it cannot
safely be recovered within the same Python process. If a crash recurs, set
`CUDA_LAUNCH_BLOCKING=1` before launching to collect a more precise trace.

The tests use fake output models or dependency injection; they do **not**
establish CUDA runtime compatibility or Hololive ASR quality. For an accuracy
comparison, replay the same 2-minute source excerpt in `whisper-v3` and
this Anime Whisper experiment, count missed/invented Japanese words, and
inspect repeated phrases such as ご視聴ありがとうございました.

## Granite cleanup

This branch is **not** derived from Granite. It was created from
`whisper-v3`. After confirming the Anime Whisper checkout works, Granite
experiment branches and the old local `C:\Games\video-translate-granite`
folder can be deleted separately. Do not delete `main` or `whisper-v3`.
