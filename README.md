# Anime Whisper + PLaMo 3 Translate 2B

Branch: `experiment/anime-whisper-plamo2b` — independent accuracy experiment.
The stable `main`, `whisper-v3`, and other non-Granite branches remain unchanged.

## Pipeline

YouTube live/VOD -> FFmpeg mono 16 kHz -> FireRedVAD (CUDA)
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
  no_repeat_ngram_size 5, repetition_penalty 1.0.
- Re-transcription on uncertain output is OFF in this initial comparison so
  model differences are easier to interpret. The existing FireRedVAD silence
  gate, Japanese script gate, and local PLaMo translator remain active.
- The model can transcribe sighs, laughter, and expressive speech. This may
  produce more nonverbal Japanese text; evaluate against the actual audio
  rather than assuming these are hallucinations. PLaMo remains the only
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
