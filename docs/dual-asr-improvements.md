# Dual-ASR improvements — original models preserved

This branch starts from `feature/haas-whisper-turbo-dual-asr-20261114`. It is not a migration to Qwen, HY-MT2, BuzzASR, or Voxtral.

## Unchanged components
- Primary: Haas-ja-v1.0 on speech.cpp (Vulkan preferred).
- Secondary: Whisper Large-v3 Turbo via faster-whisper / CTranslate2.
- Japanese selection: the existing LFM2.5-1.2B-JP reconciler and acoustic guards.
- Japanese-to-English translation: PLaMo 3 Translate 2B, unchanged model identifier, prompt template, server and quantization settings.
- Existing GTK/Qt MPV GUI, optional VSR, subtitle timing, source playback, live chat/caption evidence and source model files.

## Ported improvements
- Faster repeated-generation detection, unpaired double-quote cleanup and a shared output validation boundary. Substantial Japanese text or obvious template output cannot leak into the English overlay or its translation-memory cache. Profanity is not censored.
- VAD onset recovery after a forced segment split: padded silence no longer masquerades as spoken onset.
- GPU peak-used and lowest-free-memory metrics, without an artificial VRAM ceiling.
- MPV wraps and time-slices long English subtitles, retains original RTX VSR controls, and logs rendered-cue onset lateness. Stale prior-generation livestream overlays are discarded.
- PLaMo dialogue history clears after 15 seconds of *media timeline* silence, and on seek/generation changes. It does not reset merely because the viewer pauses video.
- Original FireRedVAD sensitivity thresholds remain unchanged. A direct port of the Qwen branch's more aggressive VAD defaults admitted game noise in regression testing and was deliberately reverted.

## Optional more sensitive VAD profile
If quiet radio speech is missed, try these settings in `config.toml`, then A/B test against the original. They do not change either ASR model:
```toml
[vad]
threshold = 0.24
neg_threshold = 0.16
min_speech_ms = 120
short_reply_ms = 80
onset_ms = 20
prob_smooth_frames = 3
adaptive_split_s = 5.0
```
This profile may increase false positives from music and game sound effects. The original profile remains the default.

## Verification boundaries
Offline unit tests run on GitHub Actions for Windows/Linux; those tests do not prove acoustic accuracy, VRAM stability, GPU concurrency under Windows WDDM or livestream sync on an RTX 3070. No prebuilt EXE is changed by a GitHub source update. Do not overwrite or remove existing model caches.
