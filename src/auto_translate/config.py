"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


VIDEO_QUALITY_HEIGHTS = (360, 480, 720, 1080)


def mpv_youtube_format(max_height: int, *, live: bool) -> str:
    """Build an mpv/yt-dlp selector with a strict resolution ceiling.

    Choose the best rendition *at or below* the requested height. Keep
    finished VODs on HTTPS/DASH first, then HLS fallback; live streams
    retain their existing HLS-first behavior.
    """
    if max_height not in VIDEO_QUALITY_HEIGHTS:
        raise ValueError(f"Unsupported YouTube video quality: {max_height}")
    height = f"[height<={max_height}]"
    if live:
        return (
            f"bv*{height}[vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
            f"/b{height}[protocol^=m3u8]"
            f"/bv*{height}+ba/b{height}"
        )
    return (
        f"bv*{height}[vcodec^=avc1][protocol=https]+ba[protocol=https]"
        f"/bv*{height}[protocol=https]+ba[protocol=https]"
        f"/bv*{height}[vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        f"/b{height}[protocol^=m3u8]"
        f"/bv*{height}+ba/b{height}"
    )


@dataclass
class IngestConfig:
    # Preserve the existing near-live audio selection independently of VODs.
    ytdlp_format: str = "bestaudio/93/best"
    # Completed VODs prefer direct HTTPS DASH-style audio tracks; the
    # unfiltered fallback still handles videos with only HLS renditions.
    vod_ytdlp_format: str = "bestaudio[protocol=https]/bestaudio/93/best"
    cookies_from_browser: str = ""
    # VOD with MPV only: cap decoded audio read-ahead relative to playback.
    # Prevent Whisper AST from decoding continuously hours ahead of the video.
    vod_max_audio_lead_s: float = 75.0  # 0 disables, preserves prior behavior

@dataclass
class SeparationConfig:
    """GPU vocal separation using audio-separator."""

    enabled: bool = False
    model: str = "pretrained_models/separation/UVR-MDX-NET-Voc_FT.onnx"


@dataclass
class VadConfig:
    threshold: float = 0.35
    neg_threshold: float = 0.30
    min_silence_ms: int = 400
    speech_pad_ms: int = 80
    min_speech_ms: int = 100
    short_reply_ms: int = 120
    short_reply_probability: float = 0.85
    onset_ms: int = 80  # continuous speech required before opening a segment
    prob_smooth_frames: int = 5  # smooth raw FireRed probabilities across 50 ms
    max_utterance_s: float = 11.0
    # At the soft limit, prefer a sustained natural pause.
    max_split_extension_s: float = 1.0  # hard deadline 12s
    split_pause_min_ms: int = 80
    split_context_ms: int = 120  # retain boundary audio, without duplication
    fallback_s: float = 0.0  # opt-in probes risk invented words on non-speech/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0


@dataclass
class WhisperConfig:
    """Whisper Large-v3 direct Japanese audio-to-English translation."""

    model: str = "large-v3"  # not large-v3-turbo (Turbo is not trained for AST)
    device: str = "cuda"
    compute_type: str = "int8_float16"  # stable RTX 3070 8 GB video + Whisper baseline; FP16 opt-in
    download_root: str = ""  # empty uses the normal Hugging Face cache
    language: str = "ja"  # fixed Japanese source language
    task: str = "translate"  # direct Japanese audio to English
    beam_size: int = 5
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = False
    # Conservative safeguards for short, noisy VTuber/gaming utterances.
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 160
    reject_low_confidence: bool = True
    min_avg_logprob: float = -1.20
    max_compression_ratio: float = 2.4
    noisy_segment_no_speech_prob: float = 0.60
    noisy_segment_min_avg_logprob: float = -0.50
    # Opt-in bounded English context; no cross-window Whisper feedback loop.
    use_text_context: bool = False
    text_context_lines: int = 2
    text_context_chars: int = 120
    text_context_max_gap_s: float = 5.0
    # Only exact, stream-relevant English names; no global Hololive glossary.
    hotwords: list[str] = field(default_factory=list)
    # Official talent roster; only stream-relevant names are sent to Whisper.
    hololive_names_enabled: bool = True
    hololive_name_limit: int = 6
    verify_standalone_names: bool = True  # re-decode isolated name without text context
    hololive_detected_names: list[str] = field(default_factory=list)
    # Optional style prompt; independent of the two cleanup switches below.
    # Whisper is not instruction-tuned: a style prompt is only a hint.
    concise_english: bool = False
    style_prompt: str = ""
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    # Retries decode the same utterance without prior audio context.
    audio_context_enabled: bool = False
    audio_context_s: float = 1.2
    audio_context_max_gap_s: float = 2.0
    retry_uncertain: bool = True
    # Do not re-decode a wholly rejected window when Whisper strongly
    # predicts non-speech. Higher values preserve more faint speech retries.
    no_speech_retry_block_threshold: float = 0.85
    # Verify stock English outro phrases against the original Japanese audio.
    verify_stock_outros: bool = True
    retry_temperature: float = 0.2
    # Only recover a weak translation when both decoding passes agree
    # verbatim, FireRedVAD is confident, and both passes indicate speech.
    rescue_stable_weak_speech: bool = True
    retry_max_audio_s: float = 12.0
    # Additional per-cue VAD/decoder timing and quality log entries.
    diagnostics: bool = False


@dataclass
class VoicePingAstConfig:
    """Direct JA-to-EN AST with no separate translator or aligner."""

    model: str = "voiceping-ai/qwen3-asr-ja-en-speech-translation"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_inference_batch_size: int = 1
    max_new_tokens: int = 192  # still ample for a 12-second English sentence
    # Native Qwen3-ASR context (not Whisper prompts or a separate translator).
    context_enabled: bool = True
    concise_context: bool = True
    hololive_names_enabled: bool = True
    hololive_name_limit: int = 6
    name_hints: list[str] = field(default_factory=list)
    detected_names: list[str] = field(default_factory=list)  # filled from video metadata
    repeat_short_cue_limit: int = 2  # keep two genuine repetitions, suppress more
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    diagnostics: bool = False


@dataclass
class OutputConfig:
    terminal: bool = True
    web: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8765


@dataclass
class MpvConfig:
    """Play the stream in mpv with subtitles drawn in sync."""

    enabled: bool = False
    path: str = ""
    delay_s: float = 8.0
    start_wait_s: float = 3.0
    port: int = 9137
    hwdec: str = "auto-safe"  # prefer hardware decode; fall back safely to software
    # Live source: retain the existing HLS-based live-edge handling.
    ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    # Completed VOD: select separately seekable HTTPS video/audio (DASH-like)
    # first. Preserve 1080p ceiling and H.264 preference, then HLS fallback.
    vod_ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol=https]+ba[protocol=https]"
        "/bv*[height<=1080][protocol=https]+ba[protocol=https]"
        "/bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]"
        "/bv*[height<=1080]+ba/b[height<=1080]"
    )
    en_size: int = 28
    ja_size: int = 0
    max_subtitle_rows: int = 3


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)  # legacy config compatibility only
    ast: VoicePingAstConfig = field(default_factory=VoicePingAstConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mpv: MpvConfig = field(default_factory=MpvConfig)
    stats_interval_s: float = 30.0


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()

    with path.open("rb") as f:
        data = tomllib.load(f)

    # Ignore legacy fields so existing config.toml files remain compatible.
    ingest_data = dict(data.get("ingest", {}))
    # Ignore both retired pitch settings; old git-ignored local configs
    # remain compatible without modifying the original PCM.
    ingest_data.pop("pitch_semitones", None)
    ingest_data.pop("pitch_shift_semitones", None)

    mpv_data = dict(data.get("mpv", {}))
    # Ignore removed playback and speaker settings in older configs.
    for obsolete in (
        "start_delay_s", "min_display_s", "hold_s", "speaker_diarization",
        "speaker_similarity_threshold", "speaker_device", "max_speakers",
    ):
        mpv_data.pop(obsolete, None)

    # Ignore retired Granite and incompatible legacy ASR/caption blocks.
    # Old git-ignored config.toml files remain safe to reuse.
    whisper_fields = {entry.name for entry in fields(WhisperConfig)}
    whisper_data = {
        name: value for name, value in data.get("whisper", {}).items()
        if name in whisper_fields
    }

    # Ignore legacy task and [translate] blocks from local config.toml.
    whisper_data.pop("task", None)
    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        vad=VadConfig(**data.get("vad", {})),
        whisper=WhisperConfig(**whisper_data),
        ast=VoicePingAstConfig(**{
            name: value for name, value in data.get("ast", {}).items()
            if name in {entry.name for entry in fields(VoicePingAstConfig)}
        }),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
