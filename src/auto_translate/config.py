"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field, fields, replace
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
    # Accuracy-oriented speech recall on Japanese gaming dialogue.
    # Sensitivity is balanced with the existing no-probe / Whisper guards.
    threshold: float = 0.43
    neg_threshold: float = 0.28
    min_silence_ms: int = 500
    speech_pad_ms: int = 300
    min_speech_ms: int = 180
    short_reply_ms: int = 120
    short_reply_probability: float = 0.85
    onset_ms: int = 70  # shorter sustained-speech gate, preserve clipped onsets
    prob_smooth_frames: int = 5
    max_utterance_s: float = 12.0  # VOD: more sentence context for direct AST
    max_split_extension_s: float = 2.0  # VOD hard cap 14 seconds
    live_max_utterance_s: float = 7.0  # live: avoid long queued subtitles
    live_max_split_extension_s: float = 1.0  # live hard cap 8 seconds
    split_pause_min_ms: int = 100
    split_context_ms: int = 160
    fallback_s: float = 0.0  # opt-in probes risk invented words on non-speech/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0


def vad_for_source(cfg: VadConfig, *, live: bool) -> VadConfig:
    """VOD gets longer utterances; live remains bounded for subtitle timing.

    Preserve every configured VAD setting and never mutate the original.
    """
    if not live:
        return cfg
    return replace(
        cfg,
        max_utterance_s=cfg.live_max_utterance_s,
        max_split_extension_s=cfg.live_max_split_extension_s,
    )


@dataclass
class WhisperConfig:
    """Whisper Large-v3 direct Japanese audio-to-English translation."""

    model: str = "large-v3"  # not large-v3-turbo (Turbo is not trained for AST)
    device: str = "cuda"
    compute_type: str = "int8_float16"  # stable RTX 3070 8 GB video + Whisper baseline; FP16 opt-in
    download_root: str = ""  # empty uses the normal Hugging Face cache
    language: str = "ja"  # fixed Japanese source language
    task: str = "translate"  # direct Japanese audio to English
    beam_size: int = 6  # accuracy-first beam search, slower but same model
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = False
    # Conservative safeguards for short, noisy VTuber/gaming utterances.
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 192  # less risk of cutting off longer VOD English
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
    # Separately filter full-cue subtitle credits / subscribe prompts.
    filter_phantom_phrases: bool = True
    retry_temperature: float = 0.2
    # Only recover a weak translation when both decoding passes agree
    # verbatim, FireRedVAD is confident, and both passes indicate speech.
    rescue_stable_weak_speech: bool = True
    retry_max_audio_s: float = 15.0  # allow one retry on VOD utterances
    # Targeted silent-first-pass recovery: only clear Japanese speech evidence
    # plus very strong VAD may rescue a completely empty AST output.
    recover_empty_speech: bool = True
    recover_min_vad_confidence: float = 0.88
    recover_max_no_speech_prob: float = 0.45
    recover_max_audio_s: float = 8.0
    # Additional per-cue VAD/decoder timing and quality log entries.
    diagnostics: bool = False
    # Apply a tiny set of neutral English rewrites only when a Japanese
    # transcription of the same audio supports them. No new model/LLM.
    neutral_subtitles: bool = True
    neutral_verify_max_audio_s: float = 8.0


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
    rtx_vsr: bool = False  # opt-in 480p NVIDIA RTX Video Super Resolution
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
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
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
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
