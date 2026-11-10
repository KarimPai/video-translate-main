"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class IngestConfig:
    ytdlp_format: str = "bestaudio/93/best"
    cookies_from_browser: str = ""

@dataclass
class SeparationConfig:
    """GPU vocal separation using audio-separator."""

    enabled: bool = False
    model: str = "pretrained_models/separation/UVR-MDX-NET-Voc_FT.onnx"


@dataclass
class GtcrnConfig:
    """Real-time 16 kHz GTCRN speech enhancement on CPU via sherpa-onnx."""

    enabled: bool = True
    num_threads: int = 1
    model_path: str = ""  # empty means cached official gtcrn_simple.onnx


@dataclass
class VadConfig:
    device: str = "cpu"  # isolate FireRedVAD from Anime Whisper GPU; "cuda" opt-in
    threshold: float = 0.50
    neg_threshold: float = 0.30
    min_silence_ms: int = 550
    speech_pad_ms: int = 250
    min_speech_ms: int = 180
    onset_ms: int = 80  # continuous speech required before opening a segment
    prob_smooth_frames: int = 5  # smooth raw FireRed probabilities across 50 ms
    max_utterance_s: float = 8.0
    # At the soft limit, prefer a sustained natural pause.
    max_split_extension_s: float = 1.0  # hard deadline remains 9s
    split_pause_min_ms: int = 80
    split_context_ms: int = 120  # retain boundary audio, without duplication
    fallback_s: float = 0.0  # opt-in probes risk invented words on non-speech/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0


@dataclass
class SpeakerConfig:
    """CPU Sherpa speaker diarization of each bounded VAD utterance."""
    enabled: bool = True
    model_dir: str = ""
    num_threads: int = 2
    min_turn_s: float = 0.55
    max_turns: int = 5
    boundary_search_ms: int = 180
    boundary_quiet_ms: int = 40
    boundary_quiet_ratio: float = 0.45
    interruption_handoffs: bool = True
    interruption_min_continuation_s: float = 0.25


@dataclass
class WhisperConfig:
    """Anime Whisper Japanese ASR (Whisper-derived); PLaMo translates later."""

    model: str = "litagin/anime-whisper"  # exact Japanese anime fine-tune
    device: str = "cuda"
    compute_type: str = "float16"  # torch fp16, not CTranslate2 INT8
    download_root: str = ""  # empty uses the normal Hugging Face cache
    language: str = "ja"  # fixed Japanese source language
    task: str = "transcribe"  # Japanese ASR ONLY; PLaMo handles English
    beam_size: int = 5
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = False
    # Conservative safeguards for short, noisy VTuber/gaming utterances.
    repetition_penalty: float = 1.0  # model card recommendation
    no_repeat_ngram_size: int = 5  # model card anti-repetition example
    max_new_tokens: int = 160
    adaptive_max_tokens: bool = True  # cap very short clips to reduce runaway filler
    reject_low_confidence: bool = True
    min_avg_logprob: float = -1.20
    max_compression_ratio: float = 2.4
    noisy_segment_no_speech_prob: float = 0.60
    noisy_segment_min_avg_logprob: float = -0.50
    # Do not categorically ban real signoffs: suppress repeated identical
    # outro-like phrases within this playback-time window.
    suppress_repeated_signoffs: bool = True
    signoff_cooldown_s: float = 1200.0
    # Standalone video outros are frequently hallucinated on weak game audio.
    # Allow at the actual end of a completed VOD, not during live/gameplay.
    suppress_generic_outros: bool = True
    suppress_standalone_laughter: bool = True  # output-only; never delete speech mixed with laughter
    suppress_repeated_japanese_outros: bool = True  # identical second occurrence only
    japanese_outro_cooldown_s: float = 120.0
    outro_allow_near_vod_end_s: float = 90.0
    # Opt-in bounded English context; no cross-window Whisper feedback loop.
    use_text_context: bool = False
    text_context_lines: int = 2
    text_context_chars: int = 120
    text_context_max_gap_s: float = 5.0
    # These legacy fields are ignored by the Anime Whisper model adapter.
    hotwords: list[str] = field(default_factory=list)
    # Legacy talent fields remain for config compatibility; no ASR prompts are sent.
    hololive_names_enabled: bool = True
    hololive_name_limit: int = 6
    hololive_detected_names: list[str] = field(default_factory=list)
    # DO NOT send ASR initial prompts to Anime Whisper (hallucination risk).
    concise_english: bool = True
    style_prompt: str = ""  # no instruction-style priming; concise output filter handles fillers
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    # Keep the recent speech waveform for *uncertain-only* contextual retries.
    audio_context_enabled: bool = True
    audio_context_s: float = 1.2
    audio_context_max_gap_s: float = 2.0
    retry_uncertain: bool = True  # rerun only objectively suspicious cues
    retry_on_outro: bool = True
    collect_token_scores: bool = True  # estimate average beam token log-prob
    retry_beam_size: int = 8  # deterministic unprimed second pass
    retry_min_logprob_gain: float = 0.18
    retry_temperature: float = 0.2  # legacy fallback/test compatibility
    retry_max_audio_s: float = 12.0
    # Additional per-cue VAD/decoder timing and quality log entries.
    diagnostics: bool = True  # record performance and transcript anomalies


@dataclass
class TranslateConfig:
    """Local PLaMo 3 Translate 2B GGUF, the sole JA->EN translation engine."""

    plamo_base_url: str = "http://127.0.0.1:8000/v1"
    plamo_model: str = "plamo-3-translate"
    plamo_hf_model: str = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    plamo_context: int = 4096  # RTX 3070 headroom for Whisper INT8/FP16
    plamo_gpu_layers: int = 16  # conservative RTX 3070 stability profile
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    context_lines: int = 8
    memory_lines: int = 24
    fragment_context_lines: int = 2
    fragment_max_gap_s: float = 1.8
    protect_numbers: bool = True
    protect_names: bool = True
    glossary: str = "glossary.toml"


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
    hwdec: str = "no"  # software decode leaves RTX 3070 VRAM for both models
    ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    en_size: int = 28
    ja_size: int = 0
    max_subtitle_rows: int = 3


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    gtcrn: GtcrnConfig = field(default_factory=GtcrnConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
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

    # Retired cross-utterance identity fields in local config are ignored.
    valid_speaker_keys = {entry.name for entry in fields(SpeakerConfig)}
    speaker_data = {
        key: value for key, value in data.get("speaker", {}).items()
        if key in valid_speaker_keys
    }

    # Ignore retired Granite and incompatible legacy ASR/caption blocks.
    # Old git-ignored config.toml files remain safe to reuse.
    whisper_fields = {entry.name for entry in fields(WhisperConfig)}
    whisper_data = {
        name: value for name, value in data.get("whisper", {}).items()
        if name in whisper_fields
    }

    # Retire the previous direct-translation setting even in old ignored
    # local config files. Japanese->English MUST happen only in PLaMo.
    whisper_data.pop("task", None)
    translate_fields = {entry.name for entry in fields(TranslateConfig)}
    translate_data = {
        name: value for name, value in data.get("translate", {}).items()
        if name in translate_fields
    }

    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        gtcrn=GtcrnConfig(**data.get("gtcrn", {})),
        vad=VadConfig(**data.get("vad", {})),
        speaker=SpeakerConfig(**speaker_data),
        whisper=WhisperConfig(**whisper_data),
        translate=TranslateConfig(**translate_data),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
