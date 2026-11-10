"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


VIDEO_QUALITY_HEIGHTS = (360, 480, 720, 1080)


def mpv_youtube_format(max_height: int, *, live: bool) -> str:
    """Choose max requested YouTube resolution, with lower-quality fallback."""
    if max_height not in VIDEO_QUALITY_HEIGHTS:
        raise ValueError(f"Unsupported video height: {max_height}")
    ceiling = f"[height<={max_height}]"
    if live:
        return (
            f"bv*{ceiling}[vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
            f"/b{ceiling}[protocol^=m3u8]"
            f"/bv*{ceiling}+ba/b{ceiling}"
        )
    return (
        f"bv*{ceiling}[vcodec^=avc1][protocol=https]+ba[protocol=https]"
        f"/bv*{ceiling}[protocol=https]+ba[protocol=https]"
        f"/bv*{ceiling}[vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        f"/b{ceiling}[protocol^=m3u8]"
        f"/bv*{ceiling}+ba/b{ceiling}"
    )


@dataclass
class IngestConfig:
    ytdlp_format: str = "bestaudio/93/best"
    cookies_from_browser: str = ""
    pitch_semitones: float = 0.0


@dataclass
class SeparationConfig:
    """GPU vocal separation using audio-separator."""

    enabled: bool = False
    model: str = "pretrained_models/separation/UVR-MDX-NET-Voc_FT.onnx"


@dataclass
class DeepFilterConfig:
    """Real-time speech enhancement using DeepFilterNet3."""

    enabled: bool = False


@dataclass
class VadConfig:
    """FireRedVAD: stricter false-speech suppression for noisy gaming streams."""

    threshold: float = 0.45
    neg_threshold: float = 0.30
    min_silence_ms: int = 500
    speech_pad_ms: int = 200
    min_speech_ms: int = 200
    short_reply_ms: int = 120
    short_reply_probability: float = 0.85
    onset_ms: int = 100
    prob_smooth_frames: int = 5
    max_utterance_s: float = 11.0
    max_split_extension_s: float = 1.0
    split_pause_min_ms: int = 80
    split_context_ms: int = 120
    fallback_s: float = 0.0  # do not ASR-decode VAD-negative music/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0


@dataclass
class AsrConfig:
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    blocklist: list[str] = field(default_factory=list)


@dataclass
class TranslateConfig:
    """PLaMo 3 Translate 2B settings (sole translation engine)."""

    plamo_base_url: str = "http://127.0.0.1:8000/v1"
    plamo_model: str = "plamo-3-translate"
    plamo_hf_model: str = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    plamo_context: int = 4096  # smaller KV/context footprint on RTX 3070
    plamo_gpu_layers: int = 99
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    context_lines: int = 6  # reduced prompt carryover and translation latency
    # Maximum recent subtitle pairs used for lightweight terminology and scene memory.
    memory_lines: int = 12  # limit speculative previous-translation hints
    glossary: str = "glossary.toml"
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    suppress_runaway_loops: bool = True
    max_identical_cues: int = 2


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
    delay_s: float = 9.0
    start_wait_s: float = 3.0
    port: int = 9137
    hwdec: str = "auto-safe"
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
    deepfilter: DeepFilterConfig = field(default_factory=DeepFilterConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
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
    # Older configs may contain a -3 semitone ASR pitch workaround. Natural pitch
    # is the default for Haas Japanese recognition; ignore that legacy override.
    ingest_data.pop("pitch_semitones", None)

    asr_data = dict(data.get("asr", {}))
    for key in (
        "backend", "model", "device", "compute_type", "beam_size",
        "initial_prompt", "no_speech_threshold", "log_prob_threshold",
        "probe_no_speech_max", "probe_log_prob_min",
        "qwen_model", "qwen_python", "parakeet_model", "parakeet_threads",
    ):
        asr_data.pop(key, None)

    mpv_data = dict(data.get("mpv", {}))
    # Ignore removed playback and speaker settings in older configs.
    for obsolete in (
        "start_delay_s", "min_display_s", "hold_s", "speaker_diarization",
        "speaker_similarity_threshold", "speaker_device", "max_speakers",
    ):
        mpv_data.pop(obsolete, None)

    # Older local config.toml files may still have an engine selector and
    # settings from removed translation paths. Ignore those keys so an
    # existing installation automatically uses PLaMo without manual edits.
    translate_data = dict(data.get("translate", {}))
    for obsolete in (
        "backend", "model", "base_url", "temperature",
        "min_request_interval_s", "extra_body", "presets", "preset",
        "mpv_delay_s", "api_key_env",
    ):
        translate_data.pop(obsolete, None)

    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**asr_data),
        translate=TranslateConfig(**translate_data),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
