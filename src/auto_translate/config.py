"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
    threshold: float = 0.35
    neg_threshold: float = 0.25
    min_silence_ms: int = 400
    speech_pad_ms: int = 200
    min_speech_ms: int = 100
    max_utterance_s: float = 8.0
    fallback_s: float = 5.0
    fallback_min_s: float = 1.5
    fallback_min_db: float = -45.0


@dataclass
class AsrConfig:
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    # Optional Haas-ASR keyword boosting in speech.cpp 0.10+.
    haas_keywords_enabled: bool = True
    haas_keywords_file: str = "haas_vocabulary.toml"
    haas_keywords_alpha: float = 1.5
    haas_keywords_guard: bool = True
    blocklist: list[str] = field(default_factory=list)
    retry_uncertain_enabled: bool = True
    retry_max_audio_s: float = 8.0
    retry_budget_s: float = 3.0


@dataclass
class TranslateConfig:
    """PLaMo 3 Translate 2B settings (sole translation engine)."""

    plamo_base_url: str = "http://127.0.0.1:8000/v1"
    plamo_model: str = "plamo-3-translate"
    plamo_hf_model: str = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    plamo_context: int = 8192
    plamo_gpu_layers: int = 99
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    context_lines: int = 8
    # Maximum recent subtitle pairs used for lightweight terminology and scene memory.
    memory_lines: int = 24
    glossary: str = "glossary.toml"
    # Only dubious outputs receive a bounded second pass through the SAME PLaMo.
    quality_recheck_enabled: bool = True
    quality_recheck_timeout_s: float = 4.0
    quality_recheck_min_interval_s: float = 3.0


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
