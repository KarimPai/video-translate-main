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
    threshold: float = 0.4
    neg_threshold: float = 0.25
    min_silence_ms: int = 350
    speech_pad_ms: int = 200
    min_speech_ms: int = 180
    max_utterance_s: float = 5.0
    fallback_s: float = 5.0
    fallback_min_s: float = 1.5
    fallback_min_db: float = -45.0


@dataclass
class AsrConfig:
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    blocklist: list[str] = field(default_factory=list)


@dataclass
class HyMt2Config:
    """HY-MT2 served through a local OpenAI-compatible chat endpoint."""

    model: str = "kaelri/hy-mt2:1.8b-q4_K_M"
    base_url: str = "http://127.0.0.1:11434/v1"
    temperature: float = 0.2
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    min_request_interval_s: float = 3.0
    extra_body: dict = field(default_factory=dict)


@dataclass
class TranslateConfig:
    preset: str = "hymt2"
    context_lines: int = 8
    glossary: str = "glossary.toml"
    presets: dict[str, dict] = field(default_factory=lambda: {
        "hymt2": {
            "backend": "hymt2",
            "model": "kaelri/hy-mt2:1.8b-q4_K_M",
        }
    })


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
        "bv*[height<=480][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=480][protocol^=m3u8]/b[protocol^=m3u8]"
    )
    en_size: int = 44
    ja_size: int = 28
    speaker_diarization: bool = False
    speaker_similarity_threshold: float = 0.72
    speaker_device: str = "cuda"
    max_speakers: int = 3
    max_subtitle_rows: int = 6
    min_display_s: float = 1.5
    hold_s: float = 1.0


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

    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**asr_data),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
