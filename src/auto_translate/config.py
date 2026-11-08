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
    max_utterance_s: float = 6.0
    fallback_s: float = 5.0
    fallback_min_s: float = 1.5
    fallback_min_db: float = -45.0


@dataclass
class SpeakerConfig:
    """CPU Sherpa speaker diarization of each bounded VAD utterance."""
    enabled: bool = True
    model_dir: str = ""
    num_threads: int = 2
    min_turn_s: float = 0.55
    max_turns: int = 5


@dataclass
class AsrConfig:
    """Qwen3-ASR-1.7B-JA GGUF on speech.cpp. No Haas or fallback backend."""

    model_path: str = ""  # empty: verified cached Hugging Face Q8_0 conversion
    device: str = "Vulkan0"
    prompt: str = ""  # optional small Japanese name hint; default off
    blocklist: list[str] = field(default_factory=list)


@dataclass
class TranslateConfig:
    """Qwen3.5-4B GGUF is the sole local Japanese-to-English translator."""

    # Port 8001 avoids accidentally reusing a lingering older model server.
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = "qwen3.5-4b"
    qwen_hf_model: str = "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    qwen_context: int = 4096
    qwen_gpu_layers: int = 99
    # "gist" favors short meaning-first subtitles; "faithful" is closer to literal.
    subtitle_style: str = "gist"
    max_tokens: int = 128
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    context_lines: int = 5
    memory_lines: int = 12
    # Tiny rule-based, source-only scene topic; no extra Qwen requests.
    scene_context_enabled: bool = True
    scene_context_max_idle_lines: int = 32
    glossary: str = "glossary.toml"
    quality_recheck_enabled: bool = True
    quality_recheck_timeout_s: float = 3.0
    quality_recheck_min_interval_s: float = 10.0


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
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
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
    # Ignore both retired pitch settings; old git-ignored local configs
    # remain compatible without modifying the original PCM.
    ingest_data.pop("pitch_semitones", None)
    ingest_data.pop("pitch_shift_semitones", None)

    # Old Haas/config values may still exist in the git-ignored config.toml.
    # Ignore them instead of re-enabling a retired engine.
    asr_data = dict(data.get("asr", {}))
    valid_asr_keys = {entry.name for entry in fields(AsrConfig)}
    asr_data = {
        key: value for key, value in asr_data.items()
        if key in valid_asr_keys
    }

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

    # Keep the existing git-ignored config.toml compatible when switching
    # branches: ignore all outdated engine fields, not just known names.
    valid_translate_keys = {entry.name for entry in fields(TranslateConfig)}
    translate_data = {
        key: value for key, value in data.get("translate", {}).items()
        if key in valid_translate_keys
    }

    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        speaker=SpeakerConfig(**speaker_data),
        asr=AsrConfig(**asr_data),
        translate=TranslateConfig(**translate_data),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
