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

    enabled: bool = False
    num_threads: int = 1
    model_path: str = ""  # empty means cached official gtcrn_simple.onnx


@dataclass
class VadConfig:
    threshold: float = 0.30
    neg_threshold: float = 0.20
    min_silence_ms: int = 400
    speech_pad_ms: int = 200
    min_speech_ms: int = 100
    max_utterance_s: float = 7.0
    # At the soft limit, prefer a sustained natural pause.
    max_split_extension_s: float = 1.0  # hard deadline remains 8s
    split_pause_min_ms: int = 80
    split_context_ms: int = 120  # retain boundary audio, without duplication
    fallback_s: float = 0.0  # opt-in probes risk invented words on non-speech/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0


@dataclass
class SpeakerConfig:
    """CPU Sherpa speaker diarization of each bounded VAD utterance."""
    enabled: bool = False
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
class GraniteConfig:
    """One local Granite Speech 5.0 Q8 model for direct Japanese audio->English."""
    model_id: str = "ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0"
    base_url: str = "http://127.0.0.1:9797"
    llama_server_path: str = ""  # use llama-server.exe on PATH
    context_tokens: int = 4096
    gpu_layers: int = 99
    prompt: str = "translate the speech to English with proper punctuation and capitalization."
    # IBM's supported AST keyword-biasing format is used when keywords are set.
    # Keep the list small: overly broad lists can bias unrelated speech.
    # Set keywords=[] to use the punctuated AST prompt above instead.
    keywords: list[str] = field(default_factory=lambda: [
        "Watame", "Towa", "Suisei", "Pekora", "Miko", "Subaru",
    ])
    request_timeout_s: float = 45.0
    startup_timeout_s: float = 1200.0


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
    gtcrn: GtcrnConfig = field(default_factory=GtcrnConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    granite: GraniteConfig = field(default_factory=GraniteConfig)
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

    # Legacy [asr], [translate] and [captions] blocks are intentionally
    # ignored. The Granite-only branch never initializes those models.
    granite_data = {
        name: value for name, value in data.get("granite", {}).items()
        if name in {entry.name for entry in fields(GraniteConfig)}
    }

    return Config(
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        gtcrn=GtcrnConfig(**data.get("gtcrn", {})),
        vad=VadConfig(**data.get("vad", {})),
        speaker=SpeakerConfig(**speaker_data),
        granite=GraniteConfig(**granite_data),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**mpv_data),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
