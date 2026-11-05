"""Load config.toml into typed dataclasses (missing keys fall back to the defaults below)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestConfig:
    # Live streams usually have no audio-only format, so fall back to 93 (360p, 128 kbps audio).
    ytdlp_format: str = "bestaudio/93/best"
    cookies_from_browser: str = ""  # e.g. "chrome" for members-only streams


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
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_silence_ms: int = 350
    speech_pad_ms: int = 200
    min_speech_ms: int = 180
    max_utterance_s: float = 5.0
    fallback_s: float = 5.0
    fallback_min_s: float = 1.5
    fallback_min_db: float = -45.0


@dataclass
class AsrConfig:
    backend: str = "faster-whisper"
    model: str = "kotoba-tech/kotoba-whisper-v2.0-faster"
    qwen_model: str = "Qwen/Qwen3-ASR-0.6B-hf"
    qwen_python: str = r"C:\Games\qwen3-asr-env\Scripts\python.exe"
    parakeet_model: str = "models/parakeet-ja/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
    parakeet_threads: int = 2
    haas_model: str = "haas-ja-v1.0"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5
    initial_prompt: str = ""
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    blocklist: list[str] = field(default_factory=list)
    probe_no_speech_max: float = 0.5
    probe_log_prob_min: float = -0.7


@dataclass
class Ct2NmtConfig:
    """A CTranslate2 seq2seq model (Opus-MT, Sugoi) + SentencePiece tokenizers."""

    model: str
    source_spm: str
    target_spm: str
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 4


@dataclass
class OpenAICompatConfig:
    """Any server with an OpenAI-style /v1/chat/completions endpoint."""

    model: str
    api_key_env: str = ""
    base_url: str = "http://127.0.0.1:11434/v1"
    temperature: float = 0.2
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0
    min_request_interval_s: float = 3.0
    extra_body: dict = field(default_factory=dict)
    


@dataclass
class TranslateConfig:
    preset: str = "balanced"
    context_lines: int = 4
    glossary: str = "glossary.toml"
    presets: dict[str, dict] = field(default_factory=dict)


@dataclass
class OutputConfig:
    terminal: bool = True
    web: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = 8765


@dataclass
class MpvConfig:
    """M4: play the stream in mpv delay_s behind live, with subtitles drawn in sync."""

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
    speaker_diarization: bool = True
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

    return Config(
        ingest=IngestConfig(**data.get("ingest", {})),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**data.get("mpv", {})),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )