"""Load config.toml into typed dataclasses with backend-agnostic model selection.

The [models] section chooses which ASR and translation backends to use.
Backend-specific profiles live under [asr.<name>] and [translate.<name>].
"""

import logging
import math
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

logger = logging.getLogger(__name__)


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
class ModelsConfig:
    """Model selection — which backends to use."""

    asr_backend: str = "haas"
    translate_backend: str = "gemma4"
    vad_backend: str = "silero"


@dataclass
class IngestConfig:
    ytdlp_format: str = "bestaudio/93/best"
    cookies_from_browser: str = ""
    pitch_semitones: float = 0.0
    vod_max_audio_lead_s: float = 75.0

    def __post_init__(self):
        if not math.isfinite(self.vod_max_audio_lead_s) or self.vod_max_audio_lead_s < 0:
            raise ValueError("vod_max_audio_lead_s must be nonnegative")


@dataclass
class VadConfig:
    """Silero VAD configuration."""

    device: str = "cpu"
    threshold: float = 0.24
    neg_threshold: float = 0.16
    min_silence_ms: int = 400
    speech_pad_ms: int = 300
    min_speech_ms: int = 120
    short_reply_ms: int = 80
    short_reply_probability: float = 0.85
    onset_ms: int = 20
    prob_smooth_frames: int = 3
    max_utterance_s: float = 14.0
    max_split_extension_s: float = 2.0
    live_max_utterance_s: float = 7.0
    live_max_split_extension_s: float = 1.0
    adaptive_split: bool = True
    adaptive_split_s: float = 5.0
    live_adaptive_split_s: float = 4.0
    adaptive_pause_ms: int = 180
    split_pause_min_ms: int = 120
    split_context_ms: int = 160
    fallback_s: float = 12.0
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0
    model_dir: str = ""

    def __post_init__(self):
        if not 0 <= self.neg_threshold <= self.threshold <= 1 or not 0 <= self.short_reply_probability <= 1:
            raise ValueError("Invalid VAD probability thresholds")
        for key in ("min_speech_ms", "min_silence_ms", "short_reply_ms", "onset_ms", "prob_smooth_frames",
                    "max_utterance_s", "live_max_utterance_s", "adaptive_split_s", "live_adaptive_split_s",
                    "adaptive_pause_ms", "split_pause_min_ms", "fallback_min_s"):
            if not getattr(self, key) > 0:
                raise ValueError(f"vad.{key} must be positive")
        for key in ("speech_pad_ms", "split_context_ms", "max_split_extension_s", "live_max_split_extension_s", "fallback_s"):
            if not getattr(self, key) >= 0:
                raise ValueError(f"vad.{key} must be nonnegative")
        if self.device != "cpu":
            raise ValueError("SileroVAD must use cpu to preserve GPU memory for the models and playback")


LLAMA_SERVER_PATH = ".runtime/llama-b12318-cuda12/llama-server.exe"


@dataclass
class LanguageFilterConfig:
    enabled: bool = False
    min_duration_s: float = 2.5
    max_window_s: float = 5.0
    drop_probability: float = 0.80
    max_japanese_probability: float = 0.12
    min_top_margin: float = 0.45

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("language filter enabled must be boolean")
        if not 1.0 <= self.min_duration_s <= 10.0:
            raise ValueError("language filter min_duration_s must be 1..10 s")
        if not self.min_duration_s <= self.max_window_s <= 15.0:
            raise ValueError("language filter max_window_s must be >= minimum and <=15 s")
        for name in ("drop_probability", "max_japanese_probability", "min_top_margin"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not 0.0 <= value <= 1.0:
                raise ValueError("language filter "+name+" must be between 0 and 1")


@dataclass
class GpuBudgetConfig:
    startup_headroom_gb: float = 0.25
    low_free_gb: float = 0.25
    emergency_free_gb: float = 0.25
    sample_interval_s: float = 3.0

    def __post_init__(self):
        values = (self.startup_headroom_gb, self.low_free_gb,
                  self.emergency_free_gb, self.sample_interval_s)
        if any(type(value) not in (int, float) for value in values):
            raise ValueError("GPU headroom settings must be numeric")
        if not (0 <= self.emergency_free_gb <= self.low_free_gb
                <= self.startup_headroom_gb <= 2
                and 0.5 <= self.sample_interval_s <= 30):
            raise ValueError("invalid GPU free-memory headroom settings")


@dataclass
class TranslateConfig:
    """Common translation settings shared across backends.

    Backend-specific overrides live in config under [translate.<backend_name>].
    """

    model_path: str = "models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"
    server_path: str = LLAMA_SERVER_PATH
    gpu_layers: int = -1
    context_size: int = 4096
    cuda_device: int = 0
    threads: int = 8
    max_tokens: int = 128
    parallel_slots: int = 1
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.6
    repeat_penalty: float = 1.05
    context_lines: int = 6
    max_context_chars: int = 600
    context_max_gap_s: float = 15.0
    context_english: bool = False
    style: str = "Plain and concise"
    merge_fragments: bool = True
    merge_max_gap_s: float = 0.6
    merge_max_s: float = 16.0
    live_merge_margin_s: float = 1.5
    timeout_s: float = 30.0
    load_timeout_s: float = 300.0
    glossary: str = "glossary.toml"
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    max_identical_cues: int = 2
    suppress_empty_translation: bool = True
    suppress_japanese_in_english: bool = True
    vod_merge_timeout_s: float = 5.0

    def __post_init__(self):
        # model_path may be empty for non-default backends (PLaMo, Hy-MT2) —
        # the backend factory or constructor will raise a clear error if needed.
        for name in ("server_path", "glossary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"translate.{name} must be a nonempty path")
        # model_path must be a string but may be empty
        if not isinstance(self.model_path, str):
            raise TypeError("translate.model_path must be a string")
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("translate.gpu_layers must be -1 (auto) or a nonnegative integer")
        if type(self.context_size) is not int or not 512 <= self.context_size <= 32768:
            raise ValueError("translate.context_size must be an integer in 512..32768")
        if type(self.context_lines) is not int or not 0 <= self.context_lines <= 20:
            raise ValueError("translate.context_lines must be an integer in 0..20")
        if type(self.max_context_chars) is not int or not 0 <= self.max_context_chars <= 4000:
            raise ValueError("translate.max_context_chars must be an integer in 0..4000")
        if type(self.max_tokens) is not int or not 16 <= self.max_tokens <= 1024:
            raise ValueError("translate.max_tokens must be in 16..1024")
        if self.max_tokens >= self.context_size:
            raise ValueError("translate.max_tokens must leave room for input in context_size")
        if type(self.parallel_slots) is not int or self.parallel_slots not in (1, 2):
            raise ValueError("translate.parallel_slots must be 1 or 2")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("translate.top_k must be a nonnegative integer")
        for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1), ("repeat_penalty", 1, 2)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f"translate.{name} must be in {low}..{high}")
        if type(self.max_identical_cues) is not int or self.max_identical_cues < 1:
            raise ValueError("translate.max_identical_cues must be a positive integer")
        if (type(self.context_max_gap_s) not in (int, float)
                or not math.isfinite(self.context_max_gap_s) or self.context_max_gap_s <= 0):
            raise ValueError("translate.context_max_gap_s must be finite and positive")
        for name in ("context_english", "merge_fragments", "suppress_empty_translation",
                      "suppress_japanese_in_english"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"translate.{name} must be boolean")
        if (type(self.merge_max_gap_s) not in (int, float) or not 0 <= self.merge_max_gap_s <= 3
                or type(self.merge_max_s) not in (int, float) or not 1 <= self.merge_max_s <= 30):
            raise ValueError("translate.merge_max_gap_s must be in 0..3 and merge_max_s in 1..30")
        if type(self.live_merge_margin_s) not in (int, float) or not 0 <= self.live_merge_margin_s <= 10:
            raise ValueError("translate.live_merge_margin_s must be in 0..10")
        if not isinstance(self.style, str) or len(self.style) > 40 or any(c in self.style for c in "【】\n"):
            raise ValueError("translate.style must be a short single-line string without 【】")
        for value in (self.timeout_s, self.load_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Translation timeouts must be finite and positive")
        if (type(self.vod_merge_timeout_s) not in (int, float)
                or not math.isfinite(self.vod_merge_timeout_s) or self.vod_merge_timeout_s <= 0):
            raise ValueError("translate.vod_merge_timeout_s must be finite and positive")


@dataclass
class AsrConfig:
    """Common ASR settings shared across backends.

    Backend-specific overrides live in config under [asr.<backend_name>].
    """

    model_path: str = "haas-ja-v1.0"
    server_path: str = ""
    speech_server_path: str = ""
    mmproj_path: str = ""  # audio projector GGUF for multimodal ASR (e.g. Qwen3-ASR)
    context: str = ""
    max_context_chars: int = 2000
    recent_context_lines: int = 0
    context_retry_limit: int = 1
    context_size: int = 4096
    cuda_device: int = 0
    threads: int = 8
    blocklist: list[str] = field(default_factory=list)
    response_timeout_s: float = 30.0
    live_response_timeout_s: float = 15.0
    startup_timeout_s: float = 180.0

    def __post_init__(self):
        for name in ("model_path", "server_path"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"asr.{name} must be a string")
        if not isinstance(self.speech_server_path, str):
            raise TypeError("asr.speech_server_path must be a string")
        if self.speech_server_path and not self.speech_server_path.strip():
            raise ValueError("asr.speech_server_path must be a nonempty path or empty")
        if not isinstance(self.mmproj_path, str):
            raise TypeError("asr.mmproj_path must be a string")
        if not isinstance(self.context, str):
            raise TypeError("asr.context must be a string")
        if type(self.max_context_chars) is not int or not 0 <= self.max_context_chars <= 8000:
            raise ValueError("asr.max_context_chars must be an integer in 0..8000")
        if type(self.recent_context_lines) is not int or not 0 <= self.recent_context_lines <= 4:
            raise ValueError("asr.recent_context_lines must be an integer in 0..4")
        if type(self.context_retry_limit) is not int or self.context_retry_limit not in (0, 1):
            raise ValueError("asr.context_retry_limit must be 0 or 1")
        if type(self.context_size) is not int or not 1024 <= self.context_size <= 32768:
            raise ValueError("asr.context_size must be an integer in 1024..32768")
        if type(self.cuda_device) is not int or self.cuda_device < -1:
            raise ValueError("asr.cuda_device must be -1 (CPU) or a nonnegative CUDA device index")
        if type(self.threads) is not int or not 1 <= self.threads <= 256:
            raise ValueError("asr.threads must be an integer in 1..256")
        if not isinstance(self.blocklist, list) or any(not isinstance(x, str) or not x.strip() for x in self.blocklist):
            raise ValueError("asr.blocklist requires nonempty strings")
        for name in ("startup_timeout_s", "response_timeout_s", "live_response_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"asr.{name} must be finite and positive")


@dataclass
class NameFusionConfig:
    """Experimental, source-backed name prompts; disabled until A/B validated."""

    enabled: bool = False
    max_hints: int = 4
    max_prompt_chars: int = 96
    confirmations: int = 2

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("name_fusion.enabled must be boolean")
        if type(self.max_hints) is not int or not 1 <= self.max_hints <= 6:
            raise ValueError("name_fusion.max_hints must be 1..6")
        if type(self.max_prompt_chars) is not int or not 24 <= self.max_prompt_chars <= 160:
            raise ValueError("name_fusion.max_prompt_chars must be 24..160")
        if type(self.confirmations) is not int or not 1 <= self.confirmations <= 3:
            raise ValueError("name_fusion.confirmations must be 1..3")


@dataclass
class OutputConfig:
    terminal: bool = True
    web: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8765


@dataclass
class MpvConfig:
    enabled: bool = False
    path: str = ""
    delay_s: float = 20.0
    start_wait_s: float = 3.0
    port: int = 9137
    hwdec: str = "auto-safe"
    window_id: int | None = None
    live_late_window_s: float = 7.0
    ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    live_forward_cache_mib: int = 1024
    live_back_cache_mib: int = 1024
    en_size: int = 32
    ja_size: int = 0
    min_subtitle_duration_s: float = 1.5
    max_subtitle_rows: int = 4
    reading_cps: float = 15.0
    max_subtitle_duration_s: float = 7.0
    max_line_chars: int = 42
    max_cue_lines: int = 2
    background_opacity: float = 0.3
    vod_pause_threshold_s: float = 0.5
    vod_resume_threshold_s: float = 8.0

    def __post_init__(self):
        for value in (self.delay_s, self.start_wait_s, self.live_late_window_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("MPV timing values must be finite and nonnegative")
        if not 0 <= self.min_subtitle_duration_s <= 10:
            raise ValueError("MPV minimum subtitle duration must be between 0 and 10 seconds")
        if not 0 <= self.max_subtitle_duration_s <= 10:
            raise ValueError("mpv.max_subtitle_duration_s must be between 0 and 10 seconds")
        if type(self.reading_cps) not in (int, float) or not 0 <= self.reading_cps <= 60:
            raise ValueError("mpv.reading_cps must be in 0..60")
        if type(self.max_line_chars) is not int or not (self.max_line_chars == 0 or 16 <= self.max_line_chars <= 120):
            raise ValueError("mpv.max_line_chars must be 0 or in 16..120")
        if type(self.max_cue_lines) is not int or not 1 <= self.max_cue_lines <= 4:
            raise ValueError("mpv.max_cue_lines must be in 1..4")
        if type(self.background_opacity) not in (int, float) or not 0 <= self.background_opacity <= 1:
            raise ValueError("mpv.background_opacity must be in 0..1")
        for name in ("vod_pause_threshold_s", "vod_resume_threshold_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"mpv.{name} must be finite and nonnegative")
        if self.vod_resume_threshold_s <= self.vod_pause_threshold_s:
            raise ValueError("mpv.vod_resume_threshold_s must be greater than vod_pause_threshold_s")
        if type(self.en_size) is not int or not 8 <= self.en_size <= 96:
            raise ValueError("mpv.en_size must be in 8..96")
        if self.live_forward_cache_mib <= 0 or self.live_back_cache_mib < 0:
            raise ValueError("Invalid MPV live cache size")


@dataclass
class SafePlaybackConfig:
    enabled: bool = True
    max_inference_concurrency: int = 2
    queue_size: int = 16
    audio_chunks: int = 48
    max_audio_lead_s: float = 30.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("safe_playback.enabled must be boolean")
        for name, upper in (("max_inference_concurrency", 2), ("queue_size", 16), ("audio_chunks", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"safe_playback.{name} must be in 1..{upper}")
        if not math.isfinite(self.max_audio_lead_s) or not 1 <= self.max_audio_lead_s <= 75:
            raise ValueError("safe_playback.max_audio_lead_s must be in 1..75")


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    sample_interval_s: float = 4.0
    max_bytes: int = 2_000_000
    backups: int = 3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("diagnostics.enabled must be boolean")
        if not math.isfinite(self.sample_interval_s) or not .5 <= self.sample_interval_s <= 30:
            raise ValueError("diagnostics.sample_interval_s must be in .5..30")
        if type(self.max_bytes) is not int or not 1024 <= self.max_bytes <= 20_000_000:
            raise ValueError("diagnostics.max_bytes must be in 1024..20000000")
        if type(self.backups) is not int or not 1 <= self.backups <= 10:
            raise ValueError("diagnostics.backups must be in 1..10")


@dataclass
class HolodexConfig:
    api_key: str = ""
    timeout_s: float = 5.0

    def __post_init__(self):
        if not isinstance(self.api_key, str):
            raise TypeError("holodex.api_key must be a string")
        if type(self.timeout_s) not in (int, float) or not 0.5 <= self.timeout_s <= 30:
            raise ValueError("holodex.timeout_s must be in 0.5..30")


def vad_for_source(cfg: VadConfig, *, live: bool) -> VadConfig:
    """Use long, context-rich VOD chunks; bound live chunks for subtitle sync."""
    if not live:
        return cfg
    return replace(
        cfg,
        max_utterance_s=cfg.live_max_utterance_s,
        adaptive_split_s=min(cfg.live_adaptive_split_s, cfg.live_max_utterance_s),
        max_split_extension_s=cfg.live_max_split_extension_s,
    )


@dataclass
class Config:
    """Root configuration with backend-agnostic model selection."""

    models: ModelsConfig = field(default_factory=ModelsConfig)
    safe_playback: SafePlaybackConfig = field(default_factory=SafePlaybackConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    language_filter: LanguageFilterConfig = field(default_factory=LanguageFilterConfig)
    name_fusion: NameFusionConfig = field(default_factory=NameFusionConfig)
    gpu_budget: GpuBudgetConfig = field(default_factory=GpuBudgetConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mpv: MpvConfig = field(default_factory=MpvConfig)
    holodex: HolodexConfig = field(default_factory=HolodexConfig)
    # Backend-specific config overrides keyed by backend name
    asr_profiles: dict[str, dict] = field(default_factory=dict)
    translate_profiles: dict[str, dict] = field(default_factory=dict)
    stats_interval_s: float = 30.0

    def __post_init__(self):
        if not math.isfinite(self.stats_interval_s) or self.stats_interval_s <= 0:
            raise ValueError("stats_interval_s must be finite and positive")


def load_config(path: Path) -> Config:
    """Load config.toml into typed dataclasses.

    Reads [models] for backend selection, [asr.<name>] and [translate.<name>]
    for backend-specific profiles, and common sections for shared settings.
    """
    if not path.exists():
        return Config()

    with path.open("rb") as f:
        data = tomllib.load(f)

    # Validate known sections
    known_sections = {section.name for section in fields(Config)}
    for key in data.keys() - known_sections - {"models"}:
        # Allow [asr.*] and [translate.*] as backend profiles
        if not key.startswith("asr.") and not key.startswith("translate."):
            logger.warning("Ignoring unknown config section %s", key)

    # Backend-specific profiles
    # In TOML, [asr.qwen3-asr] creates data["asr"]["qwen3-asr"] (subtable)
    # but [asr] creates data["asr"] with common keys. Extract profiles from subtables.
    asr_profiles = {}
    translate_profiles = {}
    
    # Check for top-level keys like "asr.qwen3-asr" (non-standard but possible)
    for key in list(data.keys()):
        if key.startswith("asr."):
            backend_name = key.split(".", 1)[1]
            asr_profiles[backend_name] = data.pop(key)
        elif key.startswith("translate."):
            backend_name = key.split(".", 1)[1]
            translate_profiles[backend_name] = data.pop(key)
    
    # Extract profiles from TOML subtables: data["asr"]["qwen3-asr"] etc.
    asr_section = data.get("asr", {})
    for key in list(asr_section.keys()):
        if key not in {f.name for f in fields(AsrConfig)}:
            # This is a backend profile subtable (e.g. "qwen3-asr")
            asr_profiles[key] = asr_section.pop(key)
    
    # Handle legacy ASR config keys

    # Extract profiles from TOML subtables: data["translate"]["plamo"] etc.
    translate_section = data.get("translate", {})
    for key in list(translate_section.keys()):
        if key not in {f.name for f in fields(TranslateConfig)}:
            translate_profiles[key] = translate_section.pop(key)


    return Config(
        models=ModelsConfig(**data.get("models", {})),
        safe_playback=SafePlaybackConfig(**data.get("safe_playback", {})),
        diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
        language_filter=LanguageFilterConfig(**data.get("language_filter", {})),
        name_fusion=NameFusionConfig(**data.get("name_fusion", {})),
        gpu_budget=GpuBudgetConfig(**data.get("gpu_budget", {})),
        ingest=IngestConfig(**data.get("ingest", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**data.get("mpv", {})),
        holodex=HolodexConfig(**data.get("holodex", {})),
        asr_profiles=asr_profiles,
        translate_profiles=translate_profiles,
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )


# Model-identity fields that must never be inherited from common config
# when the backend profile is absent. These are backend-specific, not shared.
# server_path (llama-server executable) is NOT model identity — it is a shared
# runtime dependency that may be compatible across backends.
_MODEL_IDENTITY_ASR = {"model_path", "mmproj_path"}
_MODEL_IDENTITY_TRANSLATE = {"model_path"}

# Backends that are compatible with each other for model-identity inheritance.
# Haas is the default ASR; its model identity may be used by haas backend only.
# Gemma4 is the default translation; its model identity may be used by gemma4 only.
# Qwen3-ASR, Whisper, PLaMo, Hy-MT2 must never inherit another backend's model.
_DEFAULT_ASR_BACKEND = "haas"
_DEFAULT_TRANSLATE_BACKEND = "gemma4"


def _is_compatible_identity(backend_name: str, field: str, common_value: str,
                            backend_type: str) -> bool:
    """Check whether a common config model-identity value is compatible with the target backend.

    A common config value is compatible only if it belongs to the same backend
    (i.e., the target backend IS the default backend for that type).
    This prevents Qwen3-ASR from inheriting Haas's model path, etc.
    """
    if backend_type == "asr":
        return backend_name == _DEFAULT_ASR_BACKEND
    elif backend_type == "translate":
        return backend_name == _DEFAULT_TRANSLATE_BACKEND
    return False


def get_backend_config(cfg: Config, backend_type: str, backend_name: str,
                       explicit_overrides: dict | None = None,
                       selected_backend: str | None = None) -> dict:
    """Merge common config with backend-specific profile and explicit overrides.

    Precedence (highest to lowest):
    1. explicit_overrides (GUI/CLI overrides)
    2. Backend-specific profile ([asr.<name>] or [translate.<name>])
    3. Common config (cfg.asr or cfg.translate dataclass)

    Model-identity fields are resolved per-field:
    - If the backend has a profile with the field, use it.
    - If the backend is the default for its type (haas/gemma4), the common config
      value is used as the compatible default.
    - Otherwise, the field is empty — no cross-backend contamination.
    - Explicit GUI/CLI overrides always win.

    When `selected_backend` is provided, model_path overrides in `explicit_overrides`
    are scoped to that backend only, preventing --mt-model from overriding a different
    backend's model path.
    """
    if backend_type == "asr":
        common = {f.name: getattr(cfg.asr, f.name) for f in fields(cfg.asr)}
        profile = cfg.asr_profiles.get(backend_name, {})
        identity_fields = _MODEL_IDENTITY_ASR
    elif backend_type == "translate":
        common = {f.name: getattr(cfg.translate, f.name) for f in fields(cfg.translate)}
        profile = cfg.translate_profiles.get(backend_name, {})
        identity_fields = _MODEL_IDENTITY_TRANSLATE
    else:
        common = {}
        profile = {}
        identity_fields = set()

    # Layer 1: common config (lowest priority)
    merged = dict(common)

    # Per-field model-identity resolution: prevent cross-backend contamination.
    # For each identity field, check if the profile provides it. If not, only
    # keep the common value if this backend is the default for its type.
    for field_name in identity_fields:
        if field_name in profile:
            # Profile has the field — it will override via update() below
            pass
        elif _is_compatible_identity(backend_name, field_name, common.get(field_name, ""), backend_type):
            # This IS the default backend — common config value is compatible
            pass  # keep the common value
        else:
            # Not the default backend — remove incompatible identity from common
            merged.pop(field_name, None)

    # Layer 2: backend-specific profile overrides common
    merged.update(profile)
    # Layer 3: explicit GUI/CLI overrides (highest priority)
    # Scope model_path to the selected backend to prevent cross-backend contamination
    if explicit_overrides:
        scoped_overrides = dict(explicit_overrides)
        if selected_backend is not None and backend_name != selected_backend:
            # Remove model_path override for non-selected backends
            scoped_overrides.pop("model_path", None)
        merged.update(scoped_overrides)
    return merged


def resolve_model_identity(cfg: Config, backend_type: str, backend_name: str,
                           explicit_overrides: dict | None = None) -> dict:
    """Return model-identity fields from explicit override, backend profile, or compatible default.

    Resolves per field with precedence:
    1. Explicit override (GUI/CLI)
    2. Backend-specific profile
    3. Common config default (only if backend is the default for its type)
    4. Empty string (no compatible source)

    Returns a dict with keys: model_path, mmproj_path (ASR only), server_path.
    """
    if backend_type == "asr":
        profile = cfg.asr_profiles.get(backend_name, {})
        identity_keys = _MODEL_IDENTITY_ASR
        common = {f.name: getattr(cfg.asr, f.name) for f in fields(cfg.asr)}
    elif backend_type == "translate":
        profile = cfg.translate_profiles.get(backend_name, {})
        identity_keys = _MODEL_IDENTITY_TRANSLATE
        common = {f.name: getattr(cfg.translate, f.name) for f in fields(cfg.translate)}
    else:
        profile = {}
        identity_keys = set()
        common = {}

    result = {}
    for key in identity_keys:
        # Priority: explicit override > backend profile > compatible default > empty
        if explicit_overrides and key in explicit_overrides:
            result[key] = explicit_overrides[key]
        elif key in profile:
            result[key] = profile[key]
        elif _is_compatible_identity(backend_name, key, common.get(key, ""), backend_type):
            result[key] = common.get(key, "")
        else:
            result[key] = ""
    return result


__all__ = [
    "AsrConfig",
    "Config",
    "DiagnosticsConfig",
    "GpuBudgetConfig",
    "HolodexConfig",
    "IngestConfig",
    "LanguageFilterConfig",
    "ModelsConfig",
    "MpvConfig",
    "OutputConfig",
    "SafePlaybackConfig",
    "TranslateConfig",
    "VadConfig",
    "get_backend_config",
    "load_config",
    "mpv_youtube_format",
    "resolve_model_identity",
    "vad_for_source",
]
