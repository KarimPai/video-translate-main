"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import logging
import math
import tomllib
from dataclasses import dataclass, field, fields, replace
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
    vod_max_audio_lead_s: float = 75.0  # MPV VOD only: limit idle audio read-ahead

    def __post_init__(self):
        if not math.isfinite(self.vod_max_audio_lead_s) or self.vod_max_audio_lead_s < 0:
            raise ValueError("vod_max_audio_lead_s must be nonnegative")


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
    """FireRedVAD: accuracy-first segmentation with conservative noise gating.

    Retain more quiet/initial speech, avoid word clipping, and allow longer
    complete utterances for VOD. Live has a shorter cap for subtitle timing.
    """

    device: str = "cpu"  # keep PyTorch CUDA context off the RTX 3070 by default
    threshold: float = 0.24
    neg_threshold: float = 0.16
    min_silence_ms: int = 550  # wait for genuine phrase-ending silence
    speech_pad_ms: int = 400  # preserve soft/overlapped word starts and tails
    min_speech_ms: int = 120
    short_reply_ms: int = 80
    short_reply_probability: float = 0.85
    onset_ms: int = 20
    prob_smooth_frames: int = 3
    max_utterance_s: float = 14.0  # VOD soft cap: more sentence context
    max_split_extension_s: float = 2.0  # VOD hard cap: 16s
    live_max_utterance_s: float = 7.0  # fits around 9s live buffer
    live_max_split_extension_s: float = 1.0  # live hard cap: 8s
    # Natural-pause splitting begins before the long-form hard cap.
    adaptive_split: bool = True
    adaptive_split_s: float = 5.0  # VOD: keep context unless a clear pause occurs
    live_adaptive_split_s: float = 4.0  # live: leave room for ASR + translation
    adaptive_pause_ms: int = 180  # avoid cutting at very brief VAD dips
    split_pause_min_ms: int = 120
    split_context_ms: int = 160
    fallback_s: float = 0.0  # do not ASR-decode VAD-negative music/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0

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
            raise ValueError("FireRedVAD must use cpu to preserve GPU memory for the models and playback")


LLAMA_SERVER_PATH = ".runtime/llama-b12318-cuda12/llama-server.exe"


@dataclass
class AsrConfig:
    """Select the dual Haas/Whisper engine or retain Qwen as an optional fallback."""

    backend: str = "dual"  # dual (default), qwen
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    haas_keywords_enabled: bool = False
    haas_keywords_alpha: float = 1.0
    secondary_model: str = "h2oai/faster-whisper-large-v3-turbo"
    secondary_device: str = "cuda"
    secondary_compute_type: str = "int8_float16"
    secondary_beam_size: int = 6
    secondary_live_mode: str = "selective"  # off, selective, all
    secondary_vod_mode: str = "all"
    secondary_selective_chars: int = 9
    secondary_vod_timeout_s: float = 25.0
    secondary_live_timeout_s: float = 8.0
    secondary_startup_timeout_s: float = 180.0
    model_path: str = "models/qwen3-asr/Qwen3-ASR-1.7B-Q8_0.gguf"
    mmproj_path: str = "models/qwen3-asr/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"
    server_path: str = LLAMA_SERVER_PATH
    context: str = ""
    max_context_chars: int = 2000
    recent_context_lines: int = 2  # previous accepted transcripts added to the ASR system prompt
    context_size: int = 4096
    cuda_device: int = 0
    threads: int = 8
    blocklist: list[str] = field(default_factory=list)
    response_timeout_s: float = 30.0
    live_response_timeout_s: float = 15.0
    startup_timeout_s: float = 180.0

    def __post_init__(self):
        if self.backend not in ("dual", "qwen"):
            raise ValueError("asr.backend must be dual or qwen")
        if self.haas_device == "" or not self.haas_model.strip():
            raise ValueError("Haas model and device must be nonempty")
        if self.secondary_live_mode not in ("off", "selective", "all") or self.secondary_vod_mode not in ("off", "selective", "all"):
            raise ValueError("secondary mode must be off, selective, or all")
        if self.secondary_device not in ("cpu", "cuda") or not self.secondary_model.strip():
            raise ValueError("Invalid Whisper model or device")
        if self.secondary_compute_type not in ("int8_float16", "float16", "int8", "float32"):
            raise ValueError("Invalid Whisper compute type")
        if type(self.secondary_beam_size) is not int or not 1 <= self.secondary_beam_size <= 12:
            raise ValueError("Whisper beam must be in 1..12")
        if type(self.secondary_selective_chars) is not int or not 0 <= self.secondary_selective_chars <= 100:
            raise ValueError("Secondary selective characters must be in 0..100")
        for name in ("secondary_vod_timeout_s", "secondary_live_timeout_s", "secondary_startup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"asr.{name} must be positive")
        for name in ("model_path", "mmproj_path", "server_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"asr.{name} must be a nonempty path")
        if not isinstance(self.context, str):
            raise ValueError("asr.context must be a string")
        if type(self.max_context_chars) is not int or not 0 <= self.max_context_chars <= 8000:
            raise ValueError("asr.max_context_chars must be an integer in 0..8000")
        if type(self.recent_context_lines) is not int or not 0 <= self.recent_context_lines <= 4:
            raise ValueError("asr.recent_context_lines must be an integer in 0..4")
        if type(self.context_size) is not int or not 1024 <= self.context_size <= 32768:
            raise ValueError("asr.context_size must be an integer in 1024..32768")
        if type(self.cuda_device) is not int or self.cuda_device < 0:
            raise ValueError("asr.cuda_device must be a nonnegative integer")
        if type(self.threads) is not int or not 1 <= self.threads <= 256:
            raise ValueError("asr.threads must be an integer in 1..256")
        if not isinstance(self.blocklist, list) or any(not isinstance(x, str) or not x.strip() for x in self.blocklist):
            raise ValueError("asr.blocklist requires nonempty strings")
        for name in ("startup_timeout_s", "response_timeout_s", "live_response_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"asr.{name} must be finite and positive")


@dataclass
class LanguageFilterConfig:
    """Optional pre-ASR language gate; Qwen3-ASR already tags each utterance.

    Ambiguous or very short utterances stay in the original Japanese path to
    avoid losing Japanese speech during VTuber code-switching.
    """
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
    """GPU capacity-relative headroom; no fixed application VRAM limit."""

    startup_headroom_gb: float = 1.0
    low_free_gb: float = 0.75
    emergency_free_gb: float = 0.25
    sample_interval_s: float = 2.0

    def __post_init__(self):
        values = (self.startup_headroom_gb, self.low_free_gb,
                  self.emergency_free_gb, self.sample_interval_s)
        if any(type(value) not in (int, float) for value in values):
            raise ValueError("GPU headroom settings must be numeric")
        if not (0 <= self.emergency_free_gb < self.low_free_gb
                <= self.startup_headroom_gb <= 2
                and 0.5 <= self.sample_interval_s <= 30):
            raise ValueError("invalid GPU free-memory headroom settings")


@dataclass
class TranslateConfig:
    """Tencent Hy-MT2-7B on an owned llama.cpp server (sole translation engine)."""

    model_path: str = "models/hy-mt/Hy-MT2-7B-Q4_K_M.gguf"
    server_path: str = LLAMA_SERVER_PATH
    gpu_layers: int = -1  # -1: offload as many layers as GPU headroom allows
    context_size: int = 2048
    cuda_device: int = 0
    threads: int = 8
    max_tokens: int = 192
    # Tencent's recommended sampling, at a lower temperature for subtitle stability.
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.6
    repeat_penalty: float = 1.05
    context_lines: int = 6
    max_context_chars: int = 600  # includes stream metadata
    context_max_gap_s: float = 15.0
    context_english: bool = False  # also show prior English; can propagate earlier mistakes
    # Hy-MT2 style instruction (Chinese works best); empty uses the default template.
    style: str = "简洁直白"
    # Join a cue without sentence-final punctuation with the next close cue.
    # Live holds a fragment only until playback nears its onset.
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

    def __post_init__(self):
        for name in ("model_path", "server_path", "glossary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"translate.{name} must be a nonempty path")
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("translate.gpu_layers must be -1 (auto) or a nonnegative integer")
        if type(self.context_size) is not int or not 512 <= self.context_size <= 32768:
            raise ValueError("translate.context_size must be an integer in 512..32768")
        if type(self.context_lines) is not int or not 0 <= self.context_lines <= 20:
            raise ValueError("translate.context_lines must be an integer in 0..20")
        if type(self.max_context_chars) is not int or not 0 <= self.max_context_chars <= 4000:
            raise ValueError("translate.max_context_chars must be an integer in 0..4000")
        if type(self.max_tokens) is not int or not 16 <= self.max_tokens <= 1024:
            raise ValueError("translate.max_tokens must be an integer in 16..1024")
        if self.max_tokens >= self.context_size:
            raise ValueError("translate.max_tokens must leave room for input in context_size")
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
        for name in ("context_english", "merge_fragments"):
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
    delay_s: float = 20.0
    start_wait_s: float = 3.0
    port: int = 9137
    hwdec: str = "auto-safe"
    window_id: int | None = None  # optional native Qt HWND for MPV embedding
    live_late_window_s: float = 7.0  # reject extremely stale spoken cues
    ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    live_forward_cache_mib: int = 1024
    live_back_cache_mib: int = 1024
    en_size: int = 32
    ja_size: int = 0
    min_subtitle_duration_s: float = 1.5
    max_subtitle_rows: int = 4  # wrapped screen rows across stacked cues
    # Readability: extend short cues to a reading speed, wrap and split long ones.
    reading_cps: float = 15.0  # 0 disables reading-speed extension
    max_subtitle_duration_s: float = 7.0
    max_line_chars: int = 42  # 0 disables wrapping and splitting
    max_cue_lines: int = 2
    background_opacity: float = 0.3  # dark box behind subtitles; 0 draws outline only

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
        if type(self.en_size) is not int or not 8 <= self.en_size <= 96:
            raise ValueError("mpv.en_size must be in 8..96")
        if self.live_forward_cache_mib <= 0 or self.live_back_cache_mib < 0:
            raise ValueError("Invalid MPV live cache size")


@dataclass
class SafePlaybackConfig:
    enabled: bool = False
    max_inference_concurrency: int = 1
    queue_size: int = 4
    audio_chunks: int = 32
    max_audio_lead_s: float = 20.0

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
    sample_interval_s: float = 2.0
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
    """Optional cast lookup; the key may also come from HOLODEX_API_KEY."""

    api_key: str = ""
    timeout_s: float = 5.0

    def __post_init__(self):
        if not isinstance(self.api_key, str):
            raise ValueError("holodex.api_key must be a string")
        if type(self.timeout_s) not in (int, float) or not 0.5 <= self.timeout_s <= 30:
            raise ValueError("holodex.timeout_s must be in 0.5..30")


def vad_for_source(cfg: VadConfig, *, live: bool) -> VadConfig:
    """Use long, context-rich VOD chunks; bound live chunks for subtitle sync.

    This returns a separate dataclass, preserving all local configuration and
    leaving the caller's original VAD settings unchanged.
    """
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
    safe_playback: SafePlaybackConfig = field(default_factory=SafePlaybackConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    language_filter: LanguageFilterConfig = field(default_factory=LanguageFilterConfig)
    gpu_budget: GpuBudgetConfig = field(default_factory=GpuBudgetConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    deepfilter: DeepFilterConfig = field(default_factory=DeepFilterConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mpv: MpvConfig = field(default_factory=MpvConfig)
    holodex: HolodexConfig = field(default_factory=HolodexConfig)
    stats_interval_s: float = 30.0

    def __post_init__(self):
        if not math.isfinite(self.stats_interval_s) or self.stats_interval_s <= 0:
            raise ValueError("stats_interval_s must be finite and positive")


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()

    with path.open("rb") as f:
        data = tomllib.load(f)

    for section in fields(Config):
        if section.name not in data or section.name == "stats_interval_s":
            continue
        values = data[section.name]
        if not isinstance(values, dict):
            raise ValueError(f"[{section.name}] must be a TOML table")
        allowed = {option.name for option in fields(section.type)}
        for key in values.keys() - allowed:
            logging.getLogger(__name__).warning("Ignoring unknown config key %s.%s", section.name, key)
            del values[key]
    for key in data.keys() - {section.name for section in fields(Config)}:
        logging.getLogger(__name__).warning("Ignoring unknown config section %s", key)

    return Config(
        safe_playback=SafePlaybackConfig(**data.get("safe_playback", {})),
        diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
        language_filter=LanguageFilterConfig(**data.get("language_filter", {})),
        gpu_budget=GpuBudgetConfig(**data.get("gpu_budget", {})),
        ingest=IngestConfig(**data.get("ingest", {})),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**data.get("mpv", {})),
        holodex=HolodexConfig(**data.get("holodex", {})),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
