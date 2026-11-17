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
    threshold: float = 0.30
    neg_threshold: float = 0.20
    min_silence_ms: int = 550  # wait for genuine phrase-ending silence
    speech_pad_ms: int = 400  # preserve soft/overlapped word starts and tails
    min_speech_ms: int = 160
    short_reply_ms: int = 120
    short_reply_probability: float = 0.85
    onset_ms: int = 30
    prob_smooth_frames: int = 5
    max_utterance_s: float = 14.0  # VOD soft cap: more sentence context
    max_split_extension_s: float = 2.0  # VOD hard cap: 16s
    live_max_utterance_s: float = 7.0  # fits around 9s live buffer
    live_max_split_extension_s: float = 1.0  # live hard cap: 8s
    # Natural-pause splitting begins before the long-form hard cap.
    adaptive_split: bool = True
    adaptive_split_s: float = 6.5  # VOD: keep context unless a clear pause occurs
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
            raise ValueError("FireRedVAD must use cpu to preserve GPU memory for Voxtral and playback")


@dataclass
class AsrConfig:
    """Persistent CUDA Voxtral service; complete VAD utterances use offline ASR.

    The model is loaded once by audio.cpp. Its decoder runs greedy ASR with a
    fixed seed; word timing comes from the original VAD utterance boundaries.
    """

    voxtral_model_path: str = "models/voxtral/voxtral-mini-4b-realtime-2602-q4_k.gguf"
    voxtral_server_path: str = ".runtime/audio-v0.9.5-cuda12/audiocpp_server.exe"
    cuda_device: int = 0
    threads: int = 8
    blocklist: list[str] = field(default_factory=list)
    response_timeout_s: float = 45.0
    live_response_timeout_s: float = 20.0
    startup_timeout_s: float = 180.0
    # Retry a rejected utterance once with radio enhancement and decoder padding.
    rescue_retry: bool = True
    rescue_min_vad_confidence: float = 0.45
    rescue_min_duration_s: float = 0.4

    def __post_init__(self):
        for name in ("voxtral_model_path", "voxtral_server_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"asr.{name} must be a nonempty path")
        if type(self.rescue_retry) is not bool:
            raise ValueError("asr.rescue_retry must be boolean")
        if type(self.rescue_min_vad_confidence) not in (int, float) or not 0.0 <= self.rescue_min_vad_confidence <= 1.0:
            raise ValueError("asr.rescue_min_vad_confidence must be between 0 and 1")
        if type(self.rescue_min_duration_s) not in (int, float) or not 0.0 <= self.rescue_min_duration_s <= 10.0:
            raise ValueError("asr.rescue_min_duration_s must be 0..10 s")
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
    """Drop only confidently non-Japanese speech before Japanese ASR.

    Ambiguous or very short utterances stay in the original Japanese path to
    avoid losing Japanese speech during VTuber code-switching.
    """
    enabled: bool = True
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
    """PLaMo 3 Translate 2B settings (sole translation engine)."""

    plamo_base_url: str = "http://127.0.0.1:8000/v1"
    plamo_model: str = "plamo-3-translate"
    plamo_hf_model: str = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    plamo_context: int = 4096  # smaller KV/context footprint on RTX 3070
    plamo_gpu_layers: int = 99
    max_tokens: int = 192  # prevent truncation of longer, complete JP utterances
    timeout_s: float = 30.0  # more time for full translation, not a delay target
    load_timeout_s: float = 300.0
    context_lines: int = 3  # OverlaySubs-style bounded dialogue exchange
    # Prefer reliable recent bilingual pairs to speculative long-term scene memory.
    memory_lines: int = 16
    extended_memory: bool = True
    context_profile: str = "compact"  # compact/legacy; overlay and hybrid are benchmark alternatives
    sentence_context: bool = True  # translate clearly separated completed sentences sequentially
    glossary: str = "glossary.toml"
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    suppress_runaway_loops: bool = True
    max_identical_cues: int = 2
    neutral_subjects: bool = True  # prefer unspecified/neutral speakers, no inferred gender
    retry_suspicious: bool = True  # only one additional PLaMo request for clear mismatches
    retry_timeout_s: float = 15.0  # upper bound for the optional second pass
    fragment_context: bool = True  # Japanese-only previous cue for ambiguous fragments

    def __post_init__(self):
        if type(self.extended_memory) is not bool:
            raise ValueError("extended_memory must be boolean")
        if self.context_profile not in ("compact", "legacy", "overlay", "hybrid"):
            raise ValueError("translate.context_profile must be compact, legacy, overlay or hybrid")
        for name in ("context_lines", "memory_lines"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 20:
                raise ValueError(f"translate.{name} must be an integer in 0..20")
        for value in (self.timeout_s, self.load_timeout_s, self.retry_timeout_s):
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
    en_size: int = 28
    ja_size: int = 0
    min_subtitle_duration_s: float = 1.2
    max_subtitle_rows: int = 3

    def __post_init__(self):
        for value in (self.delay_s, self.start_wait_s, self.live_late_window_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("MPV timing values must be finite and nonnegative")
        if not 0 <= self.min_subtitle_duration_s <= 10:
            raise ValueError("MPV minimum subtitle duration must be between 0 and 10 seconds")
        if self.live_forward_cache_mib <= 0 or self.live_back_cache_mib < 0:
            raise ValueError("Invalid MPV live cache size")


@dataclass
class ChatConfig:
    enabled: bool = False
    window_s: float = 30.0
    max_messages: int = 100
    poll_interval_s: float = 5.0
    live_time_uncertainty_s: float = 5.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("chat.enabled must be boolean")
        if type(self.max_messages) is not int or not 1 <= self.max_messages <= 500:
            raise ValueError("chat.max_messages must be in 1..500")
        if not math.isfinite(self.window_s) or not 1 <= self.window_s <= 120:
            raise ValueError("chat.window_s must be in 1..120")
        if not math.isfinite(self.poll_interval_s) or not 1 <= self.poll_interval_s <= 60:
            raise ValueError("chat.poll_interval_s must be in 1..60")
        if not math.isfinite(self.live_time_uncertainty_s) or not 0 <= self.live_time_uncertainty_s <= 120:
            raise ValueError("chat.live_time_uncertainty_s must be in 0..120")


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
    chat: ChatConfig = field(default_factory=ChatConfig)
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
        chat=ChatConfig(**data.get("chat", {})),
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
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
