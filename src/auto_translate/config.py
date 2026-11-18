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
    threshold: float = 0.34  # modest quiet-speech recall improvement; 0.3 is very permissive
    neg_threshold: float = 0.25
    min_silence_ms: int = 550  # wait for genuine phrase-ending silence
    speech_pad_ms: int = 400  # preserve soft/overlapped word starts and tails
    min_speech_ms: int = 200  # keep short game-sfx rejection; strong short replies still survive
    short_reply_ms: int = 120
    short_reply_probability: float = 0.85
    onset_ms: int = 50  # recover quieter and short initial speech
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
        if self.device not in ("cpu", "cuda"):
            raise ValueError("VAD device must be cpu or cuda")


@dataclass
class AsrConfig:
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    # Opt in only after benchmarking reviewed audio on the installed model.
    haas_keywords_enabled: bool = False
    haas_keywords_alpha: float = 1.0
    blocklist: list[str] = field(default_factory=list)
    response_timeout_s: float = 24.0  # per-utterance VOD recognition watchdog
    live_response_timeout_s: float = 12.0  # prevent stalled Vulkan decode from delaying live cues
    startup_timeout_s: float = 120.0  # model cold starts may take longer than decode
    retry_empty_speech: bool = True  # VOD only; no music-only speculative decoding
    retry_min_confidence: float = 0.80
    retry_min_duration_s: float = 0.6
    retry_max_duration_s: float = 10.0

    def __post_init__(self):
        if type(self.haas_keywords_enabled) is not bool:
            raise ValueError("haas_keywords_enabled must be boolean")
        if (type(self.haas_keywords_alpha) not in (int, float)
                or not math.isfinite(self.haas_keywords_alpha)
                or not 0 < self.haas_keywords_alpha <= 2):
            raise ValueError("haas_keywords_alpha must be finite in (0,2]")
        if not isinstance(self.blocklist, list) or any(not isinstance(x, str) or not x.strip() for x in self.blocklist):
            raise ValueError("asr.blocklist requires nonempty strings")
        if not 0 <= self.retry_min_confidence <= 1 or not 0 < self.retry_min_duration_s <= self.retry_max_duration_s:
            raise ValueError("Invalid ASR retry thresholds")
        if min(self.startup_timeout_s, self.response_timeout_s, self.live_response_timeout_s) <= 0:
            raise ValueError("ASR timeouts must be positive")


@dataclass
class LanguageFilterConfig:
    """Drop only confidently English/Indonesian speech before Japanese ASR.

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
class DualAsrConfig:
    enabled: bool = True
    smart_correction: bool = True
    correction_live_retries: int = 0
    correction_vod_retries: int = 1
    correction_live_budget_s: float = 2.0
    correction_vod_budget_s: float = 8.0
    secondary_model: str = "h2oai/faster-whisper-large-v3-turbo"
    secondary_device: str = "cuda"
    secondary_compute_type: str = "int8_float16"
    secondary_beam_size: int = 6
    secondary_patience: float = 1.0
    secondary_length_penalty: float = 1.0
    secondary_temperature: float = 0.0
    secondary_best_of: int = 5
    secondary_repetition_penalty: float = 1.0
    secondary_no_repeat_ngram_size: int = 0
    secondary_word_timestamps: bool = False
    secondary_log_prob_threshold: float = -1.0
    secondary_no_speech_threshold: float = 0.6
    secondary_compression_ratio_threshold: float = 2.4
    secondary_suppress_tokens: list[int] = field(default_factory=lambda: [-1])
    secondary_max_new_tokens: int | None = None
    # Only an audio-grounded per-utterance hint may be supplied by the caller.
    secondary_prompt_mode: str = "none"  # none, hotwords, initial_prompt
    japanese_outro_mode: str = "log"  # log or suppress weak uncorroborated candidates
    whisper_name_hints: bool = False  # deprecated/ignored: no unconditional Whisper name prompts
    secondary_live_mode: str = "selective"
    secondary_vod_mode: str = "all"
    live_timeout_s: float = 5.0
    vod_timeout_s: float = 25.0

    def __post_init__(self):
        if type(self.smart_correction) is not bool:
            raise ValueError("smart_correction must be boolean")
        for retries in (self.correction_live_retries, self.correction_vod_retries):
            if type(retries) is not int or not 0 <= retries <= 2:
                raise ValueError("correction retries must be integers in 0..2")
        for seconds in (self.correction_live_budget_s, self.correction_vod_budget_s):
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 30:
                raise ValueError("correction budgets must be finite in (0,30] seconds")
        if self.japanese_outro_mode not in ("log", "suppress"):
            raise ValueError("japanese_outro_mode must be log or suppress")
        if not self.secondary_model.strip():
            raise ValueError("secondary model identifier must be nonempty")
        if self.secondary_device not in ("cuda", "cpu"):
            raise ValueError("secondary_device must be cuda or cpu")
        if self.secondary_compute_type not in ("int8_float16", "float16", "int8", "float32"):
            raise ValueError("invalid Whisper Turbo compute_type")
        if self.secondary_device == "cpu" and self.secondary_compute_type in ("int8_float16", "float16"):
            raise ValueError("CPU Whisper Turbo requires int8 or float32 compute_type")
        if type(self.secondary_beam_size) is not int or not 1 <= self.secondary_beam_size <= 10:
            raise ValueError("secondary_beam_size must be in 1..10")
        for name in ("secondary_patience", "secondary_repetition_penalty",
                     "secondary_compression_ratio_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10:
                raise ValueError(name + " must be finite in (0,10]")
        for name, low, high in (("secondary_temperature", 0, 1),
                                ("secondary_length_penalty", 0, 10),
                                ("secondary_log_prob_threshold", -20, 0),
                                ("secondary_no_speech_threshold", 0, 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(name + " is out of range")
        for name, low, high in (("secondary_best_of", 1, 10),
                                ("secondary_no_repeat_ngram_size", 0, 20)):
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(name + " is out of range")
        if type(self.secondary_word_timestamps) is not bool:
            raise ValueError("secondary_word_timestamps must be boolean")
        if (not isinstance(self.secondary_suppress_tokens, list)
                or len(self.secondary_suppress_tokens) > 512
                or any(type(token) is not int or not -1 <= token < 60000
                       for token in self.secondary_suppress_tokens)):
            raise ValueError("secondary_suppress_tokens must contain Whisper token IDs")
        if (self.secondary_max_new_tokens is not None
                and (type(self.secondary_max_new_tokens) is not int
                     or not 1 <= self.secondary_max_new_tokens <= 448)):
            raise ValueError("secondary_max_new_tokens must be in 1..448 or omitted")
        if self.secondary_prompt_mode not in ("none", "hotwords", "initial_prompt"):
            raise ValueError("secondary_prompt_mode must be none, hotwords or initial_prompt")
        for mode in (self.secondary_live_mode, self.secondary_vod_mode):
            if mode not in ("off", "selective", "all"):
                raise ValueError("secondary mode must be off, selective or all")
        validate_timing(self)


def validate_timing(cfg):
    if type(cfg.enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    for timeout in (cfg.live_timeout_s, cfg.vod_timeout_s):
        if type(timeout) not in (float, int) or not 0 < timeout <= 120:
            raise ValueError("request timeout must be in (0, 120] seconds")


@dataclass
class ReconcilerConfig:
    enabled: bool = True
    gpu_layers: int = 99
    live_timeout_s: float = 4.0
    vod_timeout_s: float = 20.0

    def __post_init__(self):
        validate_timing(self)
        if type(self.gpu_layers) is not int or not 0 <= self.gpu_layers <= 99:
            raise ValueError("gpu_layers must be an integer between 0 and 99")


@dataclass
class GpuBudgetConfig:
    """GPU capacity-relative headroom; no fixed application VRAM limit."""

    startup_headroom_gb: float = 0.65
    low_free_gb: float = 0.55
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
    context_max_gap_s: float = 15.0  # reset dialogue only after actual media-timeline silence
    # Prefer reliable recent bilingual pairs to speculative long-term scene memory.
    memory_lines: int = 16
    extended_memory: bool = True
    context_profile: str = "compact"  # compact (accuracy-first) or legacy
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
        for name in ("context_lines", "memory_lines"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 20:
                raise ValueError(f"translate.{name} must be an integer in 0..20")
        if (type(self.context_max_gap_s) not in (int, float)
                or not math.isfinite(self.context_max_gap_s) or self.context_max_gap_s <= 0):
            raise ValueError("translate.context_max_gap_s must be finite and positive")
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
    rtx_vsr: bool = False  # optional NVIDIA RTX enhancement (Windows/D3D11)
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
    dual_asr: DualAsrConfig = field(default_factory=DualAsrConfig)
    reconciler: ReconcilerConfig = field(default_factory=ReconcilerConfig)
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

    # Ignore legacy fields so existing config.toml files remain compatible.
    ingest_data = dict(data.get("ingest", {}))
    # Older configs may contain a -3 semitone ASR pitch workaround. Natural pitch
    # is the default for Haas Japanese recognition; ignore that legacy override.
    ingest_data.pop("pitch_semitones", None)

    return Config(
        chat=ChatConfig(**data.get("chat", {})),
        safe_playback=SafePlaybackConfig(**data.get("safe_playback", {})),
        diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
        language_filter=LanguageFilterConfig(**data.get("language_filter", {})),
        dual_asr=DualAsrConfig(**data.get("dual_asr", {})),
        reconciler=ReconcilerConfig(**data.get("reconciler", {})),
        gpu_budget=GpuBudgetConfig(**data.get("gpu_budget", {})),
        ingest=IngestConfig(**ingest_data),
        separation=SeparationConfig(**data.get("separation", {})),
        deepfilter=DeepFilterConfig(**data.get("deepfilter", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**data.get("mpv", {})),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
