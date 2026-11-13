"""Load config.toml into typed dataclasses (missing keys fall back to defaults)."""

import tomllib
from dataclasses import dataclass, field, replace
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
    adaptive_split_s: float = 9.0  # VOD: keep context unless a clear pause occurs
    live_adaptive_split_s: float = 5.5  # live: leave room for ASR + translation
    adaptive_pause_ms: int = 180  # avoid cutting at very brief VAD dips
    split_pause_min_ms: int = 120
    split_context_ms: int = 160
    fallback_s: float = 0.0  # do not ASR-decode VAD-negative music/game audio
    fallback_min_s: float = 1.5
    fallback_min_db: float = -50.0

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("VAD device must be cpu or cuda")


@dataclass
class AsrConfig:
    haas_model: str = "haas-ja-v1.0"
    haas_device: str = "Vulkan0"
    haas_decode_on_cpu: bool = False
    blocklist: list[str] = field(default_factory=list)
    response_timeout_s: float = 24.0  # per-utterance VOD recognition watchdog
    live_response_timeout_s: float = 12.0  # prevent stalled Vulkan decode from delaying live cues
    startup_timeout_s: float = 45.0  # model cold starts may take longer than decode
    retry_empty_speech: bool = True  # VOD only; no music-only speculative decoding
    retry_min_confidence: float = 0.80
    retry_min_duration_s: float = 0.6
    retry_max_duration_s: float = 10.0


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
    language_routing: bool = False  # Parakeet JA is Japanese-only
    secondary_live_mode: str = "selective"
    secondary_vod_mode: str = "all"
    live_timeout_s: float = 2.0
    vod_timeout_s: float = 25.0

    def __post_init__(self):
        if type(self.language_routing) is not bool:
            raise ValueError("language routing flags must be boolean")
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
    context_lines: int = 4  # shorter trusted dialogue history; avoid EN error carryover
    # Only Japanese scene history and verified glossary terms belong in memory.
    memory_lines: int = 8  # bounded scene context; no speculative EN term hints
    glossary: str = "glossary.toml"
    remove_fillers: bool = True
    collapse_repetitions: bool = True
    suppress_runaway_loops: bool = True
    max_identical_cues: int = 2
    neutral_subjects: bool = True  # prefer unspecified/neutral speakers, no inferred gender
    retry_suspicious: bool = True  # only one additional PLaMo request for clear mismatches
    retry_timeout_s: float = 15.0  # upper bound for the optional second pass
    fragment_context: bool = True  # Japanese-only previous cue for ambiguous fragments


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
    rtx_vsr: bool = False  # optional RTX 480p NVIDIA enhancement (Windows/D3D11)
    live_late_window_s: float = 7.0  # reject extremely stale spoken cues
    ytdlp_format: str = (
        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
        "/b[height<=1080][protocol^=m3u8]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    en_size: int = 28
    ja_size: int = 0
    max_subtitle_rows: int = 3


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

    # Old config.toml files may contain hard VRAM thresholds; ignore them.
    # The current policy uses real NVIDIA total/free memory, with no forced stop.
    gpu_budget_data = dict(data.get("gpu_budget", {}))
    for obsolete in ("target_gb", "optimization_threshold_gb",
                     "emergency_threshold_gb", "maximum_gb"):
        gpu_budget_data.pop(obsolete, None)

    return Config(
        language_filter=LanguageFilterConfig(**data.get("language_filter", {})),
        dual_asr=DualAsrConfig(**{**{k: v for k, v in data.get("dual_asr", {}).items() if not k.startswith("indonesian_")}, "language_routing": False}),
        reconciler=ReconcilerConfig(**data.get("reconciler", {})),
        gpu_budget=GpuBudgetConfig(**gpu_budget_data),
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
