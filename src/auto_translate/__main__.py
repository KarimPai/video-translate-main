"""CLI entry point: uv run video-translate <youtube-url | audio file> [options]."""

import argparse
import logging
import math
import os
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from . import cuda_dlls


def main() -> None:
    with ExitStack() as resources:
        from .runtime_diagnostics import record_event
        try:
            _main(resources)
        except BaseException as exc:
            record_event("application_shutdown", expected=isinstance(exc, (KeyboardInterrupt, SystemExit)),
                         error=type(exc).__name__)
            raise
        else:
            record_event("application_shutdown", expected=True)


def _main(resources: ExitStack) -> None:
    parser = argparse.ArgumentParser(prog="video-translate")
    parser.add_argument("source", nargs="?", default=os.getenv("VIDEO_URL"),
                        help="YouTube URL (live or VOD) or a local audio/video file")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--mode", choices=["auto", "live", "vod"], default="auto",
                        help="auto (default) asks yt-dlp whether the URL is live right now")
    parser.add_argument("--start", metavar="TIME",
                        help="vod: start here (962, 16:02, 1:02:03, 16m2s); default: the URL's t= value")
    parser.add_argument("--fast", action="store_true",
                        help="vod: process as fast as possible instead of at 1x speed")
    parser.add_argument("--no-web", action="store_true", help="don't start the browser viewer")
    parser.add_argument("--simple-terminal", action="store_true",
                        help="print only JP/EN transcript lines (no timing columns)")
    parser.add_argument("--mpv", action="store_true",
                        help="watch in mpv, delayed so subtitles appear exactly when each line is spoken")
    parser.add_argument("--shared-gpu-playback", action="store_true",
                        help="limit VOD inference read-ahead during shared-GPU playback")
    option = parser.add_mutually_exclusive_group()
    option.add_argument("--safe-playback", dest="safe_playback", action="store_true", default=None,
                        help="coordinate inference and bound audio read-ahead")
    option.add_argument("--no-safe-playback", dest="safe_playback", action="store_false")
    parser.add_argument("--context-lines", type=int, choices=range(21), metavar="0..20",
                        help="previous dialogue lines given to the translator")
    parser.add_argument("--asr-model", type=Path, help="ASR model path")
    parser.add_argument("--speech-server", type=Path, help="speech.cpp executable path")
    parser.add_argument("--mt-model", type=Path, help="Translation GGUF model")
    parser.add_argument("--llama-server", type=Path, help="llama.cpp llama-server executable (CUDA)")
    parser.add_argument("--delay", type=float, metavar="SECONDS",
                        help="mpv: how far playback stays behind the pipeline")
    parser.add_argument("--video-quality", type=int, choices=(360, 480, 720, 1080),
                        help="YouTube maximum playback height; lower-quality fallback")
    parser.add_argument("--hwdec", choices=("no", "auto-safe"),
                        help="MPV hardware decode mode")
    parser.add_argument("--gui-control", action="store_true",
                        help="accept validated Qt player controls over loopback")
    parser.add_argument("--gui-control-port", type=int)
    parser.add_argument("--gui-control-token")
    parser.add_argument("--mpv-wid", type=int, metavar="HWND",
                        help="embed MPV into the supplied Windows native widget handle")
    diagnostics_option = parser.add_mutually_exclusive_group()
    diagnostics_option.add_argument("--diagnostics", action="store_true", default=None,
                                    help="record bounded application diagnostics")
    diagnostics_option.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    parser.add_argument("--list-backends", action="store_true",
                        help="List available ASR and translation backends")
    parser.add_argument("--asr-backend", type=str,
                        help="ASR backend name (e.g. haas, qwen3-asr)")
    parser.add_argument("--translate-backend", type=str,
                        help="Translation backend name (e.g. gemma4, plamo)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Setup logging early
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    # Show available backends (before config load)
    if args.list_backends:
        from .registry import get_default_registry
        registry = get_default_registry()
        backends = registry.list_backends()
        print("\nAvailable backends:")
        for btype, names in backends.items():
            print(f"  {btype}: {', '.join(names) if names else '(none)'}")
        for btype, names in backends.items():
            for name in names:
                caps = registry.get_capabilities(btype, name)
                if caps:
                    print(f"    {name}: {caps.description} (runtime={caps.runtime}, device={caps.device})")
        print()
        return

    from urllib.parse import urlsplit
    source = args.source or ""
    try:
        parts = urlsplit(source)
        remote = parts.scheme in ("http", "https") and parts.hostname and '"' not in source
        valid = remote or Path(source).is_file()
    except (ValueError, OSError):
        valid = False
    if not valid:
        parser.error("Enter an HTTP(S) URL or an existing media file")

    handlers = [logging.StreamHandler()]
    if os.getenv("VIDEO_TRANSLATE_CLI_LOG"):
        console = handlers[0]
        console.setLevel(logging.WARNING)
        from logging.handlers import RotatingFileHandler
        Path("logs").mkdir(exist_ok=True)
        handlers.append(RotatingFileHandler("logs/run.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"))
        def report_failure(kind, error, traceback):
            logging.getLogger("auto_translate").critical("Translator stopped", exc_info=(kind, error, traceback))
        sys.excepthook = report_failure
    logging.basicConfig(
        handlers=handlers,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for noisy in ("httpx",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from .windows_job import bind_process_tree
    bind_process_tree()
    stop = threading.Event()
    control = None
    from .windows_subprocess import check_running, guard_gui_spawns
    if args.gui_control:
        if not (args.gui_control_token and args.gui_control_port and 0 < args.gui_control_port <= 65535):
            parser.error("Missing GUI control connection")
        from .gui_control import BackendControl
        guard_gui_spawns()
        control = BackendControl(args.gui_control_port, args.gui_control_token, stop)
        resources.callback(control.close)
    cuda_dlls.setup()

    from .config import load_config, mpv_youtube_format
    from .glossary import Glossary
    from .ingest import Ingest, resolve_source
    from .output.base import Output
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
    from .session_cache import SessionSubtitleCache

    cfg = load_config(args.config)

    # Collect explicit CLI overrides for backend config precedence.
    # These must override both common config AND backend-specific profiles.
    asr_cli_overrides = {}
    translate_cli_overrides = {}
    for option, section, field_name, override_dict in (
        ("asr_model", cfg.asr, "model_path", asr_cli_overrides),
        ("speech_server", cfg.asr, "speech_server_path", asr_cli_overrides),
        ("mt_model", cfg.translate, "model_path", translate_cli_overrides),
        ("llama_server", cfg.translate, "server_path", translate_cli_overrides),
        ("context_lines", cfg.translate, "context_lines", translate_cli_overrides),
    ):
        value = getattr(args, option)
        if value is not None:
            str_value = str(value) if isinstance(value, Path) else value
            setattr(section, field_name, str_value)
            override_dict[field_name] = str_value
    if args.safe_playback is not None:
        cfg.safe_playback.enabled = args.safe_playback
    if args.diagnostics is not None:
        cfg.diagnostics.enabled = args.diagnostics
    if args.asr_backend is not None:
        cfg.models.asr_backend = args.asr_backend
    if args.translate_backend is not None:
        cfg.models.translate_backend = args.translate_backend
    from .runtime_diagnostics import configure_diagnostics, record_event
    diagnostics = configure_diagnostics(cfg.diagnostics)
    resources.callback(diagnostics.close)
    def record_cleanup():
        from .windows_subprocess import owned_process_status
        record_event("subprocess_cleanup", processes=owned_process_status())
    resources.callback(record_cleanup)
    record_event("application_start", safe_playback=cfg.safe_playback.enabled,
                 context_lines=cfg.translate.context_lines)
    from .safe_playback import SafeInferenceGate
    gate = SafeInferenceGate(cfg.safe_playback, stop)
    if os.getenv("VIDEO_TRANSLATE_CLI_LOG"):
        import shutil

        from .output.mpv import find_mpv
        for tool in ("ffmpeg", "yt-dlp", "deno"):
            if not shutil.which(tool):
                parser.error(f"{tool} missing from PATH. Install via scripts/Setup-Windows.ps1")
        if args.mpv or cfg.mpv.enabled:
            find_mpv(cfg.mpv.path)

    if args.delay is not None:
        cfg.mpv.delay_s = args.delay
    if args.hwdec is not None:
        cfg.mpv.hwdec = args.hwdec
    if args.gui_control and not (args.gui_control_token and args.gui_control_port
                                 and 0 < args.gui_control_port <= 65535):
        parser.error("GUI control socket missing. Update the GUI launcher and backend together.")
    if args.gui_control and not args.mpv_wid:
        parser.error("--gui-control requires --mpv-wid")
    if args.mpv_wid is not None:
        if sys.platform != "win32" or args.mpv_wid <= 0:
            parser.error("--mpv-wid requires a positive Windows HWND")
        cfg.mpv.window_id = args.mpv_wid

    log = logging.getLogger("auto_translate")
    if cfg.safe_playback.enabled:
        lead = cfg.ingest.vod_max_audio_lead_s
        cfg.ingest.vod_max_audio_lead_s = min(lead, cfg.safe_playback.max_audio_lead_s) if lead else cfg.safe_playback.max_audio_lead_s
        log.info("Safe Playback: at most %d simultaneous inference calls; audio lead %.1fs; MPV decoder settings preserved",
                 cfg.safe_playback.max_inference_concurrency, cfg.ingest.vod_max_audio_lead_s)
    if args.shared_gpu_playback:
        cfg.ingest.vod_max_audio_lead_s = min(cfg.ingest.vod_max_audio_lead_s, 30.0)
        log.info("Shared GPU playback: VOD audio lead capped at %.1fs",
                 cfg.ingest.vod_max_audio_lead_s)
    check_running()
    source = resolve_source(
        args.source,
        args.mode,
        args.start,
        args.fast,
        cfg.ingest,
    )

    if args.video_quality is not None and not Path(args.source).exists():
        cfg.mpv.ytdlp_format = mpv_youtube_format(
            args.video_quality, live=source.live,
        )
        log.info(
            "MPV YouTube quality: up to %dp, lower fallback (%s)",
            args.video_quality,
            "live HLS" if source.live else "VOD HTTPS/DASH first",
        )

    # Video metadata biases ASR spellings and gives translation scene context.
    from .config import vad_for_source

    # Load models before starting ingest: ASR first (fixed size), then size
    # translation offload from measured free memory.
    from .gpu_budget import GpuBudget
    from .stream_context import describe_stream
    from .vad import Segmenter
    check_running()
    segmenter = Segmenter(vad_for_source(cfg.vad, live=bool(source.live)))
    budget = GpuBudget(cfg.gpu_budget)
    resources.callback(budget.close)
    budget.operation = gate.operation
    glossary = Glossary.load(Path(cfg.translate.glossary))
    metadata = source.timeline
    holodex_key = cfg.holodex.api_key or os.getenv("HOLODEX_API_KEY", "")
    if holodex_key and metadata and metadata.get("video_id"):
        from .holodex import fetch_cast
        cast = fetch_cast(metadata["video_id"], holodex_key, cfg.holodex.timeout_s)
        if cast:
            log.info("Holodex: %d tagged collab name(s)", len(cast["mentions"]))
            metadata = dict(metadata, holodex_channel=cast["channel"], holodex_mentions=cast["mentions"])
    meta = metadata or {}
    glossary = glossary.for_stream("\n".join((str(meta.get("title") or ""),
                                              " ".join(map(str, meta.get("tags") or [])))))
    if glossary.active_topics:
        log.info("Glossary topics: %s", ", ".join(glossary.active_topics))
    stream_context = describe_stream(metadata, glossary)
    if stream_context.host or stream_context.cast:
        log.info("Stream context: host=%s cast=%s", stream_context.host or "-",
                 "、".join(stream_context.cast) or "-")
    t0 = time.perf_counter()
    check_running()

    # Create ASR backend via registry with backend-specific profile first,
    # then derive timeout from the effective merged profile.
    from .config import get_backend_config, resolve_model_identity
    from .registry import get_default_registry
    registry = get_default_registry()
    asr_backend_name = cfg.models.asr_backend
    asr_profile = get_backend_config(cfg, "asr", asr_backend_name,
                                     explicit_overrides=asr_cli_overrides if asr_cli_overrides else None)
    # Derive timeout from the already-merged profile (backend-specific values
    # take precedence over common config). For live mode, use the shorter
    # live_response_timeout_s; for VOD, use response_timeout_s.
    response_timeout = asr_profile.get("response_timeout_s", cfg.asr.response_timeout_s)
    live_timeout = asr_profile.get("live_response_timeout_s", cfg.asr.live_response_timeout_s)
    decode_timeout = min(response_timeout, live_timeout) if source.live else response_timeout
    asr_profile["response_timeout_s"] = decode_timeout
    # For Whisper, determine if model_path was explicitly configured
    asr_identity = resolve_model_identity(cfg, "asr", asr_backend_name,
                                          explicit_overrides=asr_cli_overrides if asr_cli_overrides else None)
    whisper_kwargs = {}
    if asr_backend_name == "whisper" and asr_identity.get("model_path"):
        whisper_kwargs["_whisper_is_explicit"] = True
    asr = registry.create_backend("asr", asr_backend_name, asr_profile,
                                  context=stream_context.recognition_text(),
                                  **whisper_kwargs)
    resources.callback(asr.close)
    budget.refresh()
    record_event("asr_loaded", backend=asr_backend_name, gpu=budget.snapshot())
    log.info("ASR %s ready in %.1fs (effective timeout %.1fs); measured GPU free=%s GB",
             asr_backend_name, time.perf_counter()-t0, decode_timeout, budget.free)

    t0 = time.perf_counter()
    check_running()
    translator = _load_translator(cfg, glossary, stream_context, budget, log, registry,
                                  translate_cli_overrides if translate_cli_overrides else None)
    record_event("mt_loaded", backend=cfg.models.translate_backend, gpu=budget.snapshot(),
                 gpu_layers=getattr(translator, "gpu_layers", None))
    translator_layers = getattr(translator, "gpu_layers", None)
    translator_total = getattr(translator, "layer_count", None)
    layers_str = f"{translator_layers}/{translator_total}" if translator_layers is not None else "N/A"
    log.info("Translation %s ready at %s GPU layers (%d glossary terms) in %.1fs; GPU free=%s GB",
             cfg.models.translate_backend, layers_str, len(glossary.terms),
             time.perf_counter()-t0, budget.free)
    resources.callback(translator.close)
    use_mpv = args.mpv or cfg.mpv.enabled
    if use_mpv and not source.live and source.paced:
        from dataclasses import replace as dc_replace
        source = dc_replace(source, paced=False)
        log.info("MPV VOD: fast audio ingestion enabled for seek rebuffering")

    shared_live_hls = bool(use_mpv and source.live)
    if shared_live_hls:
        log.info("MPV live: synchronized shared local HLS audio/video enabled")
    ingest = Ingest(
        source,
        cfg.ingest,
        video_out="fmp4" if shared_live_hls else None,
        video_format=cfg.mpv.ytdlp_format if shared_live_hls else "",
        separation_cfg=cfg.separation,
        deepfilter_cfg=cfg.deepfilter,
    )
    ingest.inference_gate = gate

    outputs: list[Output] = []
    subtitle_cache = SessionSubtitleCache() if use_mpv and not source.live else None

    if cfg.output.terminal:
        outputs.append(TerminalOutput(simple=args.simple_terminal))

    if cfg.output.web and not args.no_web:
        outputs.append(
            WebOutput(
                cfg.output.web_host,
                cfg.output.web_port,
            )
        )

    # Optional, CPU-only spoken-language gate.
    language_filter = None
    if cfg.language_filter.enabled:
        try:
            from .text_filter import LanguageFilter
            language_filter = LanguageFilter(cfg.language_filter)
        except ImportError:
            log.warning(
                "Spoken-language filter module not available; continuing without "
                "English/Indonesian filtering. Install a language detection backend "
                "to enable this feature."
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Spoken-language filter unavailable; continuing without "
                "English/Indonesian filtering: %s", exc,
            )

    check_running()
    playback_position = None
    if use_mpv:
        from .output.mpv import MpvOutput
        try:
            player = MpvOutput(cfg.mpv, ingest, stop, subtitle_cache=subtitle_cache)
            resources.enter_context(player)
            outputs.append(player)
            playback_position = player.playback_position
            if control:
                control.player = player
        except BaseException:
            ingest.stop()
            raise

    def relieve_pressure():
        request = getattr(translator, "request_gpu_relief", None)
        if callable(request):
            request()
        record_event("gpu_headroom_relief_requested", gpu=budget.snapshot())

    budget.refresh()
    if budget.pressure_level in ("low", "critical"):
        relieve_pressure()
    asr_name = asr_backend_name
    print(f"ASR: {asr_name} | "
          f"Japanese only (CPU language filter: {'on' if language_filter else 'off'}) | "
          f"MT: {cfg.models.translate_backend}, {layers_str} GPU layers", flush=True)
    # Apply effective backend profile values to shared config so the pipeline
    # sees backend-specific settings (context_lines, merge_fragments, timeouts, etc.)
    # instead of only common defaults. This preserves the precedence:
    # explicit CLI override > backend profile > common config.
    from .config import get_backend_config
    translate_profile = get_backend_config(cfg, "translate", cfg.models.translate_backend,
                                           explicit_overrides=translate_cli_overrides)
    for key in ("context_lines", "merge_fragments", "merge_max_gap_s", "merge_max_s",
                "live_merge_margin_s", "max_identical_cues", "timeout_s"):
        if key in translate_profile:
            setattr(cfg.translate, key, translate_profile[key])

    asr_profile = get_backend_config(cfg, "asr", asr_backend_name,
                                     explicit_overrides=asr_cli_overrides)
    for key in ("response_timeout_s", "live_response_timeout_s"):
        if key in asr_profile:
            setattr(cfg.asr, key, asr_profile[key])

    check_running()
    budget.start(relieve_pressure)
    try:
        run(
            ingest,
            cfg,
            asr,
            translator,
            outputs,
            stop,
            segmenter=segmenter,
            session_cache=subtitle_cache,
            playback_position=playback_position,
            language_filter=language_filter,
            gate=gate,
            diagnostics=diagnostics if cfg.diagnostics.enabled else None,
        )
    finally:
        budget.close()


def _load_translator(cfg, glossary, stream_context, budget, log, registry,
                     translate_cli_overrides=None):
    """Start translation backend, adapting GPU layers for llama.cpp-based models."""
    from .config import get_backend_config
    from .llama_server import LlamaServerError, gguf_block_count, require_gguf

    translate_backend_name = cfg.models.translate_backend
    translate_profile = get_backend_config(cfg, "translate", translate_backend_name,
                                           explicit_overrides=translate_cli_overrides)

    headroom = cfg.gpu_budget.startup_headroom_gb
    auto = translate_profile.get("gpu_layers", -1) < 0
    layers = None if auto else translate_profile.get("gpu_layers", -1)
    if auto and budget.free is not None:
        layers = 0
        if budget.free > headroom:
            model = require_gguf(translate_profile.get("model_path", cfg.translate.model_path), "Translation model")
            layer_count = gguf_block_count(model) or 32
            per_layer_gb = os.path.getsize(model) / (layer_count + 2) / 1e9
            available = budget.free - headroom - .25
            layers = max(0, min(layer_count, math.floor(available / per_layer_gb)))
    while True:
        try:
            translator = registry.create_backend("translate", translate_backend_name,
                                                 translate_profile, glossary=glossary, stream=stream_context,
                                                 gpu_layers=layers)
            break
        except LlamaServerError:
            if layers == 0 or not auto:
                raise
            layers = 16 if layers is None else layers // 2
            log.warning("Translation startup failed; retrying with %d GPU layers", layers, exc_info=True)
    try:
        if not auto:
            budget.refresh()
            return translator
        translator_layers = getattr(translator, "gpu_layers", None)
        translator_total = getattr(translator, "layer_count", None)
        if translator_layers is None or translator_total is None:
            return translator
        model_path = getattr(translator, "model", None)
        if model_path is None:
            return translator
        per_layer_gb = os.path.getsize(model_path) / (translator_total + 2) / 1e9
        for _ in range(3):
            budget.refresh()
            if budget.free is None or budget.free >= headroom or translator.gpu_layers == 0:
                break
            trim = math.ceil((headroom - budget.free) / per_layer_gb) + 1
            reload_fn = getattr(translator, "reload", None)
            if callable(reload_fn):
                translator.reload(translator.gpu_layers - trim,
                                  f"GPU free {budget.free:.2f} GB below {headroom:.2f} GB headroom")
        return translator
    except BaseException:
        translator.close()
        raise


if __name__ == "__main__":
    main()
