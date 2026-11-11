"""Whisper Large-v3 Japanese speech -> English direct AST -> MPV subtitles."""

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import cuda_dlls


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto-translate")
    parser.add_argument("source", help="YouTube live/VOD URL or local video/audio file")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--mode", choices=("auto", "live", "vod"), default="auto")
    parser.add_argument("--start", metavar="TIME", help="VOD start position")
    parser.add_argument("--fast", action="store_true", help="unpaced VOD audio ingestion")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--simple-terminal", action="store_true",
                        help="print direct English translation only")
    parser.add_argument("--mpv", action="store_true")
    parser.add_argument("--delay", type=float, metavar="SECONDS")
    parser.add_argument("--video-quality", type=int, choices=(360, 480, 720, 1080),
                        metavar="{360,480,720,1080}",
                        help="maximum YouTube playback height; selects a lower quality if unavailable")
    parser.add_argument("--whisper-compute", choices=("float16", "int8_float16"),
                        help="Whisper CUDA precision; FP16 OOM falls back to int8_float16")
    parser.add_argument("--hwdec", choices=("no", "auto-safe"),
                        help="MPV video decoder (no saves GPU memory)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="log each utterance's decoding and audio quality stats")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_dlls.setup()

    from .config import load_config, mpv_youtube_format
    from .hololive_names import select_hololive_hints
    from .whisper_v3 import WhisperLargeV3
    from .ingest import Ingest, resolve_source
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
    from .session_cache import SessionSubtitleCache

    cfg = load_config(args.config)
    if args.video_quality is not None:
        cfg.mpv.ytdlp_format = mpv_youtube_format(args.video_quality, live=True)
        cfg.mpv.vod_ytdlp_format = mpv_youtube_format(args.video_quality, live=False)
    if args.whisper_compute is not None:
        cfg.whisper.compute_type = args.whisper_compute
    if args.diagnostics:
        cfg.whisper.diagnostics = True
    if args.delay is not None:
        cfg.mpv.delay_s = args.delay
    if args.hwdec is not None:
        cfg.mpv.hwdec = args.hwdec
    log = logging.getLogger("auto_translate")
    source = resolve_source(
        args.source, args.mode, args.start, args.fast, cfg.ingest,
    )
    if args.video_quality is not None and not Path(args.source).exists():
        log.info(
            "YouTube playback quality: maximum %dp (lower-quality fallback; %s)",
            args.video_quality,
            "live HLS" if source.live else "VOD HTTPS/DASH first",
        )

    cfg.whisper.hololive_detected_names = select_hololive_hints(
        source.timeline,
        enabled=cfg.whisper.hololive_names_enabled,
        limit=cfg.whisper.hololive_name_limit,
    )
    log.info(
        "Hololive name hints: %s (from title/channel metadata)",
        ", ".join(cfg.whisper.hololive_detected_names) or "none",
    )

    started = time.perf_counter()
    whisper = WhisperLargeV3(cfg.whisper)
    log.info(
        "Whisper Large-v3 direct AST ready in %.1fs",
        time.perf_counter() - started,
    )
    try:
        use_mpv = args.mpv or cfg.mpv.enabled
        if use_mpv and not source.live and source.paced:
            source = replace(source, paced=False)
            log.info("MPV VOD: fast audio ingestion enabled for seek buffering")

        shared_live_hls = bool(use_mpv and source.live)
        if shared_live_hls:
            log.info("MPV live: synchronized shared local HLS audio/video enabled")
        ingest = Ingest(
            source,
            cfg.ingest,
            "fmp4" if shared_live_hls else None,  # VOD keeps original MPV video
            cfg.mpv.ytdlp_format,
            0.0,
        )
        stop = threading.Event()
        outputs = []
        cache = SessionSubtitleCache() if use_mpv and not source.live else None

        if cfg.output.terminal:
            outputs.append(TerminalOutput(simple=args.simple_terminal))
        if cfg.output.web and not args.no_web:
            outputs.append(WebOutput(cfg.output.web_host, cfg.output.web_port))
        if use_mpv:
            from .output.mpv import MpvOutput
            outputs.append(MpvOutput(cfg.mpv, ingest, stop, subtitle_cache=cache))

        run(ingest, cfg, whisper, outputs, stop, session_cache=cache)
    finally:
        whisper.close()


if __name__ == "__main__":
    main()
