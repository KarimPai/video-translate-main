"""Whisper Large-v3 Japanese ASR -> PLaMo 3 Translate 2B -> MPV subtitles."""

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
                        help="print Japanese detection and English translation only")
    parser.add_argument("--mpv", action="store_true")
    parser.add_argument("--delay", type=float, metavar="SECONDS")
    parser.add_argument(
        "--audio-profile", choices=("baseline", "gtcrn", "speaker", "both"),
        default=None, help="ablate GTCRN and Sherpa without changing config.toml",
    )
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

    from .config import load_config
    from .hololive_names import select_hololive_hints
    from .glossary import Glossary
    from .translate import create_translator
    from .whisper_v3 import WhisperLargeV3
    from .ingest import Ingest, resolve_source
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
    from .session_cache import SessionSubtitleCache

    cfg = load_config(args.config)
    if args.audio_profile is not None:
        cfg.gtcrn.enabled = args.audio_profile in ("gtcrn", "both")
        cfg.speaker.enabled = args.audio_profile in ("speaker", "both")
    if args.diagnostics:
        cfg.whisper.diagnostics = True
    if args.delay is not None:
        cfg.mpv.delay_s = args.delay

    log = logging.getLogger("auto_translate")
    log.info(
        "Audio processing: GTCRN=%s (CPU), Sherpa speaker segmentation=%s (CPU); "
        "FireRedVAD (CUDA), Whisper Large-v3 Japanese ASR -> PLaMo 3 2B",
        "on" if cfg.gtcrn.enabled else "off",
        "on" if cfg.speaker.enabled else "off",
    )
    source = resolve_source(
        args.source, args.mode, args.start, args.fast, cfg.ingest,
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
        "Whisper Large-v3 Japanese ASR ready in %.1fs",
        time.perf_counter() - started,
    )
    translator = None
    try:
        glossary = Glossary.load(Path(cfg.translate.glossary))
        timeline = source.timeline or {}
        title_context = " - ".join(
            str(timeline.get(field) or "") for field in ("title", "channel")
            if timeline.get(field)
        )[:240]
        translator = create_translator(
            cfg.translate, glossary, vod_title=title_context,
        )
        log.info(
            "PLaMo 3 Translate 2B ready (Q4_K_M, context=%d, glossary=%d terms)",
            cfg.translate.plamo_context, len(glossary.terms),
        )
        use_mpv = args.mpv or cfg.mpv.enabled
        if use_mpv and not source.live and source.paced:
            source = replace(source, paced=False)
            log.info("MPV VOD: fast audio ingestion enabled for seek buffering")

        ingest = Ingest(
            source,
            cfg.ingest,
            None,  # MPV plays the original video; FFmpeg ingests audio only
            cfg.mpv.ytdlp_format,
            0.0,
            separation_cfg=cfg.separation,
            gtcrn_cfg=cfg.gtcrn,
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

        run(ingest, cfg, whisper, outputs, stop,
            session_cache=cache, translator=translator)
    finally:
        if translator is not None:
            translator.close()
        whisper.close()


if __name__ == "__main__":
    main()
