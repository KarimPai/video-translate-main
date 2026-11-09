
"""CLI entry point: uv run auto-translate <youtube-url | audio file> [options]."""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from . import cuda_dlls


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto-translate")
    parser.add_argument("source", help="YouTube URL (live or VOD) or a local audio/video file")
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
    parser.add_argument("--delay", type=float, metavar="SECONDS",
                        help="mpv: how far playback stays behind the pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for noisy in ("httpx",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_dlls.setup()

    from .asr import create_asr
    from .captions import PendingCaptions, download_vod_captions
    from .config import load_config
    from .glossary import Glossary
    from .ingest import Ingest, resolve_source
    from .output.base import Output
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
    from .session_cache import SessionSubtitleCache
    from .translate import create_translator

    cfg = load_config(args.config)

    if args.delay is not None:
        cfg.mpv.delay_s = args.delay

    log = logging.getLogger("auto_translate")
    source = resolve_source(
        args.source,
        args.mode,
        args.start,
        args.fast,
        cfg.ingest,
    )

    # Fetch VOD captions in parallel with model loading. Captions are optional:
    # download/parse failures leave Haas ASR as the sole recognizer.
    captions = None
    video_id = (
        source.timeline.get("video_id")
        if not source.live and source.timeline
        else None
    )
    if video_id:
        captions = PendingCaptions()

        def load_captions() -> None:
            try:
                captions.index = download_vod_captions(
                    str(video_id), source.location, cfg.ingest
                )
            except Exception:
                log.exception("caption preparation failed; continuing with Haas ASR")

        caption_thread = threading.Thread(
            target=load_captions,
            name="youtube-captions",
            daemon=True,
        )
        caption_thread.start()

    # Include the channel for proper-name disambiguation on streams whose
    # titles omit the talent's name (e.g. many Watame gaming VODs).
    vod_context = " - ".join(
        part for part in (
            (source.timeline or {}).get("title", ""),
            (source.timeline or {}).get("channel", ""),
        ) if part
    )

    # Load models before starting ingest.
    t0 = time.perf_counter()
    asr = create_asr(cfg.asr)

    log.info(
        "ASR ready (Haas-ja-v1.0 Q8_0, %s speech.cpp) in %.1fs",
        cfg.asr.haas_device,
        time.perf_counter() - t0,
    )

    # If the translation model cannot start, close the already loaded
    # speech.cpp Haas ASR worker rather than leaving GPU memory occupied.
    translator = None
    try:
        t0 = time.perf_counter()
        glossary = Glossary.load(Path(cfg.translate.glossary))
        translator = create_translator(
            cfg.translate, glossary, vod_title=vod_context,
        )
        log.info(
            "PLaMo 3 Translate 8B Q3_K_M ready (%d glossary terms) in %.1fs",
            len(glossary.terms),
            time.perf_counter() - t0,
        )
        use_mpv = args.mpv or cfg.mpv.enabled
        if use_mpv and not source.live and source.paced:
            # A 1x (-re) FFmpeg reader cannot rebuild the 5-second translation
            # buffer promptly after seeking. MPV itself keeps real-time playback.
            from dataclasses import replace
            source = replace(source, paced=False)
            log.info("MPV VOD: fast audio ingestion enabled for seek rebuffering")
    
        # MPV plays the original video stream; Ingest handles audio only.
        # No local video transcode or fMP4 cache is created.
        video_out = None
        burst_s = 0.0
    
        ingest = Ingest(
            source,
            cfg.ingest,
            video_out,
            cfg.mpv.ytdlp_format,
            burst_s,
            separation_cfg=cfg.separation,
            gtcrn_cfg=cfg.gtcrn,
        )
    
        stop = threading.Event()
        outputs: list[Output] = []
        # Cache only this MPV VOD in RAM. Never reuse it for another video,
        # and never persist translated text on disk.
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
    
        if use_mpv:
            from .output.mpv import MpvOutput
            outputs.append(MpvOutput(cfg.mpv, ingest, stop, subtitle_cache=subtitle_cache))
    
        # Caption downloads normally finish during ASR/PLaMo warm-up.
        # Give a still-pending VOD caption fetch a small bounded chance to
        # finish so the FIRST utterance can use YouTube-first selection too.
        # Never block live playback or hold startup for the 90s yt-dlp timeout.
        if video_id and cfg.asr.youtube_captions_first and caption_thread.is_alive():
            caption_thread.join(timeout=2.0)
            if caption_thread.is_alive():
                log.info("Japanese captions still loading; Haas runs until available")

        run(
            ingest,
            cfg,
            asr,
            translator,
            outputs,
            stop,
            captions=captions,
            vod_title=vod_context,
            session_cache=subtitle_cache,
        )
    finally:
        # Free both owned model processes on MPV close, Ctrl+C, EOF,
        # or a model/playback startup exception.
        try:
            if translator is not None:
                translator.close()
        finally:
            asr.close()


if __name__ == "__main__":
    main()
