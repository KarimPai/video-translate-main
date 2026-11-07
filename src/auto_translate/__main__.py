
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
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for noisy in ("httpx",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_dlls.setup()

    from .asr import create_asr
    from .captions import download_vod_captions
    from .config import load_config
    from .glossary import Glossary
    from .ingest import Ingest, resolve_source
    from .output.base import Output
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
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
    # any download/parse failure leaves the Haas-only pipeline unchanged.
    caption_result = {}
    caption_thread = None
    video_id = (
        source.timeline.get("video_id")
        if not source.live and source.timeline
        else None
    )
    if video_id:
        def load_captions() -> None:
            try:
                caption_result["index"] = download_vod_captions(
                    str(video_id), source.location, cfg.ingest
                )
            except Exception:
                log.exception("caption preparation failed; continuing with Haas")

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
    ) if not source.live else ""

    # Load models before starting ingest.
    t0 = time.perf_counter()
    asr = create_asr(cfg.asr)

    asr_model = cfg.asr.haas_model
    log.info(
        "ASR ready (%s, Haas) in %.1fs",
        asr_model,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    glossary = Glossary.load(Path(cfg.translate.glossary))
    translator = create_translator(
        cfg.translate, glossary, vod_title=vod_context,
    )
    translator_name = (
        "PLaMo 3 Translate"
        if cfg.translate.backend.strip().lower() == "plamo"
        else "HY-MT2"
    )
    log.info(
        "%s translator ready (%d glossary terms) in %.1fs",
        translator_name,
        len(glossary.terms),
        time.perf_counter() - t0,
    )

    captions = None
    if caption_thread is not None:
        caption_thread.join()
        captions = caption_result.get("index")
        if captions is None:
            log.info("caption assistance unavailable; using Haas transcription only")

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
        deepfilter_cfg=cfg.deepfilter,
    )

    stop = threading.Event()
    outputs: list[Output] = []

    if cfg.output.terminal:
        outputs.append(TerminalOutput())

    if cfg.output.web and not args.no_web:
        outputs.append(
            WebOutput(
                cfg.output.web_host,
                cfg.output.web_port,
            )
        )

    if use_mpv:
        from .output.mpv import MpvOutput
        outputs.append(MpvOutput(cfg.mpv, ingest, stop))

    diarizer = None

    if cfg.mpv.speaker_diarization:
        from .speaker import PyannoteDiarizer

        t0 = time.perf_counter()

        try:
            diarizer = PyannoteDiarizer(
                device=cfg.mpv.speaker_device,
                max_speakers=cfg.mpv.max_speakers,
            )
        except Exception as exc:
            raise SystemExit(
                f"Could not initialize pyannote diarization: {exc}"
            ) from exc

        log.info(
            "pyannote diarization loaded on %s in %.1fs",
            cfg.mpv.speaker_device,
            time.perf_counter() - t0,
        )

    run(
        ingest,
        cfg,
        asr,
        translator,
        outputs,
        stop,
        diarizer=diarizer,
        captions=captions,
        vod_title=vod_context,
    )


if __name__ == "__main__":
    main()
