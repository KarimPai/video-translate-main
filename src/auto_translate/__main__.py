"""Caption-only Japanese YouTube VOD → PLaMo 8B → MPV.

This experimental branch deliberately never loads an audio recognizer,
speech VAD, enhancement model, diarizer, or FFmpeg ingest worker.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from . import cuda_dlls


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto-translate")
    parser.add_argument("source", help="YouTube VOD URL with Japanese captions")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--mode", choices=["auto", "live", "vod"], default="auto")
    parser.add_argument("--start", metavar="TIME")
    parser.add_argument("--fast", action="store_true", help="retained for CLI compatibility")
    parser.add_argument("--no-web", action="store_true", help="retained for CLI compatibility")
    parser.add_argument("--simple-terminal", action="store_true", help="show JP and EN lines")
    parser.add_argument("--mpv", action="store_true", help="play YouTube source in MPV")
    parser.add_argument("--delay", type=float, metavar="SECONDS",
                        help="ignored in captions-only VOD mode; timestamps are native")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_dlls.setup()

    from .caption_only import CaptionOnlyPlayer
    from .captions import download_vod_captions
    from .config import load_config
    from .glossary import Glossary
    from .ingest import resolve_source
    from .translate import create_translator

    log = logging.getLogger("auto_translate")
    cfg = load_config(args.config)
    source = resolve_source(
        args.source, args.mode, args.start, args.fast, cfg.ingest,
    )
    if source.live:
        raise SystemExit(
            "Captions-only branch currently supports archived VODs only. "
            "Reliable continuously-updated live captions and their MPV "
            "clock alignment need separate validation. No silent ASR fallback."
        )
    if not source.timeline or not source.timeline.get("video_id"):
        raise SystemExit("A YouTube VOD with Japanese captions is required.")

    video_id = str(source.timeline["video_id"])
    index = download_vod_captions(video_id, source.location, cfg.ingest)
    if index is None or not index.cues:
        raise SystemExit(
            "No downloadable Japanese captions for this video. "
            "The captions-only branch cannot translate without them."
        )

    print(
        f"Caption source: {index.source_format.upper()} "
        f"({index.source_origin}), {len(index.cues)} raw cues",
        flush=True,
    )

    title = " - ".join(
        part for part in (
            source.timeline.get("title", ""),
            source.timeline.get("channel", ""),
        ) if part
    )
    started = time.perf_counter()
    glossary = Glossary.load(Path(cfg.translate.glossary))
    translator = create_translator(cfg.translate, glossary, vod_title=title)
    log.info(
        "PLaMo 3 Translate 8B ready (%d glossary entries) in %.1fs",
        len(glossary.terms),
        time.perf_counter() - started,
    )
    try:
        player = CaptionOnlyPlayer(
            source, cfg, index, translator, vod_title=title,
        )
        player.run()
    finally:
        translator.close()


if __name__ == "__main__":
    main()
