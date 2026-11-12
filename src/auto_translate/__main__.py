
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
    parser.add_argument("--video-quality", type=int, choices=(360, 480, 720, 1080),
                        help="YouTube maximum playback height; lower-quality fallback")
    parser.add_argument("--hwdec", choices=("no", "auto-safe"),
                        help="MPV hardware decode mode")
    parser.add_argument("--rtx-vsr", action="store_true",
                        help="request NVIDIA RTX VSR for <=480p source video; enable High in NVIDIA App")
    parser.add_argument("--diagnostics", action="store_true",
                        help="log additional pipeline metrics")
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
    from .windows_job import bind_process_tree
    bind_process_tree()
    cuda_dlls.setup()

    from .asr import create_asr
    from .captions import PendingCaptions, download_vod_captions
    from .config import load_config, mpv_youtube_format
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
    if args.hwdec is not None:
        cfg.mpv.hwdec = args.hwdec
    if args.rtx_vsr:
        cfg.mpv.rtx_vsr = True

    log = logging.getLogger("auto_translate")
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

    # Fetch VOD captions in parallel with model loading. Captions are optional:
    # any download/parse failure leaves the Haas-only pipeline unchanged.
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
                log.exception("caption preparation failed; continuing with Haas")

        caption_thread = threading.Thread(
            target=load_captions,
            name="youtube-captions",
            daemon=True,
        )
        caption_thread.start()

    # Video metadata is useful for proper names and scene terms on live AND
    # VOD. It is context, not proof of who spoke the current utterance.
    vod_context = " - ".join(
        part for part in (
            (source.timeline or {}).get("title", ""),
            (source.timeline or {}).get("channel", ""),
        ) if part
    )

    # Load models before starting ingest.
    from .gpu_budget import GpuBudget
    from .asr.dual import DualASR
    from dataclasses import replace
    budget = GpuBudget(cfg.gpu_budget)
    primary_cfg = cfg.asr if budget.admits(1.35) else replace(cfg.asr, haas_device="cpu")
    log.info("Primary Haas device=%s; GPU target %.1f decimal GB",
             primary_cfg.haas_device, budget.cfg.target_gb)
    t0 = time.perf_counter()
    try:
        asr = create_asr(primary_cfg)
    except (RuntimeError, OSError):
        if primary_cfg.haas_device.lower() == "cpu":
            raise
        log.warning("Haas Vulkan startup failed; retrying on CPU", exc_info=True)
        asr = create_asr(replace(primary_cfg, haas_device="cpu"))

    asr_model = cfg.asr.haas_model
    log.info(
        "ASR ready (%s, Haas) in %.1fs",
        asr_model,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    glossary = Glossary.load(Path(cfg.translate.glossary))
    try:
        # Keep room for FireRedVAD, MPV and the independent Vulkan encoder.
        reserve = 5.35 if cfg.dual_asr.enabled else 2.8
        if not budget.admits(reserve):
            partial_reserve = 3.9 if cfg.dual_asr.enabled else 1.8
            limit = 12 if budget.admits(partial_reserve) else 0
            cfg.translate.plamo_gpu_layers = min(cfg.translate.plamo_gpu_layers, limit)
        log.info("PLaMo GPU layer limit=%s", cfg.translate.plamo_gpu_layers)
        try:
            translator = create_translator(cfg.translate, glossary, vod_title=vod_context)
        except (RuntimeError, OSError):
            if not cfg.translate.plamo_gpu_layers:
                raise
            log.warning("PLaMo GPU startup failed; retrying on CPU", exc_info=True)
            cfg.translate.plamo_gpu_layers = 0
            translator = create_translator(cfg.translate, glossary, vod_title=vod_context)
    except BaseException:
        close = getattr(asr, "close", None)
        if callable(close):
            close()
        raise
    log.info(
        "PLaMo 3 Translate 2B ready (%d glossary terms) in %.1fs",
        len(glossary.terms),
        time.perf_counter() - t0,
    )

    use_mpv = args.mpv or cfg.mpv.enabled
    if use_mpv and not source.live and source.paced:
        # A 1x (-re) FFmpeg reader cannot rebuild the 5-second translation
        # buffer promptly after seeking. MPV itself keeps real-time playback.
        source = replace(source, paced=False)
        log.info("MPV VOD: fast audio ingestion enabled for seek rebuffering")

    # Only livestreams share a local HLS playlist between MPV and FFmpeg
    # speech recognition. VOD keeps direct MPV playback and fast audio seek.
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

    try:
        dual = DualASR(cfg, budget, glossary, source.live)
    except BaseException:
        asr.close()
        translator.close()
        raise

    if use_mpv:
        from .output.mpv import MpvOutput
        try:
            player = MpvOutput(cfg.mpv, ingest, stop, subtitle_cache=subtitle_cache)
            outputs.append(player)
            dual.playback_position = player.playback_position
        except BaseException:
            dual.release()
            ingest.stop()
            asr.close()
            translator.close()
            raise

    def relieve_pressure():
        dual.release()
        if budget.used is not None and budget.used >= budget.cfg.maximum_gb:
            used = budget.refresh()
            if used is not None and used >= budget.cfg.maximum_gb:
                log.error("Device memory still over maximum after optional model release; stopping")
                stop.set()

    print(f"ASR: Haas + {'Qwen3-ASR' if dual.worker else 'no secondary'} | "
          f"Selector: {'Qwen3.5' if dual.selector else 'off'} | PLaMo: {cfg.translate.plamo_gpu_layers} GPU layers requested", flush=True)
    budget.start(relieve_pressure)
    try:
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
            dual=dual,
        )
    finally:
        budget.close()
        dual.release()



if __name__ == "__main__":
    main()
