
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
    decode_timeout = (min(cfg.asr.response_timeout_s, cfg.asr.live_response_timeout_s)
                      if source.live else cfg.asr.response_timeout_s)
    timed_cfg = replace(cfg.asr, response_timeout_s=decode_timeout)
    primary_cfg = timed_cfg if budget.admits(1.35) else replace(timed_cfg, haas_device="cpu")
    log.info("Haas per-cue timeout=%.1fs (startup timeout=%.1fs)",
             decode_timeout, timed_cfg.startup_timeout_s)
    log.info("Primary Haas device=%s; GPU capacity=%s GB; free=%s GB; no fixed software ceiling",
             primary_cfg.haas_device, budget.total, budget.free)
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
        # Preserve room for two ASRs, the reconciler, and MPV/VSR on 8 GiB GPUs.
        # A single 12-layer PLaMo server has proven fast enough; avoid 99-layer
        # offload that may prevent the secondary model from loading.
        if cfg.dual_asr.enabled and (budget.total is None or budget.total < 10.0):
            cfg.translate.plamo_gpu_layers = min(cfg.translate.plamo_gpu_layers, 12)
            log.info("Shared 8 GiB-class GPU: reserve PLaMo offload at 12 layers for dual ASR")
        elif not budget.admits(2.8):
            cfg.translate.plamo_gpu_layers = 0
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
        # Only real free-memory pressure causes degradation. No fixed 7.5 GB
        # cutoff, and never terminate the primary Haas/PLaMo/MPV pipeline.
        level = budget.pressure_level
        if level == "critical":
            log.warning("Near physical GPU capacity (free=%s GB): releasing optional models", budget.free)
            dual.release()
        elif level == "low":
            log.warning("Low physical GPU headroom (free=%s GB): releasing GPU selector", budget.free)
            dual.release_selector()

    # MPV/VSR allocate GPU surfaces AFTER optional model startup. Recheck
    # after the player is ready rather than waiting for the monitor's tick.
    budget.refresh()
    if budget.pressure_level in ("low", "critical"):
        relieve_pressure()

    selector_status = "off"
    if dual.selector:
        placement = f"{dual.selector.gpu_layers} GPU layers requested" if dual.selector.gpu_layers else "CPU"
        selector_status = f"LFM2.5-1.2B-JP Q4_K_M ({placement})"
    language_status = ("auto JA/EN (English passthrough)" if
                       cfg.dual_asr.language_routing and dual.worker else "Japanese only")
    print(f"ASR: Haas + {'Nemotron 3.5' if dual.worker else 'no secondary'} | "
          f"Language: {language_status} | Selector: {selector_status} | "
          f"PLaMo: {cfg.translate.plamo_gpu_layers} GPU layers requested", flush=True)
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
