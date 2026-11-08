"""Sherpa-ONNX per-window speaker-turn segmentation on CPU.

Only turn boundaries are used. No session-wide voice identity tracking,
speaker labels, or source separation. Unsafe cuts leave the original
FireRedVAD utterance untouched rather than invent a speaker boundary.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import SpeakerConfig
from .types import AudioProgress, SAMPLE_RATE, Utterance

log = logging.getLogger(__name__)
SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/nemo_en_titanet_small.onnx")
SEG_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"

# Official maintainer-hosted standalone weights: avoid GitHub's tarball,
# which sometimes responds with HTTP 504. GitHub releases are fallback only.
SEG_HF_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/"
    "resolve/main/model.int8.onnx"
)
EMB_HF_URL = (
    "https://huggingface.co/csukuangfj/speaker-embedding-models/"
    "resolve/main/nemo_en_titanet_small.onnx"
)
SEG_SHA256 = "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d"
EMB_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int | None


def _model_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) / "auto-translate" / "speaker-models") if local else (
        Path.home() / ".cache" / "auto-translate" / "speaker-models")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str,
    target: Path,
    member: str = "",
    *,
    fallback_url: str = "",
    expected_sha256: str = "",
) -> None:
    """Download validated ONNX weights, retrying and switching official hosts.

    Hugging Face serves the standalone ONNX files, avoiding a large GitHub
    release archive. The original GitHub releases remain as fallback. All
    writes are atomic and validated against maintainer-published SHA256.
    """
    if target.is_file():
        if target.stat().st_size > 1024 and (
            not expected_sha256 or _sha256(target) == expected_sha256
        ):
            return
        log.warning("Invalid or partial speaker model cache: %s; downloading again", target)
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Speaker diarization: downloading {target.name} from Hugging Face...", flush=True)
    sources = [(url, "")]
    if fallback_url:
        sources.append((fallback_url, member))
    errors: list[str] = []
    for source_url, archive_member in sources:
        host = "Hugging Face" if "huggingface.co/" in source_url else "GitHub releases"
        for attempt in range(1, 3):
            temporary: list[Path] = []
            try:
                log.info("Sherpa model download %s attempt %d/2: %s",
                         host, attempt, source_url)
                request = urllib.request.Request(
                    source_url,
                    headers={"User-Agent": "auto-translate-sherpa/1.0"},
                )
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    downloaded = Path(output.name)
                    temporary.append(downloaded)
                    with urllib.request.urlopen(request, timeout=45) as response:
                        size = 0
                        next_report = 4 * 1024 * 1024
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > 150 * 1024 * 1024:
                                raise ValueError("Speaker model download exceeded 150 MB")
                            output.write(chunk)
                            if size >= next_report:
                                print(
                                    f"Speaker diarization: {target.name} "
                                    f"{size / (1024 * 1024):.0f} MiB downloaded...",
                                    flush=True,
                                )
                                next_report += 4 * 1024 * 1024
                candidate = downloaded
                if archive_member:
                    with tarfile.open(downloaded, "r:bz2") as archive:
                        info = archive.getmember(archive_member)
                        if not info.isfile() or info.size > 25 * 1024 * 1024:
                            raise ValueError("Unsafe Sherpa model archive member")
                        source = archive.extractfile(info)
                        if source is None:
                            raise ValueError("Speaker model missing from archive")
                        with tempfile.NamedTemporaryFile(
                            dir=target.parent, delete=False
                        ) as output:
                            candidate = Path(output.name)
                            temporary.append(candidate)
                            shutil.copyfileobj(source, output)
                if candidate.stat().st_size <= 1024:
                    raise ValueError("Speaker model download was empty")
                if expected_sha256 and _sha256(candidate) != expected_sha256:
                    raise ValueError("Speaker model SHA256 verification failed")
                candidate.replace(target)
                log.info("Speaker model downloaded and verified: %s", target)
                print(f"Speaker diarization: {target.name} ready.", flush=True)
                return
            except (OSError, ValueError, tarfile.TarError) as exc:
                errors.append(f"{host} attempt {attempt}: {exc}")
                log.warning("Sherpa model download failed (%s attempt %d/2): %s",
                            host, attempt, exc)
                if attempt == 1:
                    time.sleep(1.0)
            finally:
                for path in temporary:
                    path.unlink(missing_ok=True)
        if source_url != sources[-1][0]:
            print("Speaker diarization: switching to GitHub fallback...", flush=True)
    raise RuntimeError(
        f"Unable to download {target.name} from official sources: "
        + "; ".join(errors[-4:])
    )


def _quiet_cut(
    audio: np.ndarray | None,
    center: float,
    left: float,
    right: float,
    length: float,
    *,
    search_ms: int = 180,
    quiet_ms: int = 40,
    quiet_ratio: float = 0.45,
) -> tuple[float, bool]:
    """Move an estimated speaker turn only into a sustained quiet valley.

    A one-frame dip is not sufficient evidence to relocate a turn boundary.
    Return the model midpoint unchanged when evidence is ambiguous; mark it untrusted.
    """
    if audio is None or left >= right:
        return center, False
    radius = max(0.0, search_ms / 1000)
    lo = max(0.0, left, center - radius)
    hi = min(length, right, center + radius)
    step = max(1, round(0.01 * SAMPLE_RATE))
    half = max(1, round(0.008 * SAMPLE_RATE))
    samples = np.arange(round(lo * SAMPLE_RATE), round(hi * SAMPLE_RATE) + 1, step)
    if len(samples) < 4:
        return center, False
    rms = np.asarray([
        float(np.sqrt(np.mean(audio[max(0, int(k) - half):
                                     min(len(audio), int(k) + half)] ** 2)))
        for k in samples
    ])
    baseline = float(np.percentile(rms, 75))
    if baseline <= 1e-5:
        return center, False
    quiet = rms <= max(1e-5, baseline * quiet_ratio)
    required = max(2, int(math.ceil(quiet_ms / 10)))
    runs: list[tuple[int, int]] = []
    start = None
    for i, silent in enumerate(quiet):
        if silent and start is None:
            start = i
        if (not silent or i == len(quiet) - 1) and start is not None:
            end = i + (1 if silent and i == len(quiet) - 1 else 0)
            if end - start >= required:
                runs.append((start, end))
            start = None
    if not runs:
        return center, False
    closest = min(
        runs,
        key=lambda run: abs(samples[(run[0] + run[1] - 1) // 2] / SAMPLE_RATE - center),
    )
    middle = (closest[0] + closest[1] - 1) // 2
    return float(samples[middle] / SAMPLE_RATE), True


def split_turns(
    length: float,
    raw: list[Turn],
    min_seconds: float,
    audio: np.ndarray | None = None,
    *,
    boundary_search_ms: int = 180,
    boundary_quiet_ms: int = 40,
    boundary_quiet_ratio: float = 0.45,
) -> list[Turn]:
    """Cut independent speaker turns, including clear interruptions.

    A confirmed takeover truncates the first speaker at the new onset.
    Uncertain simultaneous voices remain mixed. No PCM is dropped or
    duplicated and each output is independently transcribed.
    """
    unknown = [Turn(0.0, length, None)]
    if length <= 0 or not raw:
        return unknown
    merged: list[Turn] = []
    for turn in sorted(raw, key=lambda t: t.start):
        if turn.speaker is None or turn.end <= turn.start:
            continue
        start = min(max(0.0, turn.start), length)
        end = min(max(0.0, turn.end), length)
        if end <= start:
            continue
        if (merged and merged[-1].speaker == turn.speaker
                and start <= merged[-1].end + 0.2):
            prev = merged[-1]
            merged[-1] = Turn(prev.start, max(prev.end, end), prev.speaker)
        else:
            merged.append(Turn(start, end, turn.speaker))
    if not merged:
        return unknown
    if len(merged) == 1:
        return [Turn(0.0, length, merged[0].speaker)]

    # A later speaker who starts while the current one is still talking
    # and clearly continues beyond that speaker's endpoint has taken the
    # floor. Cut the first at the *new onset*, rather than sending both
    # voices to Haas in one utterance. This is not source separation: the
    # handoff's later audio may still contain background speech.
    #
    # Short overlaps (<120ms), same-onset voices, and interventions that
    # end before the original speaker remain ambiguous/mixed.
    groups: list[Turn] = []
    for turn in merged:
        if groups and turn.start < groups[-1].end - 0.12:
            prev = groups[-1]
            takeover = (
                prev.speaker is not None
                and turn.speaker is not None
                and turn.speaker != prev.speaker
                and turn.start - prev.start >= min_seconds
                and turn.end - turn.start >= min_seconds
                and turn.end >= prev.end + 0.25
            )
            if takeover:
                # Preserve the first speaker's interrupted fragment exactly;
                # all subsequent samples belong to the next ASR request.
                groups[-1] = Turn(prev.start, turn.start, prev.speaker)
                groups.append(Turn(turn.start, turn.end, turn.speaker))
                log.debug(
                    "Sherpa detected speaker interruption at %.2fs (previous "
                    "turn expected through %.2fs)", turn.start, prev.end,
                )
            else:
                groups[-1] = Turn(
                    prev.start, max(prev.end, turn.end), None,
                )
        else:
            groups.append(turn)
    if len(groups) < 2:
        return unknown

    cuts: list[float] = []
    labels = [groups[0].speaker]
    for prev, following in zip(groups, groups[1:]):
        if prev.speaker is not None and prev.speaker == following.speaker:
            continue
        center = max(0.0, min(length, (prev.end + following.start) / 2))
        cut, trusted = _quiet_cut(
            audio, center, min(prev.end, following.start),
            max(prev.end, following.start), length,
            search_ms=boundary_search_ms,
            quiet_ms=boundary_quiet_ms,
            quiet_ratio=boundary_quiet_ratio,
        )
        if audio is not None and following.start < prev.end - 0.015:
            # Small speaker overlaps have no reliable cut point unless
            # a sustained acoustic valley independently confirms the turn.
            if not trusted:
                return unknown
        if cut - (cuts[-1] if cuts else 0.0) < min_seconds:
            # An unreliable tiny prefix invalidates the proposed cuts.
            return unknown
        cuts.append(cut)
        labels.append(following.speaker)
    if not cuts or length - cuts[-1] < min_seconds:
        return unknown
    edges = [0.0, *cuts, length]
    return [Turn(edges[i], edges[i + 1], labels[i]) for i in range(len(labels))]


class SherpaSplitter:
    def __init__(self, cfg: SpeakerConfig) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("Install sherpa-onnx using: uv sync --extra diarization") from exc
        self.cfg = cfg
        root = Path(cfg.model_dir).expanduser() if cfg.model_dir else _model_root()
        seg = root / "pyannote-segmentation3-int8.onnx"
        emb = root / "nemo_en_titanet_small.onnx"
        _download(SEG_HF_URL, seg, SEG_MEMBER,
                  fallback_url=SEG_URL, expected_sha256=SEG_SHA256)
        _download(EMB_HF_URL, emb,
                  fallback_url=EMB_URL, expected_sha256=EMB_SHA256)
        opts = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg), window_shift_ratio=0.20),
                num_threads=cfg.num_threads, provider="cpu"),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb), num_threads=cfg.num_threads, provider="cpu"),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=-1, threshold=0.50),
            min_duration_on=0.30, min_duration_off=0.10)
        if not opts.validate():
            raise RuntimeError("Sherpa diarization model configuration invalid")
        self.diarizer = sherpa_onnx.OfflineSpeakerDiarization(opts)
        self.failures = 0
        print("Speaker diarization: Sherpa CPU ready.", flush=True)

    def split(self, original: Utterance) -> list[Utterance]:
        if original.probe:
            return [original]
        samples = len(original.audio)
        if samples < round(1.65 * SAMPLE_RATE):
            # Very short cues cannot contain two robust turn segments; do
            # not hallucinate speaker boundaries in a split-second response.
            return [original]
        try:
            audio = np.ascontiguousarray(original.audio, dtype=np.float32)
            length = samples / SAMPLE_RATE

            # Sherpa's offline window benefits from at least three seconds
            # of observation. Symmetric silence padding lets 2-3s segments
            # participate without inventing speech or shifting real timestamps.
            if length < 3.0:
                left_pad = (3 * SAMPLE_RATE - samples) // 2
                right_pad = 3 * SAMPLE_RATE - samples - left_pad
                model_audio = np.pad(audio, (left_pad, right_pad))
                offset = left_pad / SAMPLE_RATE
            else:
                model_audio = audio
                offset = 0.0

            raw = self.diarizer.process(model_audio).sort_by_start_time()
            turns: list[Turn] = []
            for turn in raw:
                start = float(turn.start) - offset
                end = float(turn.end) - offset
                if not (math.isfinite(start) and math.isfinite(end)):
                    continue
                start = max(0.0, start)
                end = min(length, end)
                if end > start:
                    turns.append(Turn(start, end, int(turn.speaker)))

            windows = split_turns(
                length, turns, self.cfg.min_turn_s, audio,
                boundary_search_ms=self.cfg.boundary_search_ms,
                boundary_quiet_ms=self.cfg.boundary_quiet_ms,
                boundary_quiet_ratio=self.cfg.boundary_quiet_ratio,
            )
            if len(windows) <= 1 or len(windows) > self.cfg.max_turns:
                self.failures = 0
                log.debug(
                    "Sherpa unsplit %.2f-%.2fs: %d detected turns, %d safe cuts",
                    original.start_s, original.end_s,
                    len(turns), len(windows) - 1,
                )
                return [original]

            result: list[Utterance] = []
            for index, window in enumerate(windows):
                lo = max(0, round(window.start * SAMPLE_RATE))
                hi = min(samples, round(window.end * SAMPLE_RATE))
                if hi <= lo:
                    return [original]
                start = original.start_s + lo / SAMPLE_RATE
                end = original.start_s + hi / SAMPLE_RATE
                onset = max(
                    start,
                    original.speech_start_s
                    if original.speech_start_s is not None else start,
                )
                result.append(replace(
                    original, start_s=start, end_s=end,
                    audio=audio[lo:hi].copy(),
                    speech_start_s=min(end, onset),
                    # The previous cue must stop displaying when this new
                    # speaker takes over, including after cached VOD seeks.
                    speaker_handoff=(index > 0 and
                                     window.speaker != windows[index - 1].speaker),
                    timings=replace(original.timings),
                ))
            # Enforce exact conservation of PCM before emitting any split cue.
            if sum(len(part.audio) for part in result) != samples:
                log.warning("Sherpa cut validation failed; preserving original audio")
                return [original]
            self.failures = 0
            log.info(
                "Sherpa separated %.2f-%.2f into %d non-overlapping turns",
                original.start_s, original.end_s, len(result),
            )
            return result
        except Exception as exc:
            self.failures += 1
            log.warning(
                "Sherpa failure %d/3; preserving original audio: %s",
                self.failures, exc,
            )
            if self.failures >= 3:
                self.diarizer = None
                log.warning("Sherpa disabled for this session after repeated errors")
            return [original]


def run_speaker(cfg: SpeakerConfig, in_q: Any, out_q: Any, generation: Any = None) -> None:
    splitter: SherpaSplitter | None = None
    checked = 0
    split_windows = 0
    separated_cues = 0
    if cfg.enabled:
        print("Speaker diarization: loading Sherpa CPU models...", flush=True)
        try:
            splitter = SherpaSplitter(cfg)
        except Exception as exc:
            log.warning("Speaker diarization unavailable: %s", exc)
            print(f"Speaker diarization unavailable: {exc}; original audio retained.", flush=True)
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            return
        if generation is not None and item.generation != generation():
            continue
        if isinstance(item, AudioProgress):
            out_q.put(item)
            continue
        if splitter is None or splitter.diarizer is None:
            out_q.put(item)
            continue
        parts = splitter.split(item)
        checked += 1
        if len(parts) > 1:
            split_windows += 1
            separated_cues += len(parts)
        if checked % 12 == 0:
            log.info(
                "speaker turns: windows=%d split_windows=%d separated_cues=%d "
                "unsplit_windows=%d",
                checked, split_windows, separated_cues, checked - split_windows,
            )
        for part in parts:
            if generation is not None and part.generation != generation():
                break
            out_q.put(part)
