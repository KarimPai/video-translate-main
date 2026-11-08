"""Best-effort, offline-window Sherpa-ONNX speaker segmentation on CPU.

No source separation: on overlapping speech or uncertain turns, preserve
FireRedVAD's original utterance. Speaker IDs are anonymous cosine clusters.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import tarfile
import tempfile
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


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int | None


def _model_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) / "auto-translate" / "speaker-models") if local else (
        Path.home() / ".cache" / "auto-translate" / "speaker-models")


def _download(url: str, target: Path, member: str = "") -> None:
    if target.is_file() and target.stat().st_size > 1024:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Speaker diarization: downloading {target.name}...", flush=True)
    log.info("Downloading Sherpa model: %s", url)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            source_path = Path(output.name)
            temporary.append(source_path)
            received = 0
            with urllib.request.urlopen(url, timeout=25) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > 150 * 1024 * 1024:
                        raise ValueError("Unexpectedly large diarization model")
                    output.write(chunk)
        if member:
            with tarfile.open(source_path, "r:bz2") as archive:
                info = archive.getmember(member)
                if not info.isfile() or info.size > 25 * 1024 * 1024:
                    raise ValueError("Unsafe Sherpa model archive member")
                stream = archive.extractfile(info)
                if stream is None:
                    raise ValueError("Speaker model missing from archive")
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    model_tmp = Path(out.name)
                    temporary.append(model_tmp)
                    shutil.copyfileobj(stream, out)
                model_tmp.replace(target)
        else:
            source_path.replace(target)
        if target.stat().st_size <= 1024:
            target.unlink(missing_ok=True)
            raise ValueError("Diarization model was empty")
        log.info("Speaker model cached: %s", target)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def split_turns(length: float, raw: list[Turn], min_seconds: float) -> list[Turn]:
    """Use only clear non-overlapping boundaries, keeping every PCM sample."""
    unknown = [Turn(0.0, length, None)]
    if length <= 0 or not raw:
        return unknown
    merged: list[Turn] = []
    for turn in sorted(raw, key=lambda t: t.start):
        if turn.speaker is None or turn.end <= turn.start:
            continue
        if merged and merged[-1].speaker == turn.speaker and turn.start <= merged[-1].end + 0.2:
            last = merged[-1]
            merged[-1] = Turn(last.start, max(last.end, turn.end), last.speaker)
        else:
            merged.append(turn)
    if not merged:
        return unknown
    if len(merged) == 1:
        return [Turn(0.0, length, merged[0].speaker)]
    cuts: list[float] = []
    labels = [merged[0].speaker]
    for prev, next_ in zip(merged, merged[1:]):
        if next_.speaker == prev.speaker:
            continue
        if prev.end > next_.start + 0.12:
            return unknown  # overlapping voices require another model
        boundary = max(0.0, min(length, (prev.end + next_.start) / 2))
        if boundary - (cuts[-1] if cuts else 0.0) < min_seconds:
            return unknown
        cuts.append(boundary)
        labels.append(next_.speaker)
    if cuts and length - cuts[-1] < min_seconds:
        return unknown
    if not cuts:
        return [Turn(0.0, length, merged[0].speaker)]
    edges = [0.0, *cuts, length]
    return [Turn(edges[i], edges[i+1], labels[i]) for i in range(len(labels))]


class VoiceProfiles:
    def __init__(self, threshold: float, limit: int) -> None:
        self.threshold, self.limit = threshold, limit
        self.vectors: list[np.ndarray] = []
        self.counts: list[int] = []

    def identify(self, embedding: np.ndarray) -> int | None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-6:
            return None
        vector = vector / norm
        scores = [float(np.dot(vector, ref)) for ref in self.vectors]
        if scores and max(scores) >= self.threshold:
            idx = int(np.argmax(scores))
            updated = self.vectors[idx] * min(6, self.counts[idx]) + vector
            self.vectors[idx] = updated / np.linalg.norm(updated)
            self.counts[idx] += 1
            return idx
        if len(self.vectors) >= self.limit:
            return None
        self.vectors.append(vector)
        self.counts.append(1)
        return len(self.vectors) - 1


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
        _download(SEG_URL, seg, SEG_MEMBER)
        _download(EMB_URL, emb)
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
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb), num_threads=cfg.num_threads, provider="cpu"))
        self.profiles = VoiceProfiles(cfg.similarity_threshold, cfg.max_speakers)
        self.failures = 0
        print("Speaker diarization: Sherpa CPU ready.", flush=True)

    def _embedding(self, audio: np.ndarray) -> np.ndarray | None:
        if len(audio) < SAMPLE_RATE:
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(audio))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return np.asarray(self.extractor.compute(stream), dtype=np.float32)

    def split(self, original: Utterance) -> list[Utterance]:
        if original.probe or len(original.audio) < SAMPLE_RATE:
            return [original]
        # Offline pyannote segmentation needs a larger observation window.
        # For brief utterances do voice identification only, without trying
        # to hallucinate a speaker boundary in a 1-2 second speech sample.
        if len(original.audio) < 3 * SAMPLE_RATE:
            try:
                v = self._embedding(original.audio)
                speaker = self.profiles.identify(v) if v is not None else None
                return [replace(original, speaker_id=speaker)] if speaker is not None else [original]
            except Exception as exc:
                log.debug("Short-turn voice embedding unavailable: %s", exc)
                return [original]
        try:
            audio = np.ascontiguousarray(original.audio, dtype=np.float32)
            raw = self.diarizer.process(audio).sort_by_start_time()
            turns = [
                Turn(max(0.0, float(t.start)), min(len(audio) / SAMPLE_RATE, float(t.end)), int(t.speaker))
                for t in raw if math.isfinite(float(t.start)) and math.isfinite(float(t.end))
            ]
            windows = split_turns(len(audio) / SAMPLE_RATE, turns, self.cfg.min_turn_s)
            if len(windows) > self.cfg.max_turns:
                return [original]
            ids: dict[int, int | None] = {}
            result: list[Utterance] = []
            for window in windows:
                lo = max(0, round(window.start * SAMPLE_RATE))
                hi = min(len(audio), round(window.end * SAMPLE_RATE))
                if hi <= lo:
                    return [original]
                part = audio[lo:hi]
                identity = None
                if window.speaker is not None:
                    if window.speaker not in ids:
                        v = self._embedding(part)
                        ids[window.speaker] = self.profiles.identify(v) if v is not None else None
                    identity = ids[window.speaker]
                start = original.start_s + lo / SAMPLE_RATE
                end = original.start_s + hi / SAMPLE_RATE
                onset = max(start, original.speech_start_s if original.speech_start_s is not None else start)
                result.append(replace(
                    original, start_s=start, end_s=end, audio=part.copy(),
                    speaker_id=identity, speech_start_s=min(end, onset),
                    timings=replace(original.timings)))
            self.failures = 0
            if len(result) > 1:
                log.info("Sherpa separated %.2f-%.2f into %d turns",
                         original.start_s, original.end_s, len(result))
            return result
        except Exception as exc:
            self.failures += 1
            log.warning("Sherpa failure %d/3; preserving original audio: %s", self.failures, exc)
            if self.failures >= 3:
                self.diarizer = None
                log.warning("Sherpa disabled for the session after repeated errors")
            return [original]


def run_speaker(cfg: SpeakerConfig, in_q: Any, out_q: Any, generation: Any = None) -> None:
    splitter: SherpaSplitter | None = None
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
        for part in splitter.split(item):
            if generation is not None and part.generation != generation():
                break
            out_q.put(part)
