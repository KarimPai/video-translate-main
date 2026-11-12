"""Exercise the live decoder with real AAC/fMP4, independently opened HLS snapshots."""
import io
from pathlib import Path
import queue
import shutil
import threading
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from auto_translate.config import IngestConfig, VadConfig
from auto_translate.ingest import Ingest, Source, probe_live_segment
from auto_translate.live_clock import Discontinuity, pcm_frames
from auto_translate.types import SAMPLE_RATE
from auto_translate.vad import FRAME, Segmenter


class LiveFramesTests(unittest.TestCase):
    def test_incomplete_missing_skipped_and_reset_frames(self):
        record = b"[Parsed_ashowinfo_3] n:0 pts:15000 pts_time:0.9375 rate:16000 nb_samples:4\n"
        for lines, pcm in [
            (record, bytes(6)),
            (record.replace(b"pts:15000", b"pts:NOPTS"), bytes(8)),
            (record + record.replace(b"n:0", b"n:1"), bytes(16)),
            (record.replace(b"n:0", b"n:2"), bytes(8)),
            (b"[hls] Failed to open segment 98\n", b""),
        ]:
            with self.subTest(lines=lines), self.assertRaises(Discontinuity):
                list(pcm_frames(io.BytesIO(lines), io.BytesIO(pcm), 16000, 1))

    def test_diagnostics_and_short_missing_pts_preserve_pcm_alignment(self):
        header = b"[Parsed_ashowinfo_3 @ decoder] "
        def frame(n, pts):
            return header + f"n:{n} pts:{pts} pts_time:0 rate:16000 nb_samples:4\n".encode()
        logs = (frame(0, 16000) + header + b" side data - unknown side data type: 23\n"
                + frame(1, "NOPTS") + frame(2, 16008))
        pcm = bytes(range(24))
        self.assertEqual(list(pcm_frames(io.BytesIO(logs), io.BytesIO(pcm), 16000, 1)),
                         [(16000, pcm[:8]), (16004, pcm[8:16]), (16008, pcm[16:])])
        with self.assertRaisesRegex(Discontinuity, "audio PTS gap"):
            list(pcm_frames(io.BytesIO(logs.replace(b"pts:16008", b"pts:26008")),
                            io.BytesIO(pcm), 16000, 1))

    def test_startup_accepts_second_fragment_in_same_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest = Ingest(Source("fixture", live=True), IngestConfig())
            playlist = root / "index.m3u8"
            ingest.video_out = str(playlist)
            playlist.write_text("#EXTINF:2,\nsegment_000000.m4s\n#EXTINF:2,\nsegment_000001.m4s\n")
            for name in ("init.mp4", "segment_000000.m4s", "segment_000001.m4s"):
                (root / name).write_bytes(b"x")
            with patch("auto_translate.ingest.probe_live_segment",
                       side_effect=[KeyError("audio"), (4, 6)]) as probe:
                self.assertTrue(ingest.wait_video_ready(1))
                self.assertTrue(ingest.wait_video_ready(1))
                self.assertEqual(probe.call_count, 2)
            self.assertEqual(ingest.generation, 0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg suite")
    def test_rotated_vfr_playlist_and_resampled_speech_onset(self):
        with tempfile.TemporaryDirectory() as tmp:
            playlist = Path(tmp) / "index.m3u8"
            # Drop every third frame after one second to vary frame durations.
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:r=30:d=4",
                "-f", "lavfi", "-i",
                "aevalsrc=if(between(t\\,1.4\\,2.2)\\,0.4*sin(2*PI*440*t)\\,0):s=44100:d=4",
                "-vf", "select='lt(t,1)+mod(n,3)'", "-fps_mode", "vfr",
                "-c:v", "libx264", "-bf", "0", "-force_key_frames", "expr:gte(t,n_forced*1)",
                "-c:a", "aac", "-f", "hls", "-hls_time", "1", "-hls_segment_type", "fmp4",
                "-hls_segment_filename", "segment_%03d.m4s", playlist.name,
            ], cwd=tmp, check=True, timeout=15)
            lines = playlist.read_text().splitlines()
            origin, _ = probe_live_segment(playlist, "segment_000.m4s")
            first = lines.index("segment_000.m4s")
            del lines[first-1:first+1]
            lines[lines.index("#EXT-X-MEDIA-SEQUENCE:0")] = "#EXT-X-MEDIA-SEQUENCE:1"
            playlist.write_text("\n".join(lines) + "\n")
            later, _ = probe_live_segment(playlist, "segment_001.m4s")
            self.assertGreater(later - origin, 0.8)
            ingest = Ingest(Source("local-fixture", live=True), IngestConfig())
            ingest._cache_root = Path(tmp)
            ingest.video_out = str(playlist)
            ingest._start_cache_server()
            chunks = queue.Queue()
            failures = []
            def decode():
                try:
                    ingest._read_session(chunks)
                except Exception as exc:
                    failures.append(exc)
            try:
                with patch.object(ingest, "start_video"):
                    reader = threading.Thread(target=decode, daemon=True)
                    reader.start()
                    reader.join(10)
                    self.assertFalse(reader.is_alive(), "live decoder pipe stalled")
            finally:
                ingest.stop()
            if failures:
                raise failures[0]
            frames = list(chunks.queue)
            self.assertTrue(frames[0].timestamped)
            self.assertAlmostEqual(ingest.first_pts, later, delta=0.002)
            self.assertAlmostEqual(frames[0].start_sample / SAMPLE_RATE, later, delta=0.002)
            vad = Mock(side_effect=lambda x: np.where(
                np.max(np.abs(x.reshape(-1, FRAME)), axis=1) > 0.04, 0.99, 0.01))
            with patch("auto_translate.vad.StreamingFireRedVAD", return_value=vad):
                seg = Segmenter(VadConfig())
            utterances = []
            for chunk in frames:
                utterances.extend(seg.feed(chunk))
            speech = [u for u in utterances + seg.flush() if not u.probe]
            self.assertEqual(len(speech), 1)
            self.assertAlmostEqual(speech[0].speech_start_s, origin + 1.4, delta=0.10)
