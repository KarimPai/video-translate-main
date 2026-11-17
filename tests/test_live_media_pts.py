"""PTS evidence at the raw-PCM boundary, including a real FFmpeg HLS decode."""
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest

from auto_translate.ingest import Ingest, Source


def make_ingest():
    ingest = object.__new__(Ingest)
    ingest.source = Source("synthetic", live=True, start_s=12.0)
    ingest._proc_lock = threading.Condition()
    ingest._session = 1
    ingest._generation = 3
    ingest._stopped = False
    ingest._audio_pts_invalid = False
    ingest._audio_pts_samples = 0
    ingest._audio_pts_rate = 0
    ingest._source_pts_origin = None
    ingest._source_pts_samples = 0
    ingest._source_pts_rate = 0
    ingest.audio_media_pts = None
    ingest.audio_media_pts_ready = threading.Event()
    ingest._video_failed = threading.Event()
    ingest.first_pts = 12.0
    return ingest


def frame(n, pts, count=1600, rate=16000):
    return (f"[Parsed_ashowinfo_3 @ decoder] n:{n} pts:{pts} pts_time:rounded "
            f"pos:0 fmt:s16 channels:1 chlayout:mono rate:{rate} nb_samples:{count}")


class LiveMediaPtsTests(unittest.TestCase):
    def test_exact_integer_pts_survives_rounded_multi_hour_decimal(self):
        ingest = make_ingest()
        origin = 560000 * 16000 + 4321
        ingest._capture_audio_pts(frame(0, origin), 1)
        ingest._capture_audio_pts(frame(1, origin + 1600), 1)
        self.assertEqual(ingest.live_clock_anchor(), (3, 12, origin / 16000))
        self.assertFalse(ingest._video_failed.is_set())

    def test_arrival_readiness_is_not_media_pts_evidence(self):
        ingest = make_ingest()
        self.assertIsNone(ingest.live_clock_anchor())
        self.assertFalse(ingest.audio_media_pts_ready.is_set())
        ingest._capture_audio_pts(frame(0, 48000), 1)
        self.assertEqual(ingest.live_clock_anchor(), (3, 12, 3))

    def test_old_session_and_missing_initial_timestamp_cannot_anchor(self):
        ingest = make_ingest()
        ingest._capture_audio_pts(frame(0, 100), 0)
        ingest._capture_audio_pts(frame(1, 1600), 1)
        ingest._capture_audio_pts(frame(0, 0).replace("pts:0", "pts:NOPTS"), 1)
        ingest._capture_audio_pts(frame(0, 0, rate=0), 1)
        self.assertIsNone(ingest.live_clock_anchor())

    def test_timestamp_gap_or_accumulating_drift_invalidates_shared_session(self):
        for delta in (16000, -16000, 1700):
            ingest = make_ingest()
            ingest._capture_audio_pts(frame(0, 0), 1)
            ingest._capture_audio_pts(frame(1, 1600 + delta), 1)
            self.assertIsNone(ingest.live_clock_anchor())
            self.assertFalse(ingest.audio_media_pts_ready.is_set())
            self.assertTrue(ingest._video_failed.is_set())

    def test_many_hours_of_sample_counts_do_not_accumulate_error(self):
        ingest = make_ingest()
        for n in range(360):
            # Each diagnostic frame represents 100 seconds of exact samples.
            ingest._capture_audio_pts(frame(n, 15260 * 16000 + n * 1600000,
                                            count=1600000), 1)
        self.assertFalse(ingest._video_failed.is_set())
        self.assertEqual(ingest._audio_pts_samples / 16000, 36000)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable unavailable")
    def test_pitch_preserves_native_48k_duration_and_large_media_origin(self):
        for pitch in (-2.0, 2.0):
            for origin in (0.0, 15260.0):
                with self.subTest(pitch=pitch, media_origin=origin):
                    ingest = make_ingest()
                    ingest.pitch_semitones = pitch
                    filters = [f"asetpts=PTS+{origin}/TB", *ingest._pitch_filter(16000),
                               "aresample=16000", "aformat=sample_fmts=s16:channel_layouts=mono",
                               "asettb=1/16000", "ashowinfo"]
                    decoded = subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "info", "-copyts",
                        "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=2",
                        "-af", ",".join(filters), "-f", "s16le", "pipe:1",
                    ], capture_output=True, timeout=20, check=True)
                    for line in decoded.stderr.decode().splitlines():
                        if "ashowinfo" in line:
                            ingest._capture_audio_pts(line, 1)
                    # Existing native48k->asetrate18k incorrectly produced
                    # almost 6s, and moved a15260s origin to40786s (+2 pitch).
                    self.assertAlmostEqual(len(decoded.stdout) / 32000, 2.0, delta=0.02)
                    self.assertAlmostEqual(ingest.audio_media_pts, origin, delta=1.01 / 16000)
                    self.assertEqual(len(decoded.stdout) // 2, ingest._audio_pts_samples)
                    self.assertFalse(ingest._video_failed.is_set())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable unavailable")
    def test_pitch_source_gap_is_detected_even_when_atempo_rewrites_pts(self):
        ingest = make_ingest()
        ingest.pitch_semitones = 2.0
        ingest.video_out = "synthetic/index.m3u8"
        ingest._cache_root = Path("synthetic")
        ingest._cache_source_start_s = 12.0
        ingest._http_server = SimpleNamespace(server_port=18650)
        ingest.separation_enabled = ingest.deepfilter_enabled = False
        command = ingest._video_audio_command()
        filters = command[command.index("-af") + 1]
        source_pts = r"asetpts=PTS+15260/TB+if(gte(T\,1)\,2/TB\,0)"
        decoded = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-copyts",
            "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=2",
            "-af", source_pts + "," + filters, "-f", "s16le", "pipe:1",
        ], capture_output=True, timeout=20, check=True)
        for line in decoded.stderr.decode().splitlines():
            if "ashowinfo" in line:
                ingest._capture_audio_pts(line, 1)
        self.assertTrue(ingest._video_failed.is_set())
        self.assertFalse(ingest.audio_media_pts_ready.is_set())
        self.assertIsNone(ingest.live_clock_anchor())

    def test_pre_pitch_observer_never_anchors_pcm_and_restart_resets_both(self):
        from auto_translate.config import IngestConfig
        ingest = Ingest(Source("synthetic", live=True), IngestConfig())
        pre = frame(0, 16000).replace("Parsed_ashowinfo_3", "ashowinfo@source")
        ingest._capture_audio_pts(pre, ingest._session)
        self.assertEqual(ingest._source_pts_origin, 1.0)
        self.assertIsNone(ingest.audio_media_pts)
        self.assertFalse(ingest.audio_media_pts_ready.is_set())
        ingest._capture_audio_pts(frame(0, 16000), ingest._session)
        self.assertTrue(ingest.audio_media_pts_ready.is_set())
        ingest.request_restart(12.0)
        self.assertIsNone(ingest._source_pts_origin)
        self.assertEqual(ingest._source_pts_samples, 0)
        self.assertIsNone(ingest.audio_media_pts)
        self.assertEqual(ingest._audio_pts_samples, 0)
        self.assertFalse(ingest.audio_media_pts_ready.is_set())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "FFmpeg/FFprobe executables unavailable")
    def test_real_hls_pcm_boundary_preserves_media_pts_and_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            playlist = Path(directory) / "index.m3u8"
            generated = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "sine=frequency=500:sample_rate=48000:duration=2",
                "-af", "asetpts=PTS+15260/TB", "-c:a", "aac",
                "-f", "hls", "-hls_segment_type", "fmp4", "-hls_time", "1",
                "-hls_list_size", "0", str(playlist),
            ], capture_output=True, timeout=20)
            self.assertEqual(generated.returncode, 0, generated.stderr.decode(errors="replace"))
            ingest = make_ingest()
            ingest.video_out = str(playlist)
            ingest._cache_root = Path(directory)
            ingest._cache_source_start_s = 12.0
            ingest._http_server = SimpleNamespace(server_port=18650)
            ingest.separation_enabled = ingest.deepfilter_enabled = False
            ingest.pitch_semitones = 0.0
            command = ingest._video_audio_command()
            command[command.index("-i") + 1] = str(playlist)
            decoded = subprocess.run(command, capture_output=True, timeout=20)
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
            for line in decoded.stderr.decode().splitlines():
                if "ashowinfo" in line:
                    ingest._capture_audio_pts(line, 1)
            self.assertTrue(ingest.audio_media_pts_ready.is_set())
            self.assertFalse(ingest._video_failed.is_set())
            probed = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(playlist),
            ], capture_output=True, timeout=20, check=True)
            first_pts = float(json.loads(probed.stdout)["frames"][0]["best_effort_timestamp_time"])
            self.assertAlmostEqual(ingest.audio_media_pts, first_pts, delta=1 / 16000)
            self.assertEqual(len(decoded.stdout) // 2, ingest._audio_pts_samples)
            self.assertGreater(ingest._audio_pts_samples, 30000)


if __name__ == "__main__":
    unittest.main()
