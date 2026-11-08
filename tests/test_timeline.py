"""yt-dlp timeline metadata command and error diagnostics."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_translate.config import IngestConfig
from auto_translate.timeline import load_vod_metadata


class YoutubeMetadataTests(unittest.TestCase):
    def test_metadata_uses_ejs_like_youtube_audio_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"id": "LkLNJ-K3368", "title": "Example stream",
                       "duration": 7200, "live_status": "was_live"}
            with patch("auto_translate.timeline._cache_dir", return_value=Path(tmp)), patch(
                "auto_translate.timeline.subprocess.run",
                return_value=SimpleNamespace(stdout=json.dumps(payload)),
            ) as run:
                timeline = load_vod_metadata(
                    "https://www.youtube.com/watch?v=LkLNJ-K3368", IngestConfig(),
                )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--remote-components") + 1], "ejs:github")
        self.assertNotIn("--no-call-home", command)
        self.assertEqual(timeline["duration_s"], 7200.0)
        self.assertEqual(timeline["video_id"], "LkLNJ-K3368")

    def test_temporary_youtube_live_error_retries_then_succeeds(self):
        transient = subprocess.CalledProcessError(
            1, ["yt-dlp"],
            stderr="ERROR: [youtube] HNsjOXRyBPA: We're experiencing technical difficulties.",
        )
        payload = json.dumps({
            "id": "HNsjOXRyBPA", "title": "Live event",
            "duration": None, "live_status": "is_live",
        })
        with tempfile.TemporaryDirectory() as tmp, patch(
            "auto_translate.timeline._cache_dir", return_value=Path(tmp)
        ), patch("auto_translate.timeline.subprocess.run", side_effect=[
            transient, transient, SimpleNamespace(stdout=payload),
        ]) as run, patch("auto_translate.timeline.time.sleep") as sleep:
            result = load_vod_metadata(
                "https://www.youtube.com/live/HNsjOXRyBPA", IngestConfig(),
            )
        self.assertEqual(result["live_status"], "is_live")
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.5, 3.0])

    def test_persistent_youtube_live_error_stops_after_three_attempts(self):
        transient = subprocess.CalledProcessError(
            1, ["yt-dlp"],
            stderr="ERROR: [youtube] HNsjOXRyBPA: We're experiencing technical difficulties.",
        )
        with patch("auto_translate.timeline.subprocess.run", side_effect=transient) as run, patch(
            "auto_translate.timeline.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(RuntimeError, "technical difficulties"):
                load_vod_metadata("https://www.youtube.com/live/HNsjOXRyBPA", IngestConfig())
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_metadata_failure_surfaces_actual_ytdlp_error(self):
        failure = subprocess.CalledProcessError(
            1, ["yt-dlp"], stderr="ERROR: [youtube] Sign in to confirm you're not a bot",
        )
        with patch("auto_translate.timeline.subprocess.run", side_effect=failure) as run, patch(
            "auto_translate.timeline.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(RuntimeError, "Sign in to confirm"):
                load_vod_metadata("https://www.youtube.com/watch?v=LkLNJ-K3368", IngestConfig())
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
