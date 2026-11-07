"""Local yt-dlp cookie authentication is consistent across pipeline and MPV."""
import tempfile
import unittest
from pathlib import Path

from auto_translate.auth import cookie_args, mpv_auth_args
from auto_translate.config import IngestConfig


class YoutubeAuthenticationTests(unittest.TestCase):
    def test_no_cookie_file_is_safe_and_unauthenticated(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.assertEqual(cookie_args(IngestConfig(), directory), [])
            self.assertEqual(mpv_auth_args(IngestConfig(), directory), [])

    def test_existing_local_cookie_file_is_used_by_ytdlp_and_mpv(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            file = directory / "cookies.txt"
            file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            expected = str(file.resolve())
            self.assertEqual(
                cookie_args(IngestConfig(), directory),
                ["--cookies", expected],
            )
            self.assertEqual(
                mpv_auth_args(IngestConfig(), directory),
                [f"--ytdl-raw-options=cookies={expected}"],
            )

    def test_explicit_browser_config_takes_precedence_over_file(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "cookies.txt").write_text(
                "# Netscape HTTP Cookie File\n", encoding="utf-8",
            )
            cfg = IngestConfig(cookies_from_browser="edge")
            self.assertEqual(
                cookie_args(cfg, directory), ["--cookies-from-browser", "edge"]
            )
            self.assertEqual(
                mpv_auth_args(cfg, directory),
                ["--ytdl-raw-options=cookies-from-browser=edge"],
            )

    def test_legacy_youtube_cookies_file_is_supported(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            file = directory / "youtube-cookies.txt"
            file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            self.assertEqual(
                cookie_args(IngestConfig(), directory),
                ["--cookies", str(file.resolve())],
            )


if __name__ == "__main__":
    unittest.main()
