"""Quality selector regressions: do not exceed selected YouTube height."""
import re
import unittest

from auto_translate.config import (
    MpvConfig, VIDEO_QUALITY_HEIGHTS, mpv_youtube_format,
)


class VideoQualityTests(unittest.TestCase):
    def test_resolution_cap_preserved_for_all_fallbacks(self):
        for quality in VIDEO_QUALITY_HEIGHTS:
            for live in (True, False):
                with self.subTest(quality=quality, live=live):
                    selection = mpv_youtube_format(quality, live=live)
                    matches = re.findall(r"\[height<=(\d+)\]", selection)
                    self.assertTrue(matches)
                    self.assertEqual(set(map(int, matches)), {quality})
                    self.assertNotIn("[height>=", selection)

    def test_finished_vod_https_first_and_hls_fallback(self):
        for quality in VIDEO_QUALITY_HEIGHTS:
            selector = mpv_youtube_format(quality, live=False)
            self.assertTrue(selector.startswith(
                f"bv*[height<={quality}][vcodec^=avc1][protocol=https]"
            ))
            self.assertIn(f"/b[height<={quality}][protocol^=m3u8]", selector)
            self.assertIn(f"/bv*[height<={quality}]+ba/b[height<={quality}]", selector)

    def test_live_hls_preference_preserved(self):
        for quality in VIDEO_QUALITY_HEIGHTS:
            selector = mpv_youtube_format(quality, live=True)
            self.assertTrue(selector.startswith(
                f"bv*[height<={quality}][vcodec^=avc1][protocol^=m3u8]"
            ))
            self.assertNotIn("[protocol=https]", selector)

    def test_unselected_cli_keeps_prior_1080_format_defaults(self):
        config = MpvConfig()
        self.assertEqual(config.ytdlp_format, mpv_youtube_format(1080, live=True))
        self.assertEqual(config.vod_ytdlp_format, mpv_youtube_format(1080, live=False))

    def test_invalid_quality_does_not_reach_ytdlp(self):
        for quality in (0, 240, 540, 1440):
            with self.assertRaises(ValueError):
                mpv_youtube_format(quality, live=False)


if __name__ == "__main__":
    unittest.main()
