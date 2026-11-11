"""Live MPV and ASR must consume the same local HLS playlist; VOD stays direct."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from auto_translate.ingest import Ingest, Source
from auto_translate.output.mpv import _mpv_media_url


class SharedLiveHlsTests(unittest.TestCase):
    @staticmethod
    def fake_ingest(*, live: bool, shared: bool):
        return SimpleNamespace(
            source=SimpleNamespace(
                location="https://www.youtube.com/watch?v=example",
                live=live,
            ),
            video_out="temp/fmp4/session/index.m3u8" if shared else None,
            video_playlist_url="http://127.0.0.1:18650/session/index.m3u8",
        )

    def test_live_mpv_uses_same_playlist_as_audio_reader(self):
        ingest = self.fake_ingest(live=True, shared=True)
        self.assertEqual(_mpv_media_url(ingest), ingest.video_playlist_url)

    def test_vod_always_uses_original_media_even_when_cache_exists(self):
        ingest = self.fake_ingest(live=False, shared=True)
        self.assertEqual(_mpv_media_url(ingest), ingest.source.location)

    def test_live_without_shared_cache_keeps_original_source(self):
        ingest = self.fake_ingest(live=True, shared=False)
        self.assertEqual(_mpv_media_url(ingest), ingest.source.location)

    def test_shared_audio_reader_starts_at_first_hls_segment(self):
        ingest = Ingest.__new__(Ingest)
        ingest.source = Source(
            location="https://www.youtube.com/watch?v=example",
            live=True,
        )
        ingest.video_out = str(Path("temp/fmp4/session/index.m3u8"))
        ingest._cache_root = Path("temp/fmp4")
        ingest._cache_source_start_s = 0.0
        ingest._http_server = SimpleNamespace(server_port=18650)
        ingest.separation_enabled = False
        ingest.deepfilter_enabled = False
        ingest.pitch_semitones = 0.0
        cmd = ingest._video_audio_command()
        pos = cmd.index("-live_start_index")
        self.assertEqual(cmd[pos + 1], "0")
        self.assertEqual(
            cmd[cmd.index("-i") + 1],
            ingest.video_playlist_url,
        )


if __name__ == "__main__":
    unittest.main()
