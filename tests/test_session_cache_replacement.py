"""Corrected VOD subtitles replace cached and queued copies safely."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from auto_translate.output.mpv import MpvOutput
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle


def _sub(start: float, end: float, ja: str, en: str, *, generation: int = 0) -> Subtitle:
    return Subtitle(start, end, ja, en, generation=generation)


def test_cache_replaces_corrected_translation_without_duplicate_overlap():
    cache = SessionSubtitleCache()
    first = _sub(1.0, 2.0, "ありがとう、みんな", "Thank you, everyone.")
    corrected = _sub(1.1, 2.1, "ありがとう、みんな", "Thanks, everyone.")

    assert cache.add(first)
    assert cache.add(corrected)
    assert len(cache) == 1
    assert cache.match(1.1, 2.1).en == "Thanks, everyone."
    assert [sub.en for sub in cache.active_at(1.5)] == ["Thanks, everyone."]


def test_cache_keeps_overlapping_different_dialogue():
    cache = SessionSubtitleCache()
    assert cache.add(_sub(1.0, 2.0, "ありがとう", "Thank you."))
    assert cache.add(_sub(1.0, 2.0, "何してるの", "What are you doing?"))
    assert len(cache) == 2


def test_mpv_replaces_visible_copy_of_corrected_same_source_cue():
    output = object.__new__(MpvOutput)
    output._session_start = 0.0
    output._now = None
    output.cfg = SimpleNamespace(live_late_window_s=7.0)
    output.ingest = SimpleNamespace(source=SimpleNamespace(live=False), generation=0)
    output._display_until = lambda _sub, _start, until: until
    output._lock = threading.Lock()
    output._lines = []
    output._subtitle_ready = threading.Event()

    output.show(_sub(1.0, 2.0, "ありがとう、みんな", "Thank you, everyone."))
    output.show(_sub(1.1, 2.1, "ありがとう、みんな", "Thanks, everyone."))

    assert len(output._lines) == 1
    assert output._lines[0].sub.en == "Thanks, everyone."