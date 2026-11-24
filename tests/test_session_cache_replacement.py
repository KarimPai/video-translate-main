"""Corrected VOD subtitles replace cached and queued copies safely."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from auto_translate.ingest import _vod_eof_tolerance_s
from auto_translate.output.mpv import MpvOutput
from auto_translate.session_cache import SessionSubtitleCache
from auto_translate.types import Subtitle
from auto_translate.vod_coverage import VodCoverage


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


def test_timestamp_match_declines_ambiguous_overlapping_sources():
    cache = SessionSubtitleCache()
    assert cache.add(_sub(1.0, 2.0, "ありがとう", "Thank you."))
    assert cache.add(_sub(1.05, 2.05, "何してるの", "What are you doing?"))

    assert cache.match(1.02, 2.02) is None


def test_unresolved_speech_clips_reusable_coverage_until_resolved():
    cache = SessionSubtitleCache()
    cache.record_processed(0.0, 10.0)
    cache.record_unprocessed(4.0, 5.0, identity=1)

    assert cache.covered_until(2.0) == 4.0
    assert cache.covered_until(4.5) == 4.5

    cache.record_confirmed_silence(1)
    assert cache.covered_until(2.0) == 10.0


def test_translated_speech_resolves_only_one_matching_failed_interval():
    cache = SessionSubtitleCache()
    cache.record_processed(0.0, 10.0)
    cache.record_unprocessed(4.0, 5.0, identity=1)
    cache.record_unprocessed(4.0, 5.0, identity=2)

    cache.record_translated(1)

    assert cache.covered_until(2.0) == 4.0


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


def test_retry_split_resolves_only_completed_portions():
    cache = SessionSubtitleCache()
    work = VodCoverage(cache)
    work.set_generation(1)
    work.register(4.0, 8.0, 1)
    work.advance(0.0, 10.0, 1)
    work.set_generation(2)
    first = work.register(4.0, 6.0, 2)
    second = work.register(6.0, 8.0, 2)
    work.advance(0.0, 10.0, 2)
    assert cache.covered_until(0.0) == 4.0
    work.resolve(first, 2)
    assert cache.covered_until(0.0) == 6.0
    work.resolve(second, 2)
    assert cache.covered_until(0.0) == 10.0


def test_retry_merge_resolves_nonoverlapping_failures():
    cache = SessionSubtitleCache()
    work = VodCoverage(cache)
    work.set_generation(1)
    work.register(4.0, 5.0, 1)
    work.register(5.0, 6.0, 1)
    work.advance(0.0, 10.0, 1)
    work.set_generation(2)
    ids = work.register(3.9, 6.1, 2)
    work.resolve(ids, 2)
    assert cache.covered_until(0.0) == 10.0


def test_retry_partial_overlap_keeps_unverified_edges_and_stale_generation():
    cache = SessionSubtitleCache()
    work = VodCoverage(cache)
    work.set_generation(1)
    old = work.register(4.0, 8.0, 1)
    work.advance(0.0, 10.0, 1)
    work.set_generation(2)
    work.resolve(old, 1)
    ids = work.register(4.2, 7.8, 2)
    work.resolve(ids, 2)
    assert cache.covered_until(0.0) == 4.0
    assert not work.needs_retry(4.2, 7.8)
    assert work.needs_retry(7.8, 8.0)


def test_exact_retry_does_not_resolve_second_identical_failure():
    cache = SessionSubtitleCache()
    work = VodCoverage(cache)
    work.set_generation(1)
    work.register(4.0, 5.0, 1)
    work.register(4.0, 5.0, 1)
    work.advance(0.0, 10.0, 1)
    work.set_generation(2)
    assert work.needs_retry(4.0, 5.0)
    ids = work.register(4.0, 5.0, 2)
    work.resolve(ids, 2)
    assert cache.covered_until(0.0) == 4.0


def test_duplicate_resolution_requires_equivalent_source():
    cache = SessionSubtitleCache()
    work = VodCoverage(cache)
    work.set_generation(1)
    work.advance(0.0, 10.0, 1)
    cue = _sub(4.0, 5.0, "ありがとう", "Thanks.", generation=1)
    assert cache.add(cue)
    ids = work.register(4.0, 5.0, 1)
    work.resolve_duplicate(_sub(4.0, 5.0, "感謝", "Thanks.", generation=1), ids)
    assert cache.covered_until(0.0) == 4.0
    work.resolve_duplicate(cue, ids)
    assert cache.covered_until(0.0) == 10.0


def test_remote_eof_tolerance_does_not_scale_with_duration():
    assert _vod_eof_tolerance_s(60.0, remote=True) == 3.0
    assert _vod_eof_tolerance_s(21600.0, remote=True) == 3.0
    assert _vod_eof_tolerance_s(60.0) == 5.0
    assert _vod_eof_tolerance_s(21600.0) == 10.0
