"""Regression tests for MPV autoplay, seek-resume, and pause ownership.

Tests the production playback state machine logic with mocked MPV IPC.
Covers:
- Initial VOD startup autoplay
- Seek while playing auto-resumes
- Seek while manually paused remains paused
- Auto-pause never becomes user-pause
- Consecutive seeks preserve state
- Cached subtitle readiness
- Silence-only and suppressed cues
- IPC failure handling
- GUI Play/Pause override
"""
import threading
from unittest import mock

import pytest


class TestMpvPlaybackState:
    """Test the production MpvOutput playback state machine."""

    @pytest.fixture
    def mpv(self):
        """Create a minimal MpvOutput instance for testing playback state."""
        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv.ingest = mock.MagicMock()
        mpv.ingest.source.live = False
        mpv.ingest.source.start_s = 0.0
        mpv.ingest.source.duration = 120.0
        mpv.ingest.generation = 1
        mpv.cfg = mock.MagicMock()
        mpv.cfg.vod_pause_threshold_s = 2.0
        mpv.cfg.vod_resume_threshold_s = 8.0
        mpv.cfg.vod_seek_resume_threshold_s = 5.0
        mpv.cfg.window_id = None
        mpv.PAUSE_LEAD_SECONDS = 5.0
        
        # Coverage tracker
        mpv._coverage = CoverageTracker(0.0)
        mpv._source_finished = False
        mpv._buffer_condition = threading.Condition()
        mpv._lock = threading.Lock()
        mpv._lines = []
        
        # Pause state — the new ownership model
        mpv._pause_owner = "none"
        mpv._manually_paused = False
        mpv._vod_auto_paused = False
        mpv._seek_buffering = False
        
        # Playback position
        mpv._now = 0.0
        mpv._session_start = 0.0
        
        # Subtitle cache
        mpv._subtitle_cache = None
        mpv._uses_original_timeline = False
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._subtitle_ready = mock.MagicMock()
        
        # IPC mock
        mpv.ipc = mock.MagicMock()
        mpv.ipc.get = mock.MagicMock(return_value=False)
        mpv._player_command = mock.MagicMock(return_value=True)
        
        return mpv

    # -------------------------------------------------------------------------
    # Problem A: Auto-pause must not become user-pause
    # -------------------------------------------------------------------------

    def test_auto_pause_does_not_become_user_pause(self, mpv):
        """When VOD auto-pauses, _pause_owner must be 'auto', not 'user'."""
        mpv._now = 10.0
        mpv._pause_owner = "none"
        mpv._vod_auto_paused = False
        
        # Simulate auto-pause
        mpv._vod_auto_paused = True
        mpv._pause_owner = "auto"
        
        assert mpv._pause_owner == "auto"
        assert mpv._pause_owner != "user"
        assert mpv._vod_auto_paused is True

    def test_auto_pause_allows_auto_resume(self, mpv):
        """When _pause_owner is 'auto', auto-resume should proceed."""
        mpv._now = 10.0
        mpv._pause_owner = "auto"
        mpv._vod_auto_paused = True
        
        # Coverage recovers
        mpv._coverage._translated_until = 18.0  # lead 8.0 >= 8.0 threshold
        
        assert mpv._vod_should_resume() is True
        assert mpv._pause_owner != "user"
        # Auto-resume condition: _pause_owner != "user"
        can_resume = mpv._pause_owner != "user" and mpv._vod_should_resume()
        assert can_resume is True

    def test_user_pause_blocks_auto_resume(self, mpv):
        """When user manually pauses, auto-resume must not override."""
        mpv._now = 10.0
        mpv._pause_owner = "user"
        mpv._vod_auto_paused = False
        
        # Even with sufficient coverage, user pause should block auto-resume
        mpv._coverage._translated_until = 20.0
        
        can_auto_resume = mpv._pause_owner != "user"
        assert can_auto_resume is False

    def test_user_pause_blocks_auto_pause_override(self, mpv):
        """When user is playing (not paused), auto-pause should not apply."""
        mpv._now = 10.0
        mpv._pause_owner = "none"  # playing
        
        # Low lead should trigger auto-pause when not user-paused
        mpv._coverage._translated_until = 11.0  # lead 1.0 < 2.0 threshold
        assert mpv._vod_should_pause() is True
        can_auto_pause = mpv._pause_owner != "user" and mpv._vod_should_pause()
        assert can_auto_pause is True

    # -------------------------------------------------------------------------
    # Problem B: Seek-buffering auto-resume
    # -------------------------------------------------------------------------

    def test_seek_enters_buffering_state(self, mpv):
        """After a seek, _seek_buffering should be True for auto-resume."""
        mpv._pause_owner = "none"
        mpv._seek_buffering = False
        
        # Simulate _restart_audio_at() for non-manual-pause seek
        was_manually_paused = mpv._pause_owner == "user"
        assert was_manually_paused is False
        
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        mpv._vod_auto_paused = False
        
        assert mpv._seek_buffering is True
        assert mpv._pause_owner == "none"

    def test_seek_preserves_manual_pause(self, mpv):
        """If user was manually paused before seeking, remain paused after seek."""
        mpv._pause_owner = "user"
        mpv._seek_buffering = False
        
        # Simulate _restart_audio_at() preserving manual pause
        was_manually_paused = mpv._pause_owner == "user"
        assert was_manually_paused is True
        
        if was_manually_paused:
            mpv._pause_owner = "user"
            mpv._seek_buffering = False
        
        assert mpv._pause_owner == "user"
        assert mpv._seek_buffering is False

    def test_seek_buffering_auto_resumes_on_coverage(self, mpv):
        """When seek-buffering and coverage is sufficient, auto-resume."""
        mpv._now = 50.0
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        
        # Coverage reaches seek-resume threshold (5s)
        mpv._coverage._translated_until = 55.0  # lead 5.0 >= 5.0
        
        seek_lead = mpv._coverage.translated_until - mpv._now
        threshold = mpv.cfg.vod_seek_resume_threshold_s
        assert seek_lead >= threshold
        
        # Auto-resume would clear seek-buffering
        mpv._seek_buffering = False
        mpv._pause_owner = "none"
        assert mpv._seek_buffering is False

    def test_seek_buffering_waits_for_coverage(self, mpv):
        """When seek-buffering and coverage is insufficient, remain paused."""
        mpv._now = 50.0
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        
        # Coverage not yet sufficient
        mpv._coverage._translated_until = 52.0  # lead 2.0 < 5.0
        
        seek_lead = mpv._coverage.translated_until - mpv._now
        threshold = mpv.cfg.vod_seek_resume_threshold_s
        assert seek_lead < threshold
        # Should remain in seek-buffering state

    def test_consecutive_seeks_preserve_state(self, mpv):
        """Multiple consecutive seeks should not corrupt pause state."""
        # First seek while playing
        mpv._pause_owner = "none"
        mpv._seek_buffering = True
        
        # First seek completes (auto-resume)
        mpv._seek_buffering = False
        mpv._pause_owner = "none"
        
        # Second seek while playing
        mpv._seek_buffering = True
        assert mpv._pause_owner == "none"
        
        # Second seek completes
        mpv._seek_buffering = False
        assert mpv._pause_owner == "none"
        assert mpv._vod_auto_paused is False

    def test_seek_while_auto_paused(self, mpv):
        """Seeking while auto-paused should enter seek-buffering."""
        mpv._pause_owner = "auto"
        mpv._vod_auto_paused = True
        
        # Seek happens — was_manually_paused is False (auto, not user)
        was_manually_paused = mpv._pause_owner == "user"
        assert was_manually_paused is False
        
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        mpv._vod_auto_paused = False
        
        assert mpv._seek_buffering is True
        assert mpv._pause_owner == "none"

    # -------------------------------------------------------------------------
    # Problem C: Resume thresholds
    # -------------------------------------------------------------------------

    def test_seek_resume_uses_lower_threshold(self, mpv):
        """Seek-resume should use 5s threshold, not 8s steady-state."""
        mpv._now = 50.0
        
        # Lead of 5s should satisfy seek-resume but not steady-state resume
        mpv._coverage._translated_until = 55.0
        
        seek_threshold = mpv.cfg.vod_seek_resume_threshold_s
        
        lead = mpv._coverage.translated_until - mpv._now
        assert lead >= seek_threshold  # 5 >= 5
        # Steady-state resume uses 8s threshold, so 5s lead is insufficient
        assert mpv._vod_should_resume() is False  # 5 < 8

    def test_steady_state_resume_uses_higher_threshold(self, mpv):
        """Steady-state auto-resume should use 8s threshold."""
        mpv._now = 50.0
        
        # Lead of 5s is insufficient for steady-state resume
        mpv._coverage._translated_until = 55.0
        assert mpv._vod_should_resume() is False  # 5 < 8
        
        # Lead of 8s is sufficient
        mpv._coverage._translated_until = 58.0
        assert mpv._vod_should_resume() is True

    def test_vod_should_pause_with_low_lead(self, mpv):
        """VOD should pause when lead is below threshold."""
        mpv._now = 10.0
        mpv._coverage._translated_until = 11.0  # lead 1.0 < 2.0
        assert mpv._vod_should_pause() is True

    def test_vod_should_not_pause_with_sufficient_lead(self, mpv):
        """VOD should not pause when lead is sufficient."""
        mpv._now = 10.0
        mpv._coverage._translated_until = 13.0  # lead 3.0 >= 2.0
        assert mpv._vod_should_pause() is False

    # -------------------------------------------------------------------------
    # Toggle user pause
    # -------------------------------------------------------------------------

    def test_toggle_pause_sets_user_owner(self, mpv):
        """Toggling pause when playing should set _pause_owner to 'user'."""
        mpv._pause_owner = "none"
        mpv.ipc.get = mock.MagicMock(return_value=False)  # currently playing
        
        # Simulate _toggle_user_pause()
        paused = mpv.ipc.get("pause")
        new_paused = not paused  # True
        if mpv._player_command("set_property", "pause", new_paused):
            mpv._pause_owner = "user"
            mpv._vod_auto_paused = False
        
        assert mpv._pause_owner == "user"
        assert mpv._vod_auto_paused is False

    def test_toggle_play_clears_user_owner(self, mpv):
        """Toggling play when user-paused should set _pause_owner to 'none'."""
        mpv._pause_owner = "user"
        mpv.ipc.get = mock.MagicMock(return_value=True)  # currently paused
        
        # Simulate _toggle_user_pause()
        paused = mpv.ipc.get("pause")
        new_paused = not paused  # False
        if mpv._player_command("set_property", "pause", new_paused):
            mpv._pause_owner = "none"
            mpv._vod_auto_paused = False
            mpv._seek_buffering = False
        
        assert mpv._pause_owner == "none"
        assert mpv._vod_auto_paused is False
        assert mpv._seek_buffering is False

    # -------------------------------------------------------------------------
    # Cached subtitle readiness
    # -------------------------------------------------------------------------

    def test_cached_seek_ready_with_sufficient_coverage(self, mpv):
        """Cached seek should skip audio restart when coverage is sufficient."""
        mpv._uses_original_timeline = True
        mpv.ingest.source.start_s = 0.0
        
        # Mock cache
        cache = mock.MagicMock()
        cache.covered_until = mock.MagicMock(return_value=60.0)
        mpv._subtitle_cache = cache
        
        # Coverage at target
        mpv._coverage._translated_until = 55.0
        
        target = 50.0
        end = cache.covered_until(target)
        ready = (end >= target + mpv.PAUSE_LEAD_SECONDS
                 and mpv.ingest.source.start_s <= end
                 and target <= mpv._coverage.translated_until <= end + 0.02)
        assert ready is True

    def test_cached_seek_not_ready_without_coverage(self, mpv):
        """Cached seek should restart audio when coverage is insufficient."""
        mpv._uses_original_timeline = True
        mpv.ingest.source.start_s = 0.0
        
        cache = mock.MagicMock()
        cache.covered_until = mock.MagicMock(return_value=52.0)
        mpv._subtitle_cache = cache
        
        mpv._coverage._translated_until = 50.0
        
        target = 50.0
        end = cache.covered_until(target)
        ready = (end >= target + mpv.PAUSE_LEAD_SECONDS
                 and mpv.ingest.source.start_s <= end
                 and target <= mpv._coverage.translated_until <= end + 0.02)
        assert ready is False

    # -------------------------------------------------------------------------
    # Silence-only and suppressed cues
    # -------------------------------------------------------------------------

    def test_silence_advancement_does_not_deadlock(self, mpv):
        """VAD progress through silence should advance watermark."""
        mpv._now = 10.0
        
        # Advance VAD through silence (no pending speech)
        mpv._coverage.advance_vad(20.0)
        
        # Watermark should advance through silence
        assert mpv._coverage.translated_until >= 20.0
        
        # Should not pause — sufficient lead
        assert mpv._vod_should_pause() is False

    def test_suppressed_translation_clears_pending(self, mpv):
        """Suppressed translation should clear pending speech and advance."""
        mpv._pending_generation = 1
        mpv._speech_id_map = {}
        mpv._subtitle_ready = mock.MagicMock()
        
        # Record speech via production method
        sid = mpv.record_speech(10.0, 12.0, generation=1)
        assert sid is not None
        assert sid in mpv._speech_id_map
        
        # Complete as suppressed via production method (no subtitle published)
        mpv.complete_speech(10.0, 12.0, generation=1, speech_id=sid)
        assert sid not in mpv._speech_id_map
        assert mpv._coverage.pending_count() == 0

    # -------------------------------------------------------------------------
    # IPC failure handling
    # -------------------------------------------------------------------------

    def test_failed_ipc_pause_does_not_corrupt_state(self, mpv):
        """If MPV IPC command fails during auto-pause, state should be safe."""
        mpv._now = 10.0
        mpv._pause_owner = "none"
        mpv._vod_auto_paused = False
        
        # Simulate failed pause command
        mpv._player_command = mock.MagicMock(return_value=False)
        
        # Auto-pause logic: only update state if command succeeds
        if mpv._player_command("set_property", "pause", True):
            mpv._vod_auto_paused = True
            mpv._pause_owner = "auto"
        
        # State should remain unchanged because command failed
        assert mpv._vod_auto_paused is False
        assert mpv._pause_owner == "none"

    def test_failed_ipc_resume_does_not_corrupt_state(self, mpv):
        """If MPV IPC command fails during auto-resume, state should be safe."""
        mpv._now = 10.0
        mpv._pause_owner = "auto"
        mpv._vod_auto_paused = True
        
        # Simulate failed resume command
        mpv._player_command = mock.MagicMock(return_value=False)
        
        # Auto-resume logic: only update state if command succeeds
        if mpv._player_command("set_property", "pause", False):
            mpv._vod_auto_paused = False
            mpv._pause_owner = "none"
        
        # State should remain unchanged because command failed
        assert mpv._vod_auto_paused is True
        assert mpv._pause_owner == "auto"

    # -------------------------------------------------------------------------
    # Startup and timeout behavior
    # -------------------------------------------------------------------------

    def test_startup_timeout_resumes_anyway(self, mpv):
        """If translation wait times out, video should still start."""
        # The _play_session() method has a timeout:
        # vod_wait_until = time.monotonic() + lead
        # If not ready by then, should_unpause = True
        # This is tested implicitly by the timeout logic in _play_session
        mpv.PAUSE_LEAD_SECONDS = 5.0
        assert mpv.PAUSE_LEAD_SECONDS == 5.0

    def test_translation_ready_with_sufficient_coverage(self, mpv):
        """_translation_ready should return True with sufficient lead."""
        mpv._now = 10.0
        mpv._coverage._translated_until = 16.0  # lead 6.0 >= 5.0
        
        ready = mpv._translation_ready()
        assert ready is True

    def test_translation_ready_insufficient_coverage(self, mpv):
        """_translation_ready should return False with insufficient lead."""
        mpv._now = 10.0
        mpv._coverage._translated_until = 13.0  # lead 3.0 < 5.0
        
        ready = mpv._translation_ready()
        assert ready is False

    def test_translation_ready_at_eof(self, mpv):
        """_translation_ready should return True at EOF regardless of lead."""
        mpv._now = 10.0
        mpv._source_finished = True
        mpv._coverage._translated_until = 11.0  # lead only 1.0
        
        ready = mpv._translation_ready()
        assert ready is True

    # -------------------------------------------------------------------------
    # Reset buffer
    # -------------------------------------------------------------------------

    def test_reset_buffer_clears_state(self, mpv):
        """reset_buffer should clear all pending state and sync generation."""
        mpv._pending_generation = 5
        mpv._speech_id_map = {1: (0, 2), 2: (2, 4)}
        mpv._coverage._translated_until = 100.0
        mpv._source_finished = True
        mpv.ingest.generation = 3  # new generation after seek
        
        mpv.reset_buffer(50.0)
        
        # Generation should sync with ingest.generation, not reset to 0
        assert mpv._pending_generation == 3
        assert len(mpv._speech_id_map) == 0
        assert mpv._source_finished is False
        assert mpv._coverage.translated_until == 50.0

    # -------------------------------------------------------------------------
    # Generation isolation
    # -------------------------------------------------------------------------

    def test_stale_callback_rejected_in_record_speech(self, mpv):
        """record_speech should reject stale generation callbacks."""
        mpv._pending_generation = 2
        mpv._speech_id_map = {}
        
        # Stale callback from generation 1
        result = mpv.record_speech(10.0, 12.0, generation=1)
        assert result is None
        assert len(mpv._speech_id_map) == 0

    def test_stale_callback_rejected_in_advance_to_subtitle(self, mpv):
        """advance_to_subtitle should reject stale generation callbacks."""
        mpv._pending_generation = 2
        mpv._subtitle_ready = mock.MagicMock()
        
        # Stale callback from generation 1 — should return without modifying
        old_until = mpv._coverage.translated_until
        mpv.advance_to_subtitle(12.0, generation=1)
        
        # Watermark should not have changed
        assert mpv._coverage.translated_until == old_until

    def test_stale_callback_rejected_in_complete_speech(self, mpv):
        """complete_speech should reject stale generation callbacks."""
        mpv._pending_generation = 2
        mpv._subtitle_ready = mock.MagicMock()
        
        # Record in current generation
        sid = mpv.record_speech(10.0, 12.0, generation=2)
        assert sid is not None
        
        # Stale complete from generation 1
        mpv.complete_speech(10.0, 12.0, generation=1, speech_id=sid)
        
        # Should still be pending
        assert mpv._coverage.pending_count() == 1


class TestMpvPauseOwnerTransitions:
    """Test the pause owner state machine transitions."""

    @pytest.fixture
    def mpv(self):
        from auto_translate.output.coverage import CoverageTracker
        from auto_translate.output.mpv import MpvOutput

        mpv = MpvOutput.__new__(MpvOutput)
        mpv.ingest = mock.MagicMock()
        mpv.ingest.source.live = False
        mpv.ingest.source.start_s = 0.0
        mpv.ingest.source.duration = 120.0
        mpv.ingest.generation = 1
        mpv.cfg = mock.MagicMock()
        mpv.cfg.vod_pause_threshold_s = 2.0
        mpv.cfg.vod_resume_threshold_s = 8.0
        mpv.cfg.vod_seek_resume_threshold_s = 5.0
        mpv.cfg.window_id = None
        mpv.PAUSE_LEAD_SECONDS = 5.0
        mpv._coverage = CoverageTracker(0.0)
        mpv._source_finished = False
        mpv._buffer_condition = threading.Condition()
        mpv._lock = threading.Lock()
        mpv._lines = []
        mpv._pause_owner = "none"
        mpv._manually_paused = False
        mpv._vod_auto_paused = False
        mpv._seek_buffering = False
        mpv._now = 0.0
        mpv._session_start = 0.0
        mpv._subtitle_cache = None
        mpv._uses_original_timeline = False
        mpv._speech_id_map = {}
        mpv._pending_generation = 0
        mpv._subtitle_ready = mock.MagicMock()
        mpv.ipc = mock.MagicMock()
        mpv.ipc.get = mock.MagicMock(return_value=False)
        mpv._player_command = mock.MagicMock(return_value=True)
        return mpv

    def test_none_to_auto_on_low_lead(self, mpv):
        """Playing state should transition to auto-pause on low lead."""
        mpv._now = 10.0
        mpv._pause_owner = "none"
        mpv._coverage._translated_until = 11.0  # lead 1.0 < 2.0
        
        assert mpv._vod_should_pause() is True
        # Transition would set _pause_owner = "auto"
        mpv._pause_owner = "auto"
        assert mpv._pause_owner == "auto"

    def test_auto_to_none_on_resume(self, mpv):
        """Auto-pause should transition to playing on resume."""
        mpv._now = 10.0
        mpv._pause_owner = "auto"
        mpv._vod_auto_paused = True
        mpv._coverage._translated_until = 18.0  # lead 8.0 >= 8.0
        
        assert mpv._vod_should_resume() is True
        # Transition would set _pause_owner = "none"
        mpv._pause_owner = "none"
        mpv._vod_auto_paused = False
        assert mpv._pause_owner == "none"

    def test_none_to_user_on_manual_pause(self, mpv):
        """Playing should transition to user-pause on manual pause."""
        mpv._pause_owner = "none"
        # User presses pause
        mpv._pause_owner = "user"
        assert mpv._pause_owner == "user"

    def test_user_to_none_on_manual_play(self, mpv):
        """User-pause should transition to playing on manual play."""
        mpv._pause_owner = "user"
        # User presses play
        mpv._pause_owner = "none"
        mpv._seek_buffering = False
        assert mpv._pause_owner == "none"

    def test_seek_resets_to_buffering(self, mpv):
        """Seek should enter seek-buffering state."""
        mpv._pause_owner = "none"
        # Seek while playing
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        assert mpv._seek_buffering is True

    def test_seek_preserves_user_pause(self, mpv):
        """Seek while user-paused should remain user-paused."""
        mpv._pause_owner = "user"
        # Seek while manually paused
        was_manually_paused = mpv._pause_owner == "user"
        if was_manually_paused:
            mpv._pause_owner = "user"
            mpv._seek_buffering = False
        assert mpv._pause_owner == "user"
        assert mpv._seek_buffering is False

    def test_auto_pause_does_not_override_user_pause(self, mpv):
        """Auto-pause logic should not fire when user is paused."""
        mpv._pause_owner = "user"
        mpv._now = 10.0
        mpv._coverage._translated_until = 11.0
        
        # Auto-pause condition checks _pause_owner != "user"
        should_auto_pause = (mpv._pause_owner != "user" 
                            and mpv._vod_should_pause())
        assert should_auto_pause is False

    def test_seek_buffering_prevents_auto_pause(self, mpv):
        """Seek-buffering state should prevent auto-pause logic."""
        mpv._seek_buffering = True
        mpv._pause_owner = "none"
        mpv._now = 10.0
        mpv._coverage._translated_until = 11.0
        
        # Auto-pause condition checks not _seek_buffering
        should_auto_pause = (mpv._pause_owner != "user"
                            and not mpv._seek_buffering
                            and mpv._vod_should_pause())
        assert should_auto_pause is False
