import json
import queue
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvError, MpvIpc, MpvOutput, _WindowsPipe, _Line, _ass_many, _after_speaker_handoff
from auto_translate.types import Subtitle


class WindowsPipeTests(unittest.TestCase):
    def test_overlapped_transport_preserves_fragmented_json_lines(self):
        chunks = iter([b'{"event":', b'"start-file"}\n{"error":"success"}\n'])
        writes = []

        class Operation:
            event = 1

            def __init__(self, data):
                self.data = data

            def GetOverlappedResult(self, wait):
                return len(self.data), 0

            def getbuffer(self):
                return self.data

        api = SimpleNamespace(
            GENERIC_READ=1, GENERIC_WRITE=2, OPEN_EXISTING=3,
            FILE_FLAG_OVERLAPPED=4,
            CreateFile=lambda *args: args,
            ReadFile=lambda handle, size, *, overlapped: (Operation(next(chunks)), 0),
            WriteFile=lambda handle, data, *, overlapped: (writes.append(data) or Operation(data), 0),
            WaitForSingleObject=lambda event, timeout: 0,
        )
        with patch.dict(sys.modules, {'_winapi': api}):
            pipe = _WindowsPipe('test-pipe')
        self.assertEqual(pipe._handle[3], 0)  # NULL security attributes must be an integer.
        self.assertEqual(pipe._handle[-2], api.FILE_FLAG_OVERLAPPED)
        pipe.write(b'command\n')
        self.assertEqual(json.loads(pipe.readline()), {'event': 'start-file'})
        self.assertEqual(json.loads(pipe.readline()), {'error': 'success'})
        self.assertEqual(writes, [b'command\n'])


class IpcTests(unittest.TestCase):
    def test_repeated_windows_launches_connect_to_their_own_player(self):
        mod = "auto_translate.output.mpv"
        ingest = SimpleNamespace(source=SimpleNamespace(start_s=0, live=False, location="video"))
        with (
            patch(mod + ".sys.platform", "win32"),
            patch(mod + ".find_mpv", return_value="mpv"),
            patch(mod + ".subprocess.Popen") as launch,
            patch(mod + "._create_kill_job"),
            patch(mod + "._gpu_memory_snapshot", return_value="unavailable"),
            patch(mod + ".MpvIpc") as connect,
            patch(mod + ".time.sleep"),
            patch(mod + ".threading.Thread.start"),
        ):
            for _ in range(2):
                MpvOutput(MpvConfig(), ingest, threading.Event())
        paths = [call.args[0] for call in connect.call_args_list]
        self.assertNotEqual(*paths)
        for path, call in zip(paths, launch.call_args_list):
            self.assertTrue(path.startswith(r"\\.\pipe\auto-translate-mpv-"))
            self.assertIn("--input-ipc-server=" + path, call.args[0])

    def test_vod_prefers_https_different_from_live_hls_video(self):
        mod = "auto_translate.output.mpv"
        with (
            patch(mod + ".find_mpv", return_value="mpv"),
            patch(mod + ".subprocess.Popen") as launch,
            patch(mod + "._gpu_memory_snapshot", return_value="unavailable"),
            patch(mod + ".MpvIpc") as ipc,
            patch(mod + ".threading.Thread.start"),
            patch(mod + ".time.sleep"),
        ):
            for live in (False, True):
                src = SimpleNamespace(start_s=0, live=live,
                    location="https://www.youtube.com/watch?v=abc")
                ingest = SimpleNamespace(source=src)
                cfg = MpvConfig()
                MpvOutput(cfg, ingest, threading.Event())
                cmd = launch.call_args.args[0]
                choice = next(x for x in cmd if x.startswith("--ytdl-format="))
                expected = cfg.ytdlp_format if live else cfg.vod_ytdlp_format
                self.assertEqual(choice, "--ytdl-format=" + expected)
                if not live:
                    self.assertTrue(expected.startswith(
                        "bv*[height<=1080][vcodec^=avc1][protocol=https]"
                    ))
                    self.assertIn("[protocol^=m3u8]", expected)
                else:
                    self.assertTrue(expected.startswith(
                        "bv*[height<=1080][vcodec^=avc1][protocol^=m3u8]"
                    ))

    def test_reader_can_wait_while_command_is_sent(self):
        replies = queue.Queue()
        seen = []

        class Transport:
            def readline(self):
                return replies.get(timeout=2)

            def write(self, data):
                self.assert_newline(data)
                command = json.loads(data)
                seen.append(command)
                replies.put(b'{"event":"start-file"}\n')
                replies.put(b'{"event":"file-loaded"}\n')
                replies.put(json.dumps({'request_id': command['request_id'], 'error': 'success', 'data': 2.5}).encode() + b'\n')

            def assert_newline(self, data):
                if not data.endswith(b'\n'):
                    raise AssertionError('IPC command lacks newline')

            def close(self):
                replies.put(b'')

        with patch('auto_translate.output.mpv.sys.platform', 'win32'), patch(
            'auto_translate.output.mpv._WindowsPipe', return_value=Transport()
        ):
            ipc = MpvIpc('test-pipe')
        try:
            self.assertEqual(ipc.get('time-pos'), 2.5)
            out = object.__new__(MpvOutput)
            out.ipc = ipc
            self.assertTrue(out._wait_for_event('start-file', 1))
            self.assertTrue(out._wait_for_event('file-loaded', 1))
            self.assertEqual(seen[0]['command'], ['get_property', 'time-pos'])
        finally:
            ipc.close()
            ipc._reader.join(2)


class OverlayTests(unittest.TestCase):
    def test_overlay_is_committed_only_after_acknowledgement(self):
        out = object.__new__(MpvOutput)
        out.cfg = MpvConfig()
        # Match the production MpvOutput fields; _render logs the first live
        # overlay only for live sources, after MPV acknowledges it.
        out.ingest = SimpleNamespace(source=SimpleNamespace(live=False))
        out._live_overlay_confirmed = False
        out._lock = threading.Lock()
        out._lines = [_Line(Subtitle(1, 3, '日本語', 'Visible translation'), 1, 3)]
        out._shown = ''
        out._lags = []
        out._logged_displayed_subtitles = set()
        calls = []

        def command(*args, **named):
            calls.append((args, named))
            if len(calls) == 1:
                raise MpvError('invalid parameter')

        out.ipc = SimpleNamespace(command=command)
        with self.assertRaises(MpvError):
            out._render(2)
        self.assertEqual(out._shown, '')
        out._render(2)
        self.assertIn('Visible translation', out._shown)
        self.assertEqual(calls[-1][1]['format'], 'ass-events')
        out._render(4)
        self.assertEqual(calls[-1][1]['format'], 'none')
        self.assertEqual(out._shown, '')



    def test_live_first_overlay_is_logged_only_after_successful_ack(self):
        out = object.__new__(MpvOutput)
        out.cfg = MpvConfig()
        out.ingest = SimpleNamespace(source=SimpleNamespace(live=True))
        out._live_overlay_confirmed = False
        out._lock = threading.Lock()
        out._lines = [_Line(Subtitle(1, 3, "日本語", "Visible translation"), 1, 3)]
        out._shown = ""
        out._logged_displayed_subtitles = set()
        out._lags = []
        attempts = []

        def command(*args, **named):
            attempts.append((args, named))
            if len(attempts) == 1:
                raise MpvError("overlay rejected")

        out.ipc = SimpleNamespace(command=command)
        with patch("auto_translate.output.mpv.log") as log:
            with self.assertRaises(MpvError):
                out._render(2)
            self.assertFalse(out._live_overlay_confirmed)
            log.info.assert_not_called()

            out._render(2)
            self.assertTrue(out._live_overlay_confirmed)
            self.assertIn("Visible translation", out._shown)
            log.info.assert_called_once()

            out._render(2)
            log.info.assert_called_once()


class MpvShutdownTests(unittest.TestCase):
    def test_normal_mpv_close_does_not_report_ipc_as_playback_failure(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = SimpleNamespace(poll=lambda: 0)

        def closed_player(show_hint):
            raise BrokenPipeError("MPV window closed")

        out._play_session = closed_player
        with patch("auto_translate.output.mpv.log") as log:
            out._run()
            self.assertTrue(out.stop.is_set())
            log.info.assert_called_once()
            log.exception.assert_not_called()
            log.error.assert_not_called()

    def test_pipe_can_close_before_process_exit_is_visible(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = Mock()
        out._proc.poll.return_value = None
        out._proc.wait.return_value = 0
        out._play_session = Mock(side_effect=BrokenPipeError("pipe closing"))
        with patch("auto_translate.output.mpv.log") as log:
            out._run()
            log.exception.assert_not_called()
            log.error.assert_not_called()
        self.assertTrue(out.stop.is_set())
        out._proc.wait.assert_called_once_with(timeout=0.75)

    def test_final_frame_poll_tolerates_clean_window_close(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = Mock()
        out._proc.poll.return_value = None
        out._proc.wait.return_value = 0
        out.ipc = Mock()
        out.ipc.get.side_effect = BrokenPipeError("pipe closing")
        out.drain()
        out._proc.wait.assert_called_once_with(timeout=0.75)

    def test_unexpected_mpv_exit_exposes_exit_code(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._proc = SimpleNamespace(poll=lambda: 2)

        def crashed_player(show_hint):
            raise BrokenPipeError("MPV IPC closed unexpectedly")

        out._play_session = crashed_player
        with patch("auto_translate.output.mpv.log") as log:
            out._run()
            self.assertTrue(out.stop.is_set())
            log.error.assert_called_once()
            self.assertIn("code %s", log.error.call_args.args[0])
            self.assertEqual(log.error.call_args.args[1], 2)


class TranslationBufferTests(unittest.TestCase):
    def test_live_buffer_defaults_to_eight_seconds(self):
        self.assertEqual(MpvConfig().delay_s, 8.0)

    def test_live_waits_for_processing_up_to_ten_seconds(self):
        out = object.__new__(MpvOutput)
        out.cfg = MpvConfig()
        out.ingest = SimpleNamespace(first_pts_at=100.0)
        out._buffer_condition = threading.Condition()
        out._translated_until = 0.0
        out._session_start = 0.0
        self.assertFalse(out._live_playback_ready(107.9))
        self.assertFalse(out._live_playback_ready(108.1))
        out.advance(4.99)
        self.assertFalse(out._live_playback_ready(108.1))
        out.advance(5.0)
        self.assertTrue(out._live_playback_ready(108.1))
        out._translated_until = 0.0
        self.assertTrue(out._live_playback_ready(110.0))

    def test_live_clock_maps_large_mpv_pts_to_audio_relative_seconds(self):
        out = object.__new__(MpvOutput)
        out._uses_original_timeline = False
        out._mpv_anchor = 5400.0
        out._clock_source_anchor = 0.0
        self.assertEqual(out._source_time_from_mpv(5403.5), 3.5)
        self.assertEqual(out._mpv_time_from_source(3.5), 5403.5)

    def test_vsr_health_is_noop_before_optional_state_initialized(self):
        # MPV may be partially initialized in tests or during teardown.
        # Missing optional VSR state must not crash playback/session control.
        out = object.__new__(MpvOutput)
        out._check_vsr_health()
        out._vsr_active = False
        out._check_vsr_health()
        out._vsr_active = True
        out._playback_started = False
        out._check_vsr_health()

    def test_manual_play_bypasses_prebuffer_without_forced_pause(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out.cfg = MpvConfig()
        out._direct_video_loaded = True
        out._subtitle_ready = threading.Event()
        out._lock = threading.Lock()
        out._lines = []
        out._buffer_condition = threading.Condition()
        out._source_finished = False
        out._translated_until = 0.0
        out._now = None
        out._clock_last_sample_at = 0.0
        out._clock_last_sample_pos = None
        out._logged_displayed_subtitles = set()
        out._lags = []
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(
                start_s=0.0, live=False, duration=200.0, timeline=None
            ),
            generation=0,
            first_pts=0.0,
            first_pts_ready=SimpleNamespace(wait=lambda timeout: True),
            video_failed=threading.Event(),
        )

        commands = []
        out.ipc = SimpleNamespace(
            command=lambda *args, **kwargs: commands.append(args),
            get=lambda key: False if key == 'pause' else (
                0.0 if key == 'time-pos' else None
            ),
        )
        out._translation_ready = lambda: False
        out._render = lambda now: None
        out._replay_requested = lambda: False
        out._jump_request = lambda: 0

        def exit_after_tick(timeout):
            out.stop.set()

        out._subtitle_ready.wait = exit_after_tick
        out._play_session(show_hint=False)
        self.assertFalse(
            any(args[:3] == ('set_property', 'pause', True) for args in commands),
            'Manual Play must not be immediately reversed by MPV controller',
        )

    def test_vod_never_has_a_continuous_playback_low_watermark(self):
        # Fast ingest and the initial/seek buffer should not force recurring
        # stop/start playback when translation runs slower than real-time.
        self.assertFalse(hasattr(MpvOutput, 'PAUSE_LOW_WATERMARK_SECONDS'))

    def test_five_seconds_of_translated_coverage_required(self):
        out = object.__new__(MpvOutput)
        out._buffer_condition = threading.Condition()
        out._now = 120.0
        out._translated_until = 124.9
        out._source_finished = False

        self.assertFalse(out._translation_ready())
        out.advance(125.0)
        self.assertTrue(out._translation_ready())
        out._now = 126.0
        self.assertFalse(out._translation_ready())
        out.finish()
        self.assertTrue(out._translation_ready())

    def test_seek_resets_old_translation_coverage_immediately(self):
        out = object.__new__(MpvOutput)
        out._buffer_condition = threading.Condition()
        out._lock = threading.Lock()
        out._now = 120.0
        out._translated_until = 900.0
        out._source_finished = True
        out._lines = [_Line(Subtitle(119, 121, '日本語', 'Old'), 119, 121)]
        out._shown = 'Old'
        out._mpv_anchor = 120.0
        out._clock_source_anchor = 120.0
        out._pending_seek_offset = None
        out._clock_sample_logged = True
        out._last_direct_pos = 120.0
        out._last_direct_sample_at = 0.0
        calls = []
        out.ipc = SimpleNamespace(
            get=lambda prop: 120.0 if prop == 'time-pos' else None,
            command=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        restarts = []
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(duration=3600.0),
            request_restart=lambda pos: restarts.append(pos),
            go=lambda: restarts.append('go'),
        )

        self.assertTrue(out._restart_audio_at(500.0, seek_video=False))
        self.assertEqual(restarts, [500.0, 'go'])
        self.assertEqual(out._translated_until, 500.0)
        self.assertFalse(out._source_finished)
        self.assertIsNone(out._now)
        self.assertFalse(out._lines)
        self.assertFalse(out._translation_ready())
        self.assertTrue(any(args[:1] == ('osd-overlay',) for args, _ in calls))

    def test_keyboard_seek_starts_audio_before_confirming_video_position(self):
        out = object.__new__(MpvOutput)
        out._uses_original_timeline = True
        out._lock = threading.Lock()
        out._lines = []
        out.stop = threading.Event()
        out._buffer_condition = threading.Condition()
        out._subtitle_ready = threading.Event()
        events = []
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(duration=900.0),
            request_restart=lambda target: events.append(("restart", target)),
            go=lambda: events.append(("go",)),
        )

        def position(prop):
            events.append(("position",))
            return 500.0

        out.ipc = SimpleNamespace(
            command=lambda *args, **kwargs: events.append(args),
            get=position,
        )
        self.assertTrue(out._restart_audio_at(500.0, seek_video=True))
        self.assertLess(events.index(("seek", 500.0, "absolute+exact")), events.index(("go",)))
        self.assertLess(events.index(("go",)), events.index(("position",)))
        out.advance(504.99)
        self.assertFalse(out._subtitle_ready.is_set())
        out.advance(505.0)
        self.assertTrue(out._subtitle_ready.is_set())

    def test_session_anchor_uses_new_ingest_start_after_seek(self):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out.cfg = MpvConfig()
        out._direct_video_loaded = True
        out._now = None
        out._buffer_condition = threading.Condition()
        out._translated_until = 0.0
        out._source_finished = False
        source = SimpleNamespace(start_s=10.0, live=False, duration=3600.0)

        def first_audio_ready(timeout):
            source.start_s = 600.0
            out.ingest.first_pts = 600.0
            return True

        out.ingest = SimpleNamespace(
            source=source,
            generation=1,
            first_pts=None,
            first_pts_ready=SimpleNamespace(wait=first_audio_ready),
        )

        def get(prop):
            if prop == 'time-pos':
                out.stop.set()
                return 600.0
            return None

        out.ipc = SimpleNamespace(get=get)
        out._play_session(show_hint=False)
        self.assertEqual(out._session_start, 600.0)
        self.assertEqual(out._clock_source_anchor, 600.0)
        self.assertEqual(out._mpv_anchor, 600.0)

class CachedSeekTests(unittest.TestCase):
    """Rewinding a translated VOD region must not freeze MPV for audio restart."""

    def make_output(self, *, cached_until=120.0):
        out = object.__new__(MpvOutput)
        out.stop = threading.Event()
        out._direct_video_loaded = True
        out._uses_original_timeline = True
        out._subtitle_cache = SimpleNamespace(
            covered_until=lambda position: cached_until
        )
        out._now = 112.0
        out._clock_source_anchor = 0.0
        out._mpv_anchor = 0.0
        out._last_direct_pos = 112.0
        out._last_direct_sample_at = 0.0
        out.ingest = SimpleNamespace(
            source=SimpleNamespace(live=False, duration=500.0),
        )
        calls = []
        out.ipc = SimpleNamespace(
            get=lambda key: 102.0 if key == "time-pos" else None,
            command=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        renders = []
        out._render = lambda now: renders.append(now)
        restarts = []
        out._restart_audio_at = lambda at, *, seek_video: (
            restarts.append((at, seek_video)) or True
        )
        return out, calls, renders, restarts

    def test_ten_second_backseek_replays_cached_subtitles_without_pause(self):
        out, calls, renders, restarts = self.make_output()
        self.assertFalse(out._jump(-10))
        self.assertEqual(restarts, [])
        self.assertIn((("seek", 102.0, "absolute+exact"), {}), calls)
        self.assertFalse(
            any(args[:2] == ("set_property", "pause") for args, _ in calls)
        )
        self.assertEqual(out._now, 102.0)
        self.assertEqual(out._last_direct_pos, 102.0)
        self.assertEqual(renders, [102.0])

    def test_original_video_does_not_accumulate_offset_across_seeks(self):
        out, calls, renders, restarts = self.make_output(cached_until=500.0)
        # A previous audio restart can create unrelated anchor values.
        out._clock_source_anchor = 110.0
        out._mpv_anchor = 112.5
        self.assertEqual(out._source_time_from_mpv(112.0), 112.0)
        self.assertEqual(out._mpv_time_from_source(112.0), 112.0)

        # Simulate MPV landing each native cached seek immediately.
        position = [112.0]
        out.ipc.get = lambda name: position[0] if name == "time-pos" else None

        def command(*args, **kwargs):
            calls.append((args, kwargs))
            if args[0] == "seek":
                position[0] = float(args[1])

        out.ipc.command = command
        for step, expected in [(-10, 102.0), (-10, 92.0), (10, 102.0),
                               (-10, 92.0), (20, 112.0), (10, 122.0)]:
            self.assertFalse(out._jump(step))
            self.assertAlmostEqual(position[0], expected)
            self.assertAlmostEqual(out._now, expected)
            self.assertAlmostEqual(out._source_time_from_mpv(position[0]), expected)
        self.assertEqual(restarts, [])
        self.assertEqual(renders, [102.0, 92.0, 102.0, 92.0, 112.0, 122.0])
        self.assertFalse(
            any(args[:2] == ("set_property", "pause") for args, _ in calls)
        )

    def test_uncached_original_seek_uses_absolute_video_timestamp(self):
        out, calls, renders, restarts = self.make_output(cached_until=103.0)
        out._clock_source_anchor = 100.0
        out._mpv_anchor = 110.0
        self.assertTrue(out._jump(-10))
        self.assertEqual(restarts, [(102.0, True)])
        self.assertEqual(out._mpv_time_from_source(102.0), 102.0)

    def test_uncached_backseek_still_restarts_audio(self):
        out, calls, renders, restarts = self.make_output(cached_until=105.0)
        self.assertTrue(out._jump(-10))
        self.assertEqual(restarts, [(102.0, True)])
        self.assertEqual(calls, [])
        self.assertEqual(renders, [])

    def test_cached_seek_respects_existing_mpv_source_anchor(self):
        out, calls, renders, restarts = self.make_output(cached_until=450.0)
        out._now = 410.0
        out._clock_source_anchor = 400.0
        out._mpv_anchor = 450.0
        # Only the fMP4 fallback has a non-identity source/MPV clock.
        out._uses_original_timeline = False
        # Source 400 maps to MPV 450, not absolute 400.
        out.ipc.get = lambda name: 450.0 if name == "time-pos" else None
        self.assertFalse(out._jump(-10))
        self.assertIn((("seek", 450.0, "absolute+exact"), {}), calls)
        self.assertEqual(out._now, 400.0)
        self.assertEqual(renders, [400.0])
        self.assertEqual(restarts, [])



class TurnSubtitleDisplayTests(unittest.TestCase):
    def test_confirmed_handoff_hides_previous_speakers_while_preserving_current_cues(self):
        previous = _Line(
            Subtitle(0, 1.7, "speaker one", "I was going to—"),
            0, 4.7,
        )
        interrupting = _Line(
            Subtitle(1.7, 3.0, "speaker two", "Wait!", speaker_handoff=True),
            1.7, 6.0,
        )
        continuing = _Line(
            Subtitle(3.0, 4.0, "speaker two", "Come over here."),
            3.0, 7.0,
        )
        self.assertEqual(
            _after_speaker_handoff([previous, interrupting, continuing]),
            [interrupting, continuing],
        )
        self.assertEqual(_after_speaker_handoff([previous]), [previous])

    def test_newer_handoff_replaces_previous_speaker_stack(self):
        first = _Line(Subtitle(0, 1, "a", "First"), 0, 4)
        second = _Line(
            Subtitle(1, 2, "b", "Second", speaker_handoff=True), 1, 5)
        third = _Line(
            Subtitle(2, 3, "c", "Third", speaker_handoff=True), 2, 6)
        self.assertEqual(_after_speaker_handoff([first, second, third]), [third])

    def test_normal_three_line_stacking_remains_available(self):
        lines = [
            _Line(Subtitle(float(i), float(i+1), "ja", str(i)),
                  float(i), float(i+4))
            for i in range(3)
        ]
        self.assertEqual(_after_speaker_handoff(lines), lines)

    def test_ass_overlay_stacks_english_without_identity_labels(self):
        body = _ass_many([
            Subtitle(5, 6, "ja", "Watch out!"),
            Subtitle(4, 5, "ja", "On my way."),
            Subtitle(3, 4, "ja", "Unknown"),
        ], MpvConfig())
        self.assertIn("Watch out!", body)
        self.assertIn("On my way.", body)
        self.assertIn("Unknown", body)
        self.assertNotIn("A: ", body)
        self.assertNotIn("B: ", body)
        self.assertLess(body.index("Watch out!"), body.index("On my way."))


if __name__ == '__main__':
    unittest.main()
