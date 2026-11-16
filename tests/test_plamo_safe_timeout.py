"""Safe Playback must retire owned abandoned inference before releasing admission."""

from contextlib import contextmanager
import subprocess
import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.config import SafePlaybackConfig, TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.safe_playback import SafeInferenceGate
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.translate.plamo import PlamoTranslator


class OwnedServer:
    def __init__(self, pid=100, *, stubborn=False, unkillable=False):
        self.pid = pid
        self.returncode = None
        self.stubborn = stubborn
        self.unkillable = unkillable
        self.operations = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.operations.append("terminate")
        if not self.stubborn:
            self.returncode = -15

    def wait(self, timeout):
        self.operations.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode

    def kill(self):
        self.operations.append("kill")
        if not self.unkillable:
            self.returncode = -9


class SafePlamoTimeoutTests(unittest.TestCase):
    def translator(self, process, *, safe=True):
        with patch.object(PlamoTranslator, "_start_server"), \
                patch.object(PlamoTranslator, "_warmup"):
            translator = PlamoTranslator(TranslateConfig(), Glossary({}))
        translator.client.close()
        translator.client = Mock()
        translator.process = process
        translator.safe_playback = safe
        self.addCleanup(translator.close)
        return translator

    @staticmethod
    def response(text="Wait here."):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": text}}]}
        return response

    def test_owned_timeout_stops_inference_before_gpu_admission_is_released(self):
        process = OwnedServer()
        translator = self.translator(process)
        translator.client.post.side_effect = httpx.ReadTimeout("generation stalled")
        lease_checks = []
        parent = self

        class CheckingGate(SafeInferenceGate):
            @contextmanager
            def operation(self, *args, **kwargs):
                with super().operation(*args, **kwargs):
                    try:
                        yield
                    finally:
                        # This runs while the actual admission lease is held.
                        parent.assertIsNotNone(process.poll())
                        lease_checks.append(True)

        gate = CheckingGate(SafePlaybackConfig(enabled=True))
        with self.assertRaises(httpx.ReadTimeout):
            with gate.operation("plamo"):
                translator._translate_one("待って", [])
        self.assertEqual(lease_checks, [True])
        self.assertFalse(translator._closed)
        self.assertIs(translator.process, process)
        self.assertEqual(translator.client.post.call_count, 1)
        self.assertEqual(process.operations, ["terminate", ("wait", 2)])
        with gate.operation("next-model"):
            pass

    def test_owned_timeout_escalates_to_bounded_kill(self):
        process = OwnedServer(stubborn=True)
        translator = self.translator(process)
        translator.client.post.side_effect = httpx.ReadTimeout("stalled")
        with self.assertRaises(httpx.ReadTimeout):
            translator._chat("待って", [], timeout=1)
        self.assertEqual(process.operations, ["terminate", ("wait", 2), "kill", ("wait", 2)])
        self.assertEqual(process.poll(), -9)

    def test_owned_server_recovers_at_most_once_after_safe_timeouts(self):
        first, second = OwnedServer(100), OwnedServer(101)
        translator = self.translator(first)
        translator.client.post.side_effect = [
            httpx.ReadTimeout("first stalled generation"), self.response(),
            httpx.ReadTimeout("second stalled generation"),
        ]
        translator._start_server = Mock(side_effect=lambda: setattr(translator, "process", second))
        gate = SafeInferenceGate(SafePlaybackConfig(enabled=True))
        with gate.operation("plamo"):
            self.assertEqual(translator.translate("待って", []), "Wait here.")
        self.assertIsNotNone(first.poll())
        self.assertIsNone(second.poll())
        self.assertTrue(translator._restart_used)
        with self.assertRaisesRegex(TranslatorUnavailable, "again after recovery"):
            with gate.operation("plamo"):
                translator.translate("待って", [])
        self.assertIsNotNone(second.poll())
        translator._start_server.assert_called_once()
        self.assertEqual(translator.client.post.call_count, 3)

    def test_external_timeout_never_kills_or_replays_request(self):
        translator = self.translator(None)
        translator.client.post.side_effect = httpx.ReadTimeout("external inference timed out")
        translator._start_server = Mock()
        with patch("auto_translate.translate.plamo.record_event") as diagnostic, \
                patch("auto_translate.translate.plamo.owned_popen") as spawn:
            with self.assertRaises(httpx.ReadTimeout):
                translator.translate("待って", [])
        self.assertIsNone(translator.process)
        self.assertEqual(translator.client.post.call_count, 1)
        translator._start_server.assert_not_called()
        spawn.assert_not_called()
        diagnostic.assert_called_once_with(
            "inference_cancellation_unverified", worker="plamo", owned=False,
            reason="external_server_timeout", automatic_retry=False)

    def test_owned_validation_timeout_can_return_first_result_after_confirmed_stop(self):
        process = OwnedServer()
        translator = self.translator(process)
        translator.client.post.side_effect = [self.response("That's bad!"), httpx.ReadTimeout("validation stalled")]
        self.assertEqual(translator._translate_one("すごい!", []), "That's bad!")
        self.assertIsNotNone(process.poll())
        self.assertEqual(translator.client.post.call_count, 2)
        self.assertFalse(translator._closed)

    def test_external_validation_timeout_does_not_hide_abandoned_inference(self):
        translator = self.translator(None)
        translator.client.post.side_effect = [self.response("That's bad!"), httpx.ReadTimeout("validation stalled")]
        with self.assertRaises(httpx.ReadTimeout):
            translator._translate_one("すごい!", [])
        self.assertEqual(translator.client.post.call_count, 2)

    def test_default_mode_retains_existing_transient_timeout_retry(self):
        process = OwnedServer()
        translator = self.translator(process, safe=False)
        translator.client.post.side_effect = [httpx.ReadTimeout("temporary"), self.response()]
        self.assertEqual(translator.translate("待って", []), "Wait here.")
        self.assertEqual(translator.client.post.call_count, 2)
        self.assertEqual(process.operations, [])
        self.assertIsNone(process.poll())

    def test_unverified_owned_cleanup_surfaces_terminal_failure_without_replay(self):
        process = OwnedServer(stubborn=True, unkillable=True)
        translator = self.translator(process)
        translator.client.post.side_effect = httpx.ReadTimeout("stalled")
        # Avoid asking the normal close fixture to wait again for an unkillable
        # mocked process after this test has checked the terminal error.
        self.addCleanup(setattr, process, "returncode", -9)
        with patch("auto_translate.translate.plamo.record_event") as diagnostic:
            with self.assertRaisesRegex(TranslatorUnavailable, "cleanup failed"):
                translator._translate_one("待って", [])
        self.assertEqual(translator.client.post.call_count, 1)
        self.assertFalse(translator._safe_timeout_stopped)
        diagnostic.assert_called_once_with(
            "inference_cancellation_unverified", worker="plamo", owned=True,
            reason="TimeoutExpired", automatic_retry=False)


if __name__ == "__main__":
    unittest.main()
