"""GPU pressure changes PLaMo placement only between serialized requests."""

import unittest
from unittest.mock import Mock, patch

import httpx

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.base import TranslatorUnavailable
from auto_translate.translate.plamo import PlamoTranslator


class OwnedProcess:
    def __init__(self, pid=100, returncode=None):
        self.pid = pid
        self.returncode = returncode
        self.operations = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.operations.append("terminate")
        self.returncode = -15

    def wait(self, timeout):
        self.operations.append(("wait", timeout))
        return self.returncode

    def kill(self):
        self.operations.append("kill")
        self.returncode = -9


def allocation_error(detail="CUDA error: out of memory"):
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(500, text=detail, request=request)
    return httpx.HTTPStatusError("allocation failed", request=request, response=response)


class PlamoGpuReliefTests(unittest.TestCase):
    def translator(self, layers=12, *, owned=True):
        with patch.object(PlamoTranslator, "_start_server"), patch.object(PlamoTranslator, "_warmup"):
            translator = PlamoTranslator(TranslateConfig(plamo_gpu_layers=layers), Glossary({}))
        translator.process = OwnedProcess() if owned else None
        self.addCleanup(translator.close)
        return translator

    @staticmethod
    def install_restart(translator):
        placements = []

        def restart():
            placements.append(translator.cfg.plamo_gpu_layers)
            translator.process = OwnedProcess(100 + len(placements))

        translator._start_server = Mock(side_effect=restart)
        return placements

    def test_sampler_schedules_without_stopping_an_active_request(self):
        translator = self.translator()
        first = translator.process
        placements = self.install_restart(translator)

        def inference(*_args):
            self.assertTrue(translator.request_gpu_relief())
            self.assertEqual(first.operations, [])
            self.assertEqual(placements, [])
            return "Wait here."

        translator._translate_cue = Mock(side_effect=inference)
        self.assertEqual(translator.translate("待って", []), "Wait here.")
        self.assertEqual(first.operations, [])
        translator._translate_cue.side_effect = None
        translator._translate_cue.return_value = "Understood."
        self.assertEqual(translator.translate("了解", []), "Understood.")
        self.assertEqual(first.operations, ["terminate", ("wait", 2)])
        self.assertEqual(placements, [6])

    def test_repeated_pressure_reaches_cpu_with_no_model_or_context_change(self):
        translator = self.translator()
        model = translator.cfg.plamo_hf_model
        placements = self.install_restart(translator)
        translator._translate_cue = Mock(return_value="Yes.")
        history = [("待って", "Wait.")]
        for _ in range(4):
            self.assertTrue(translator.request_gpu_relief())
            self.assertTrue(translator.request_gpu_relief())  # coalesce one sample interval
            self.assertEqual(translator.translate("うん", history), "Yes.")
        self.assertEqual(placements, [6, 3, 1, 0])
        self.assertFalse(translator.request_gpu_relief())
        self.assertEqual(translator.cfg.plamo_hf_model, model)
        self.assertTrue(all(call.args == ("うん", history)
                            for call in translator._translate_cue.call_args_list))
        self.assertFalse(translator.client.is_closed)

    def test_cuda_allocation_error_replays_same_cue_after_smaller_offload(self):
        translator = self.translator()
        first = translator.process
        placements = self.install_restart(translator)
        history = [("待って", "Wait.")]
        translator._translate_cue = Mock(side_effect=[allocation_error(), "Yes."])
        self.assertEqual(translator.translate("うん", history), "Yes.")
        self.assertEqual(placements, [6])
        self.assertEqual(first.operations, ["terminate", ("wait", 2)])
        self.assertEqual(translator._translate_cue.call_args_list[0],
                         translator._translate_cue.call_args_list[1])

    def test_failed_gpu_reload_reduces_until_cpu_starts(self):
        translator = self.translator(layers=4)
        placements = []

        def restart():
            placements.append(translator.cfg.plamo_gpu_layers)
            if translator.cfg.plamo_gpu_layers:
                raise RuntimeError("CUDA allocation failed during model load")
            translator.process = OwnedProcess(200)

        translator._start_server = Mock(side_effect=restart)
        translator._translate_cue = Mock(return_value="Yes.")
        translator.request_gpu_relief()
        self.assertEqual(translator.translate("うん", []), "Yes.")
        self.assertEqual(placements, [2, 1, 0])

    def test_persistent_allocation_failure_is_bounded_and_explicit(self):
        translator = self.translator(layers=2)
        placements = self.install_restart(translator)
        translator._translate_cue = Mock(side_effect=allocation_error())
        with self.assertRaisesRegex(TranslatorUnavailable, "after CPU fallback"):
            translator.translate("うん", [])
        self.assertEqual(placements, [1, 0])
        self.assertEqual(translator._translate_cue.call_count, 3)

    def test_external_endpoint_is_never_reconfigured_on_pressure_or_oom(self):
        translator = self.translator(owned=False)
        translator._translate_cue = Mock(side_effect=allocation_error())
        translator._start_server = Mock()
        self.assertFalse(translator.request_gpu_relief())
        with self.assertRaises(httpx.HTTPStatusError):
            translator.translate("うん", [])
        translator._start_server.assert_not_called()
        self.assertEqual(translator.cfg.plamo_gpu_layers, 12)

    def test_nonallocation_server_error_does_not_change_placement(self):
        translator = self.translator()
        translator._translate_cue = Mock(side_effect=allocation_error("Model is still loading"))
        translator._start_server = Mock()
        with self.assertRaises(httpx.HTTPStatusError):
            translator.translate("うん", [])
        self.assertEqual(translator.cfg.plamo_gpu_layers, 12)
        self.assertEqual(translator.process.operations, [])
        translator._start_server.assert_not_called()

    def test_owned_crash_recovery_reduces_placement_but_remains_bounded(self):
        translator = self.translator()
        translator.process.returncode = 1
        placements = self.install_restart(translator)
        translator._translate_cue = Mock(side_effect=[httpx.ReadError("closed"), "Yes."])
        self.assertEqual(translator.translate("うん", []), "Yes.")
        self.assertEqual(placements, [6])
        translator.process.returncode = 1
        translator._translate_cue.side_effect = httpx.ReadError("closed again")
        with self.assertRaisesRegex(TranslatorUnavailable, "again after recovery"):
            translator.translate("うん", [])
        self.assertEqual(placements, [6])


if __name__ == "__main__":
    unittest.main()
