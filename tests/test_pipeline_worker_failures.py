"""Pipeline worker failures must be visible, not silently starve MPV."""
import queue
import unittest

from auto_translate.pipeline import _guard_stage, _raise_stage_failure


class PipelineWorkerFailureTests(unittest.TestCase):
    def test_background_exception_is_reported_to_main_thread(self):
        errors = queue.Queue()
        def crash():
            raise RuntimeError("simulated ASR worker crash")

        self._call_guard("asr", crash, errors)
        with self.assertRaisesRegex(
            RuntimeError, "pipeline stage 'asr' failed: simulated ASR",
        ):
            _raise_stage_failure(errors)

    def test_healthy_stage_produces_no_false_failure(self):
        errors = queue.Queue()
        called = []
        _guard_stage("speaker", lambda: called.append("ok"), errors)
        self.assertEqual(called, ["ok"])
        _raise_stage_failure(errors)
        self.assertTrue(errors.empty())

    @staticmethod
    def _call_guard(name, func, errors):
        # Logging a simulated failure is expected and not a test failure.
        _guard_stage(name, func, errors)


if __name__ == "__main__":
    unittest.main()
