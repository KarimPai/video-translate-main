"""Deterministic offline diagnostics, metrics and export privacy regressions."""
import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from auto_translate.config import GpuBudgetConfig
from auto_translate.gpu_budget import GpuBudget, gpu_memory_gb
from auto_translate import gpu_budget, runtime_diagnostics as diagnostics


class GpuMonitoringTests(unittest.TestCase):
    def test_nvml_initialized_once_and_supported_fields_measured(self):
        nvml = Mock(NVML_TEMPERATURE_GPU=0)
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=3_000_000_000, total=8_000_000_000)
        nvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=73)
        nvml.nvmlDeviceGetTemperature.return_value = 65
        nvml.nvmlDeviceGetPowerUsage.return_value = 123000
        with patch.object(gpu_budget, "_NVML", gpu_budget._NVML_UNTRIED), \
                patch.object(gpu_budget.importlib, "import_module", return_value=nvml), \
                patch.object(gpu_budget.subprocess, "check_output") as smi:
            budget = GpuBudget(GpuBudgetConfig())
            budget.refresh()
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["used_gb"], 3)
            self.assertEqual(snapshot["free_gb"], 5)
            self.assertEqual(snapshot["utilization_pct"], 73)
            self.assertEqual(snapshot["temperature_c"], 65)
            self.assertEqual(snapshot["power_w"], 123)
            self.assertEqual(snapshot["source"], "nvml")
            nvml.nvmlInit.assert_called_once()
            smi.assert_not_called()

    def test_nvml_unsupported_metrics_are_unknown(self):
        nvml = Mock(NVML_TEMPERATURE_GPU=0)
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=1_000_000_000, total=8_000_000_000)
        nvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError("unsupported on WDDM")
        nvml.nvmlDeviceGetTemperature.side_effect = RuntimeError("unsupported")
        nvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("unsupported")
        with patch.object(gpu_budget, "_NVML", nvml):
            snapshot = GpuBudget(GpuBudgetConfig()).snapshot()
        for key in ("utilization_pct", "temperature_c", "power_w"):
            self.assertIsNone(snapshot[key])
        self.assertEqual(snapshot["used_gb"], 1)

    def test_no_gpu_tools_remains_unknown_and_does_not_crash(self):
        with patch.object(gpu_budget, "_NVML", None), \
                patch.object(gpu_budget.subprocess, "check_output", side_effect=FileNotFoundError):
            budget = GpuBudget(GpuBudgetConfig())
            self.assertFalse(budget.admits(-1))
        self.assertEqual(budget.pressure_level, "unknown")
        self.assertIsNone(budget.snapshot()["used_gb"])

    def test_missing_nvml_uses_existing_hidden_smi_fallback(self):
        with patch.object(gpu_budget, "_NVML", None), \
                patch.object(gpu_budget.subprocess, "check_output", return_value="4096, 8192\n") as read:
            used, total = gpu_memory_gb()
        self.assertAlmostEqual(used, 4096 * 1048576 / 1e9)
        self.assertAlmostEqual(total, 8192 * 1048576 / 1e9)
        self.assertEqual(read.call_args.kwargs["timeout"], 2)

    def test_snapshot_never_polls_gpu(self):
        with patch.object(gpu_budget, "gpu_memory_gb", return_value=(3.0, 8.0)) as read:
            budget = GpuBudget(GpuBudgetConfig())
            read.reset_mock()
            for _ in range(10):
                budget.snapshot()
            read.assert_not_called()

    def test_nonfinite_driver_values_are_unknown(self):
        nvml = Mock(NVML_TEMPERATURE_GPU=0)
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=1_000_000_000, total=float("inf"))
        nvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=float("nan"))
        nvml.nvmlDeviceGetTemperature.return_value = float("inf")
        nvml.nvmlDeviceGetPowerUsage.return_value = float("inf")
        with patch.object(gpu_budget, "_NVML", nvml), \
                patch.object(gpu_budget.subprocess, "check_output", side_effect=FileNotFoundError):
            snapshot = GpuBudget(GpuBudgetConfig()).snapshot()
        for key in ("used_gb", "total_gb", "utilization_pct", "temperature_c", "power_w"):
            self.assertIsNone(snapshot[key])


class DiagnosticsTests(unittest.TestCase):
    def create(self, directory, **kwargs):
        cfg = SimpleNamespace(enabled=True, max_bytes=65536, backups=2, **kwargs)
        return diagnostics.RuntimeDiagnostics(cfg, Path(directory))

    def test_nested_metrics_merge_and_snapshot_is_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.create(directory)
            recorder.update(stages={"haas_s": 1.5}, counters={"expired": 1})
            recorder.update(stages={"whisper_s": 0.9})
            recorder.increment("expired", 2)
            snapshot = recorder.snapshot()
            self.assertEqual(snapshot["stages"], {"haas_s": 1.5, "whisper_s": 0.9})
            self.assertEqual(snapshot["counters"]["expired"], 3)
            snapshot["stages"].clear()
            self.assertIn("haas_s", recorder.snapshot()["stages"])
            recorder.close()

    def test_log_records_are_structured_and_rotation_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.create(directory)
            for index in range(200):
                recorder.record("model_failure", number=index, detail="x" * 4000)
            recorder.close()
            paths = list(Path(directory).glob("diagnostics.jsonl*"))
            self.assertLessEqual(len(paths), 3)
            self.assertGreaterEqual(len(paths), 2)
            for path in paths:
                self.assertLess(path.stat().st_size, 70000)
                for line in path.read_text(encoding="utf-8").splitlines():
                    self.assertIsInstance(json.loads(line), dict)

    def test_capture_application_logs_and_redact_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.create(directory)
            handler = recorder._handler
            event = logging.LogRecord("auto_translate.test", logging.ERROR, "", 0,
                                      "model timeout token=SECRET path=C:\\Users\\Alice\\logs", (), None)
            handler.emit(event)
            recorder.close()
            text = (Path(directory) / "diagnostics.jsonl").read_text(encoding="utf-8")
            self.assertIn("application_log", text)
            self.assertNotIn("SECRET", text)
            self.assertNotIn("Alice", text)

    def test_simultaneous_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.create(directory)
            threads = [threading.Thread(target=recorder.close) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
            self.assertFalse(recorder._thread.is_alive())

    def test_previous_interruption_is_reported_without_claiming_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            path.write_text(json.dumps({"session": "old", "event": "model_start"}) + "\n")
            recorder = self.create(directory)
            recorder.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(any(item.get("event") == "unclean_shutdown_suspected" for item in records))
            self.assertTrue(any("cause unknown" in item.get("reason", "") for item in records))
            recorder = self.create(directory)
            recorder.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(item.get("event") == "unclean_shutdown_suspected" for item in records), 1)

    def test_full_queue_still_drains_and_closes_writer_on_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.create(directory)
            entered, unblock = threading.Event(), threading.Event()

            def blocked_write(record):
                entered.set()
                unblock.wait(timeout=3)

            # Wait for the session marker before replacing the sink callback.
            recorder._queue.join()
            with patch.object(recorder._sink, "emit", side_effect=blocked_write):
                recorder.record("blocked_disk")
                self.assertTrue(entered.wait(timeout=1))
                for _ in range(recorder._queue.maxsize):
                    recorder._queue.put_nowait("{}")
                closer = threading.Thread(target=recorder.close)
                closer.start()
                self.assertTrue(recorder._stop.wait(timeout=1))
                unblock.set()
                closer.join(timeout=3)
                self.assertFalse(closer.is_alive())
                self.assertFalse(recorder._thread.is_alive())
                self.assertIsNone(recorder._sink.stream)

    def test_unwritable_logging_does_not_fail_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "file"
            invalid.write_text("not a directory")
            recorder = diagnostics.RuntimeDiagnostics(SimpleNamespace(enabled=True), invalid)
            self.assertFalse(recorder.enabled)
            recorder.update(stages={"haas_s": 1})
            self.assertEqual(recorder.snapshot()["stages"]["haas_s"], 1)
            recorder.close()

    def test_rotating_gui_stream_handles_large_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            with diagnostics.open_rotating_log(path, max_bytes=65536, backups=2) as stream:
                stream.write("字幕" * 100000)
                stream.flush()
            files = list(Path(directory).glob("run.log*"))
            self.assertLessEqual(len(files), 3)
            self.assertTrue(all(item.stat().st_size < 90000 for item in files))


class DiagnosticExportTests(unittest.TestCase):
    def test_export_only_known_logs_without_events_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            (logs / "run.log").write_text("failure token=SECRET C:\\Users\\Alice\\thing", encoding="utf-8")
            (logs / "config.toml").write_text("secret")
            (logs / "unrelated-personal.log").write_text("personal")
            (root / "cookies.txt").write_text("private")
            with patch.object(diagnostics, "_windows_event_metadata") as events:
                destination = diagnostics.export_diagnostics(root / "export.zip", logs)
                events.assert_not_called()
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()), {"logs/run.log", "manifest.json"})
                text = archive.read("logs/run.log").decode()
                self.assertNotIn("SECRET", text)
                self.assertNotIn("Alice", text)
                self.assertIn("failure", text)

    def test_export_tail_is_bounded_and_events_require_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            (logs / "mpv.log").write_bytes(b"x" * 1_000_000)
            with patch.object(diagnostics, "_windows_event_metadata", return_value={"available": False}) as events:
                destination = diagnostics.export_diagnostics(root / "export.zip", logs, include_windows_events=True)
                events.assert_called_once()
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(len(archive.read("logs/mpv.log")), diagnostics._EXPORT_FILE_BYTES)
                self.assertIn("windows-event-metadata.json", archive.namelist())

    def test_export_ignores_symlinked_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            private = root / "private.txt"
            private.write_text("private")
            try:
                (logs / "run.log").symlink_to(private)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            destination = diagnostics.export_diagnostics(root / "export.zip", logs)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["manifest.json"])

    def test_windows_events_return_only_metadata_and_bounded_count(self):
        records = [{"event_id": 4101, "provider": "Display", "time_utc": "2026-11-17T00:00:00Z",
                    "record_id": index, "level": 2, "message": "private", "user": "Alice"}
                   for index in range(120)]
        with patch.object(diagnostics.os, "name", "nt"), \
                patch.object(diagnostics.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(records))) as run:
            result = diagnostics._windows_event_metadata()
        self.assertEqual(len(result["events"]), 100)
        self.assertNotIn("message", result["events"][0])
        self.assertNotIn("user", result["events"][0])
        self.assertEqual(run.call_args.kwargs["timeout"], 10)


if __name__ == "__main__":
    unittest.main()
