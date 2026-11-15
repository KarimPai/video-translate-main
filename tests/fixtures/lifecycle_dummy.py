"""Windows integration fixture. Never imports inference or downloads models."""
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from auto_translate.gui_control import BackendControl, emit_event
from auto_translate.gui_runner import READY_MARKER
from auto_translate.windows_job import bind_process_tree


def sleeping_worker():
    time.sleep(120)


def backend(folder, scenario):
    bind_process_tree()
    stop = threading.Event()
    options = sys.argv
    control = BackendControl(int(options[options.index("--gui-control-port") + 1]),
                             options[options.index("--gui-control-token") + 1], stop)

    class Player:
        def queue_gui_command(self, action, value):
            if action == "seek":
                (folder / "seek").touch()

    control.player = Player()
    # Three indefinite waits stand in for native media, HTTP and CUDA calls.
    sleepers = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
                for _ in ("mpv", "llama-server")]
    worker = multiprocessing.get_context("spawn").Process(target=sleeping_worker)
    worker.start()
    if scenario != "initializing":
        print(READY_MARKER, flush=True)
        emit_event("ready", live=False, duration=300)
    info = folder / "ready.tmp"
    info.write_text(json.dumps([os.getpid(), worker.pid, *(p.pid for p in sleepers)]))
    info.replace(folder / "ready")
    while True:
        if scenario == "backend-crash" and (folder / "act").exists():
            os._exit(59)
        if stop.wait(.02) and scenario != "hang":
            # Exit while all three are still alive to exercise kernel cleanup.
            os._exit(0)


def frontend(folder, scenario):
    from PySide6.QtWidgets import QApplication
    from auto_translate import gui
    from auto_translate.windows_job import session_members

    app = QApplication([])
    gui.GUI_SHUTDOWN_MS = 700 if scenario == "hang" else 8000
    pyw = Path(sys.executable).with_name("pythonw.exe")
    gui.build_backend_command = lambda *a, **kw: (
        str(pyw), ["-u", str(Path(__file__).resolve()), "backend", str(folder), scenario])
    (folder / "config.example.toml").touch()
    window = gui.VideoTranslateWindow(folder)
    window.show()
    window.url_input.setText("https://youtu.be/dummy")
    for cycle in range(2 if scenario == "stop" else 1):
        window.start_translation()
        assert window.process is not None
        (folder / f"session{cycle}").write_text(window._session)
        deadline = time.monotonic() + 30
        acted = False
        while window.process is not None:
            app.processEvents()
            assert time.monotonic() < deadline
            if acted or not (folder / "act").exists():
                time.sleep(.01)
                continue
            if scenario == "seek":
                window._request_seek(10)
                if not (folder / "seek").exists():
                    continue
            acted = True
            if scenario in ("close", "initializing", "seek"):
                window.close()
            elif scenario == "disconnect":
                window._close_control()
            elif scenario in ("stop", "hang"):
                window.stop_button.click()
                window.stop_translation()
        assert not session_members(window._session)
        if scenario == "hang":
            assert "forcing session shutdown" in window.console.toPlainText()
        (folder / "act").unlink(missing_ok=True)
        (folder / "ready").unlink(missing_ok=True)
    window.close()
    app.processEvents()


if __name__ == "__main__":
    role, directory, mode = sys.argv[1:4]
    (backend if role == "backend" else frontend)(Path(directory), mode)
