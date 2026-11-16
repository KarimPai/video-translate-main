"""Structured GUI review events and explicit CLI subtitle export."""
from dataclasses import asdict
import logging
import threading


class ReviewOutput:
    def __init__(self, *, gui=False, export_path=None, language="bilingual"):
        self.gui = gui
        self.export_path = export_path
        self.language = language
        self._cues = {}
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._final_ends = {}

    def show(self, sub):
        if self.export_path:
            with self._lock:
                self._cues[(round(sub.start_s * 1000), round(sub.end_s * 1000))] = sub
        if self.gui:
            from ..gui_control import emit_event
            payload = asdict(sub)
            payload.pop("timings", None)
            with self._event_lock:
                self._final_ends[sub.generation] = max(sub.end_s, self._final_ends.get(sub.generation, 0.0))
                emit_event("subtitle", **payload)

    def show_preview(self, preview):
        if self.gui:
            from ..gui_control import emit_event
            with self._event_lock:
                if preview.get("end_s", float("inf")) <= self._final_ends.get(preview.get("generation"), -1):
                    return
                emit_event("preview", **preview)

    def drain(self):
        self._export()

    def close(self):
        self._export()

    def _export(self):
        if self.export_path:
            from ..subtitle_files import export_subtitles
            with self._lock:
                cues = [self._cues[k] for k in sorted(self._cues)]
            if cues:
                export_subtitles(self.export_path, cues, language=self.language)


class PreviewDispatcher:
    """Latest-only viewer dispatch so a slow UI cannot block segmentation."""
    def __init__(self, outputs, current_generation, stop):
        self.outputs = outputs
        self.current_generation = current_generation
        self.stop = stop
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="preview-output", daemon=True)
        self._thread.start()

    def submit(self, event):
        with self._condition:
            if not self._closed:
                self._pending = event
                self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                event, self._pending = self._pending, None
            if self.stop.is_set() or event["generation"] != self.current_generation():
                continue
            for output in self.outputs:
                if self.stop.is_set() or event["generation"] != self.current_generation():
                    break
                callback = getattr(output, "show_preview", None)
                if callable(callback):
                    try:
                        callback(event)
                    except Exception:
                        logging.getLogger(__name__).exception("Optional preview viewer failed")

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=1.0)
