"""Bounded pipeline queue operations that cooperate with cancellation."""

import queue


class QueueStopped(RuntimeError):
    """A queue consumer was cancelled before another item arrived."""


def get_until_stopped(work_queue, stop=None, timeout_s: float = 0.1):
    while stop is None or not stop.is_set():
        try:
            return work_queue.get(timeout=timeout_s)
        except queue.Empty:
            pass
    raise QueueStopped()


def put_until_stopped(work_queue, item, stop=None, timeout_s: float = 0.1) -> bool:
    # Preserve best-effort EOF notification for existing consumers, without
    # ever waiting on a full queue after cancellation.
    if item is None:
        try:
            work_queue.put_nowait(item)
            return True
        except queue.Full:
            pass
    while stop is None or not stop.is_set():
        try:
            work_queue.put(item, timeout=timeout_s)
            return True
        except queue.Full:
            pass
    return False
