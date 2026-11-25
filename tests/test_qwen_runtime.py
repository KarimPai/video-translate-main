"""CPU-only tests for Qwen cancellation, HTTP completions and server ownership."""
import asyncio
import http.server
import queue
import threading
import time
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from auto_translate import llama_server as ls
from auto_translate.asr import qwen3_asr as qwen
from auto_translate.config import LLAMA_SERVER_PATH, AsrConfig, IngestConfig
from auto_translate.ingest import Ingest, Source
from auto_translate.pipeline import run_asr
from auto_translate.types import AudioProgress, Transcript, Utterance

PCM = np.zeros(16000, dtype=np.float32)


def test_llama_server_defaults_use_b12719_cuda12_runtime():
    expected = ".runtime/llama-b12719-cuda12/llama-server.exe"
    assert LLAMA_SERVER_PATH == expected
    assert qwen.LLAMA_SERVER_PATH is LLAMA_SERVER_PATH


def completion(text="はい。", *, finish="stop", tokens=None):
    choice = {"message": {"content": f"language Japanese<asr_text>{text}"}}
    if finish is not None:
        choice["finish_reason"] = finish
    body = {"choices": [choice]}
    if tokens is not None:
        body["usage"] = {"completion_tokens": tokens}
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://localhost"))


@pytest.fixture
def backend(monkeypatch):
    server = Mock(base_url="http://localhost")
    server.client.post.return_value = completion()
    monkeypatch.setattr(qwen, "require_gguf", lambda path, label: Path(path))
    monkeypatch.setattr(qwen, "resolve_executable", Path)
    monkeypatch.setattr(qwen, "detect_device", lambda *args: "CUDA0")
    monkeypatch.setattr(qwen, "LlamaServer", lambda **kwargs: server)
    asr = qwen.Qwen3ASR(AsrConfig(language="auto", model_path="q.gguf", mmproj_path="mq.gguf"))
    server.reset_mock()
    yield asr
    asr.close()


def start_ingest(asr):
    ingest = Ingest(Source("clip.wav", live=False), IngestConfig())
    ingest.set_generation_listener(asr.cancel_pending)
    return ingest


def test_initial_warmup_timeout_keeps_healthy_server(monkeypatch):
    server = Mock(base_url="http://localhost")
    monkeypatch.setattr(qwen, "require_gguf", lambda path, label: Path(path))
    monkeypatch.setattr(qwen, "resolve_executable", Path)
    monkeypatch.setattr(qwen, "detect_device", lambda *args: "CUDA0")
    monkeypatch.setattr(qwen, "LlamaServer", lambda **kwargs: server)

    post = server.client.post

    def reply(*args, timeout, **kwargs):
        if post.call_count == 1:
            time.sleep(timeout + 0.02)
            raise httpx.ReadTimeout("first decode took longer than the warmup budget")
        return completion()

    post.side_effect = reply
    asr = qwen.Qwen3ASR(AsrConfig(language="auto", model_path="q.gguf", mmproj_path="mq.gguf", response_timeout_s=0.08))
    try:
        server.start.assert_called_once_with()
        server.close.assert_not_called()
        assert asr.transcribe(PCM, deadline=time.monotonic() + 1) == "はい。"
        assert post.call_count == 2
    finally:
        asr.close()


def test_recognize_defers_context_memory_until_pipeline_acceptance(backend, monkeypatch):
    monkeypatch.setattr(backend, "_transcribe_with_retries", lambda *args: "今日はいい天気です")

    result = backend._recognize(
        PCM, probe=False, deadline=time.monotonic() + 2.0,
        generation=backend._generation,
    )

    assert result == "今日はいい天気です"
    assert not backend._recent
    assert not backend._accepted_keys


def test_seek_generation_clears_accepted_asr_context(backend):
    backend._recent = deque(maxlen=4)
    backend.remember("前の世代の発言です")
    assert backend._recent
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(AudioProgress(sample=0, reset=True, generation=0))
    incoming.put(None)

    run_asr(backend, incoming, outgoing, get_generation=lambda: 0)

    assert not backend._recent
    assert not backend._accepted_keys


@pytest.mark.parametrize("old_reply", [
    "success", "length", "connect_error", "timeout",
])
def test_seek_signals_blocked_asr_and_only_latest_generation_reaches_output(backend, old_reply):
    ingest = start_ingest(backend)
    entered, release = threading.Event(), threading.Event()
    asr_in, asr_out = queue.Queue(), queue.Queue()
    replies = {"success": completion("古い発話です。"), "length": completion(finish="length")}

    def post(*args, **kwargs):
        if backend.server.client.post.call_count == 1:
            entered.set()
            assert release.wait(2)
            if old_reply == "connect_error":
                raise httpx.ConnectError("connection dropped during seek")
            if old_reply == "timeout":
                raise httpx.ReadTimeout("read timed out during seek")
            return replies[old_reply]
        return completion("新しい発話です。")

    backend.server.client.post.side_effect = post
    asr_in.put(Utterance(0, 1, PCM, generation=0))
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(run_asr, backend, asr_in, asr_out, lambda: ingest.generation)
        try:
            assert entered.wait(2)
            ingest.request_restart(30)
            ingest.request_restart(45)
            assert backend._generation == 2  # no reset item has been dequeued
            asr_in.put(Utterance(30, 31, PCM, generation=1))
            asr_in.put(Utterance(45, 46, PCM, generation=2))
        finally:
            asr_in.put(None)
            release.set()
        worker.result(timeout=2)

    outputs = [item for item in list(asr_out.queue) if isinstance(item, Transcript)]
    assert [(item.generation, item.ja) for item in outputs] == [(2, "新しい発話です。")]
    assert backend.server.client.post.call_count == 2
    backend.server.reset_for_restart.assert_not_called()
    backend.server.close.assert_not_called()


@pytest.mark.parametrize("shutdown", [False, True])
def test_seek_or_shutdown_wakes_timeout_backoff(backend, shutdown):
    ingest = start_ingest(backend)
    shutdown_event = threading.Event()
    backend.bind_session_stop(shutdown_event)
    waiting = threading.Event()
    original_event = backend._cancel_event

    def wait(seconds):
        waiting.set()
        return original_event.wait(seconds)

    backend._cancel_event = SimpleNamespace(wait=wait, set=original_event.set)
    backend.server.client.post.side_effect = httpx.ReadTimeout("read timed out")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.transcribe, PCM)
        assert waiting.wait(2)
        started = time.monotonic()
        if shutdown:
            shutdown_event.set()
        else:
            ingest.request_restart(15)
        assert future.result(timeout=0.4) == ""
        assert time.monotonic() - started < 0.4
    backend.server.client.post.assert_called_once()
    backend.server.reset_for_restart.assert_not_called()


def test_seek_during_first_recovery_window_never_sends_second(backend):
    ingest = start_ingest(backend)
    entered, release = threading.Event(), threading.Event()
    audio = np.zeros(16 * len(PCM), dtype=np.float32)

    def post(*args, **kwargs):
        if backend.server.client.post.call_count == 1:
            return completion(finish="length")
        entered.set()
        assert release.wait(2)
        return completion()

    backend.server.client.post.side_effect = post
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.transcribe, audio)
        try:
            assert entered.wait(2)
            ingest.request_restart(60)
        finally:
            release.set()
        assert future.result(timeout=1) == ""
    assert backend.server.client.post.call_count == 2
    backend.server.reset_for_restart.assert_not_called()


@pytest.mark.parametrize("cancel", ["seek", "session_stop", "close"])
def test_cancellation_interrupts_active_http_and_releases_inference(backend, monkeypatch, cancel):
    ingest = start_ingest(backend)
    stop = threading.Event()
    backend.bind_session_stop(stop)
    entered, disconnected = threading.Event(), threading.Event()
    requests = []

    async def request(_client, method, url, **kwargs):
        requests.append(method)
        if len(requests) > 1:
            return completion("新しい発話です。")
        entered.set()
        try:
            await asyncio.Future()
        finally:
            disconnected.set()

    transport = ls.DeadlineHttpClient()
    backend.server.client = transport
    backend.server.close.side_effect = transport.close
    monkeypatch.setattr(httpx.AsyncClient, "request", request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(backend.transcribe, PCM, deadline=time.monotonic() + 2)
        assert entered.wait(1)
        if cancel == "seek":
            ingest.request_restart(30)
        elif cancel == "session_stop":
            stop.set()
        else:
            backend.close()
        assert active.result(timeout=0.75) == ""

    assert disconnected.is_set()
    assert requests == ["POST"]
    backend.server.reset_for_restart.assert_not_called()
    if cancel == "seek":
        assert backend.transcribe(PCM, generation=1) == "新しい発話です。"
        backend.server.close.assert_not_called()


@pytest.mark.parametrize("bad_response", [
    httpx.Response(200, text="not JSON", request=httpx.Request("POST", "http://localhost")),
    httpx.Response(200, json={"choices": [0]}, request=httpx.Request("POST", "http://localhost")),
    httpx.Response(200, json={"choices": [{"finish_reason": "stop"}]},
                   request=httpx.Request("POST", "http://localhost")),
    completion(finish=None), completion(finish=None, tokens=15), completion(finish="tool_calls"),
])
def test_malformed_or_ambiguous_reply_rejects_one_cue(backend, bad_response):
    backend.server.client.post.side_effect = [bad_response, completion("はい。", tokens=256)]
    assert backend.transcribe(PCM) == ""
    assert backend.last_rejection == "invalid_response"
    assert backend.last_raw_transcript == ""
    assert backend.transcribe(PCM) == "はい。"
    assert backend.last_detected_language == "ja"
    backend.server.reset_for_restart.assert_not_called()


def test_missing_finish_at_token_cap_triggers_bounded_retry(backend):
    backend.server.client.post.side_effect = [completion(finish=None, tokens=256), completion()]
    assert backend.transcribe(PCM) == "はい。"
    assert [call.kwargs["json"]["max_tokens"]
            for call in backend.server.client.post.call_args_list] == [256, 512]
    backend.server.reset_for_restart.assert_not_called()


def test_invalid_http_encoding_rejects_only_one_cue(backend):
    backend.server.client.post.side_effect = [httpx.DecodingError("invalid gzip"), completion()]
    assert backend.transcribe(PCM) == ""
    assert backend.last_rejection == "invalid_response"
    assert backend.transcribe(PCM) == "はい。"
    assert backend.server.client.post.call_count == 2
    backend.server.reset_for_restart.assert_not_called()


def test_second_token_limit_after_restart_skips_only_the_utterance(backend, monkeypatch):
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = [
        *[httpx.ConnectError("old child unavailable") for _ in range(4)],
        completion(),  # restart warmup
        completion(finish="length"), completion(finish="length"),
        completion(),
    ]
    assert backend.transcribe(PCM) == ""
    assert backend.last_rejection == "token_limit"
    assert backend.transcribe(PCM) == "はい。"
    backend.server.reset_for_restart.assert_called_once()
    backend.server.start.assert_called_once()
    assert backend.server.client.post.call_count == 8


def test_repeated_transport_failure_is_bounded_per_utterance(backend, monkeypatch):
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ConnectError("offline")
    assert backend.transcribe(PCM) == ""
    assert backend.last_rejection == "transport_error"
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    assert backend.transcribe(PCM) == ""
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    with pytest.raises(qwen.Qwen3ASRError, match="three consecutive utterances"):
        backend.transcribe(PCM)


def test_restart_allowance_recovers_after_window(backend, monkeypatch):
    backend._restart_timestamps.clear()
    # Both timestamps are >300s old, so both are pruned
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend._restart_timestamps.append(time.monotonic() - 302)
    monkeypatch.setattr(backend, "_warmup", Mock())
    backend._restart(deadline=time.monotonic() + 1, generation=0)
    # After pruning both old timestamps, only the new one remains
    assert backend._restarts == 1
    backend.server.reset_for_restart.assert_called_once()


def test_successful_utterance_resets_consecutive_transport_failures(backend, monkeypatch):
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    offline = httpx.ConnectError("offline")
    for outcome, failures in ((offline, 1), (offline, 2), (None, 0), (offline, 1), (offline, 2)):
        backend._restart_timestamps.clear()
        backend._restart_timestamps.append(time.monotonic() - 301)
        backend.server.client.post.side_effect = outcome
        backend.transcribe(PCM)
        assert backend._transport_failures == failures


def test_seek_after_transport_failure_is_not_counted(backend, monkeypatch):
    def fail_then_seek(*args):
        backend._generation += 1
        raise qwen.Qwen3ASRTransportError("offline")

    monkeypatch.setattr(backend, "_recognize", fail_then_seek)
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 0


def test_actual_http_timeout_exhausting_deadline_counts_as_transport(backend):
    """An expired HTTP read must not turn into an unclassified deadline."""
    def read_after_budget(*args, **kwargs):
        time.sleep(0.035)
        raise httpx.ReadTimeout("HTTP read exceeded the utterance budget")

    backend.server.client.post.side_effect = read_after_budget
    assert backend.transcribe(PCM, deadline=time.monotonic() + 0.01) == ""
    assert backend._transport_failures == 1
    assert backend.last_rejection == "deadline"
    backend.server.client.post.assert_called_once()


def test_http_retry_backoff_expiry_counts_as_transport(backend):
    """A retry delay must retain the preceding transport error's cause."""
    backend.server.client.post.side_effect = httpx.ConnectError("offline")
    assert backend.transcribe(PCM, deadline=time.monotonic() + 0.03) == ""
    assert backend._transport_failures == 1
    assert backend.last_rejection == "deadline"
    backend.server.client.post.assert_called_once()


def test_three_real_deadline_expirations_are_fatal(backend):
    """Count separate expired HTTP requests, not only immediate mock errors."""
    def slow_timeout(*args, **kwargs):
        time.sleep(0.025)
        raise httpx.ReadTimeout("socket timed out")

    backend.server.client.post.side_effect = slow_timeout
    for _ in range(2):
        assert backend.transcribe(PCM, deadline=time.monotonic() + 0.01) == ""
    with pytest.raises(qwen.Qwen3ASRError, match="three consecutive utterances"):
        backend.transcribe(PCM, deadline=time.monotonic() + 0.01)
    assert backend._transport_failures == 3


def test_mixed_english_tag_keeps_recognized_japanese_dialogue():
    """Restore the established mixed-audio parser contract, without forcing JA."""
    lang, transcript = qwen.parse_qwen_asr_response(
        "language English<asr_text>Go, go! 危ない、逃げて！"
    )
    assert lang == "ja"
    assert transcript == "Go, go! 危ない、逃げて！"
    assert qwen.parse_qwen_asr_response(
        "language English<asr_text>Get in the car!"
    ) == ("en", "Get in the car!")


def test_recovered_japanese_guard_matches_downstream_kana_gate():
    """Single-kana reactions pass; Latin-only and kanji-only do not."""
    assert qwen.has_japanese_recovery_text("え")
    assert qwen.has_japanese_recovery_text("ｱ")
    assert qwen.has_japanese_recovery_text("危ない！")
    assert not qwen.has_japanese_recovery_text("Get in the car")
    assert not qwen.has_japanese_recovery_text("危険")


# ------------------------------------------------------------------ #
# Finding #1 — Transport-related deadlines count as transport failures
# ------------------------------------------------------------------ #

def test_transport_deadline_counts_as_transport_failure(backend, monkeypatch):
    """When HTTP timeouts exhaust the utterance budget, the deadline raises
    with transport=True and increments _transport_failures."""
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    # All 4 decode attempts raise HTTP timeout, then restart also fails.
    # The path raises Qwen3ASRTransportError which increments _transport_failures.
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend.server.client.post.side_effect = httpx.ReadTimeout("timeout")
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 1


def test_three_transport_deadlines_trigger_fatal_error(backend, monkeypatch):
    """Three consecutive transport-related deadline expirations should raise
    the fatal Qwen3ASRError, just like three Qwen3ASRTransportErrors."""
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ReadTimeout("timeout")
    for _ in range(2):
        backend._restart_timestamps.clear()
        backend._restart_timestamps.append(time.monotonic() - 301)
        assert backend.transcribe(PCM) == ""
    # Third consecutive transport failure should be fatal
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    with pytest.raises(qwen.Qwen3ASRError, match="three consecutive utterances"):
        backend.transcribe(PCM)
    assert backend._transport_failures == 3


def test_non_transport_deadline_does_not_count(backend, monkeypatch):
    """A deadline raised without transport=True (e.g., inference queue delay)
    should not increment _transport_failures."""
    monkeypatch.setattr(backend, "_recognize", lambda *a, **kw: (_ for _ in ()).throw(
        qwen._UtteranceDeadline("inference queue timeout", transport=False)
    ))
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 0


# ------------------------------------------------------------------ #
# Finding #2 — Non-transport failures reset the counter
# ------------------------------------------------------------------ #

def test_invalid_response_resets_transport_failure_counter(backend, monkeypatch):
    """An invalid response between transport errors should reset the counter."""
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    offline = httpx.ConnectError("offline")
    bad_json = httpx.Response(200, text="not JSON", request=httpx.Request("POST", "http://localhost"))

    # First transport error
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend.server.client.post.side_effect = offline
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 1

    # Invalid response should reset counter
    backend.server.client.post.side_effect = lambda *a, **kw: bad_json
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 0
    assert backend.last_rejection == "invalid_response"

    # Next transport error starts counting from 1
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend.server.client.post.side_effect = offline
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 1


def test_token_limit_resets_transport_failure_counter(backend, monkeypatch):
    """A token limit rejection between transport errors should reset the counter."""
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    offline = httpx.ConnectError("offline")

    # First transport error
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend.server.client.post.side_effect = offline
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 1

    # Token limit should reset counter
    backend.server.client.post.side_effect = lambda *a, **kw: completion(finish="length")
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 0
    assert backend.last_rejection == "token_limit"

    # Next transport error starts counting from 1
    backend._restart_timestamps.clear()
    backend._restart_timestamps.append(time.monotonic() - 301)
    backend.server.client.post.side_effect = offline
    assert backend.transcribe(PCM) == ""
    assert backend._transport_failures == 1


# ------------------------------------------------------------------ #
# Finding #3 — Rolling restart window
# ------------------------------------------------------------------ #

def test_rolling_restart_window_allows_two_within_300s(backend, monkeypatch):
    """Two restarts within 300 seconds should succeed; the third should fail."""
    monkeypatch.setattr(backend, "_warmup", Mock())
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ConnectError("offline")

    # First restart
    assert backend.transcribe(PCM) == ""
    assert backend._restarts == 1

    # Second restart (still within window)
    assert backend.transcribe(PCM) == ""
    assert backend._restarts == 2

    # Third restart should fail — either with restart limit or transport failure limit.
    # The restart limit fires first here since _transport_failures reaches 3.
    with pytest.raises(qwen.Qwen3ASRError, match="restart limit reached|three consecutive"):
        backend.transcribe(PCM)


def test_rolling_restart_window_prunes_old_timestamps(backend, monkeypatch):
    """After 300 seconds pass since the oldest restart, a new restart should be allowed."""
    monkeypatch.setattr(backend, "_warmup", Mock())
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ConnectError("offline")

    # Two restarts
    assert backend.transcribe(PCM) == ""
    assert backend.transcribe(PCM) == ""
    assert backend._restarts == 2

    # Reset transport failures so the next call isn't blocked by the 3-failure limit
    backend._transport_failures = 0
    # Advance both timestamps to be >300s old so both are pruned
    backend._restart_timestamps[0] = backend._restart_timestamps[0] - 301
    backend._restart_timestamps[1] = backend._restart_timestamps[1] - 301

    # Third restart should now be allowed (both old timestamps pruned)
    assert backend.transcribe(PCM) == ""
    assert backend._restarts == 1


def test_rolling_restart_window_boundary_cannot_accumulate(backend, monkeypatch):
    """A fixed window allows 2 restarts at t=0 and 2 more at t=301 (4 total).
    A rolling window should still limit to 2 within any 300s span."""
    monkeypatch.setattr(backend, "_warmup", Mock())
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ConnectError("offline")

    # Two restarts at t=0
    assert backend.transcribe(PCM) == ""
    assert backend.transcribe(PCM) == ""

    # Reset transport failures so the next call isn't blocked by the 3-failure limit
    backend._transport_failures = 0
    # Advance only the first timestamp so it prunes, but the second is still recent
    backend._restart_timestamps[0] = backend._restart_timestamps[0] - 301

    # One more restart should succeed (first pruned, second still in window)
    assert backend.transcribe(PCM) == ""
    assert backend._restarts == 2

    # Another should fail (two recent timestamps remain)
    with pytest.raises(qwen.Qwen3ASRError, match="restart limit reached"):
        backend.transcribe(PCM)


def test_cancelled_restart_does_not_consume_rolling_slot(backend, owned_server, monkeypatch):
    """When a restart is cancelled by a seek, its timestamp should be removed
    so it doesn't consume a slot in the rolling window."""
    server, _spawned = owned_server
    ingest = start_ingest(backend)
    entered, release = threading.Event(), threading.Event()

    def get(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(status_code=200)

    reconnect(backend, server, monkeypatch, get=get)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.transcribe, PCM)
        try:
            assert entered.wait(2)
            ingest.request_restart(90)
        finally:
            release.set()
        assert future.result(timeout=1) == ""

    # Cancelled restart should not have consumed a timestamp slot
    assert len(backend._restart_timestamps) == 0
    assert backend._restarts == 0


def test_expired_startup_deadline_still_reaps_owned_child(owned_server):
    server, spawned = owned_server
    server.client.get = Mock(return_value=SimpleNamespace(status_code=503))
    with pytest.raises(ls.LlamaServerError, match="deadline"):
        server.start(deadline=time.monotonic() + 0.05)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert server.process is None


def test_close_during_reset_never_calls_start(backend, monkeypatch):
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server.client.post.side_effect = httpx.ConnectError("child unavailable")
    entered, release = threading.Event(), threading.Event()

    def reset(**kwargs):
        entered.set()
        assert release.wait(2)

    backend.server.reset_for_restart.side_effect = reset
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(backend.transcribe, PCM)
        try:
            assert entered.wait(2)
            backend.close()
        finally:
            release.set()
        assert result.result(timeout=1) == ""
    backend.server.start.assert_not_called()
    assert backend.server.client.post.call_count == 4


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def owned_server(monkeypatch, tmp_path):
    spawned = []

    def spawn(*args, **kwargs):
        process = FakeProcess()
        spawned.append(process)
        return process

    monkeypatch.setattr(ls, "owned_popen", spawn)
    monkeypatch.setattr(ls, "_free_port", lambda: 54321)
    server = ls.LlamaServer("runtime-test", tmp_path / "llama-server.exe", [], startup_timeout_s=50)
    server.log_path = tmp_path / "server.log"
    yield server, spawned
    server.close()


def reconnect(backend, server, monkeypatch, *, get, post=None):
    server.client.close()
    server.client = Mock()
    server.client.post.side_effect = httpx.ConnectError("old child unavailable")
    replacement = Mock(get=Mock(side_effect=get), post=Mock(side_effect=post))
    monkeypatch.setattr(server, "_new_client", lambda: replacement)
    monkeypatch.setattr(backend, "_wait_retry", Mock())
    backend.server = server
    return replacement


def test_seek_during_old_child_reap_keeps_transport_restartable(backend, owned_server, monkeypatch):
    server, spawned = owned_server
    ingest = start_ingest(backend)
    waiting, release = threading.Event(), threading.Event()
    healthy = SimpleNamespace(status_code=200)
    client = reconnect(backend, server, monkeypatch, get=lambda *args, **kwargs: healthy,
                       post=lambda *args, **kwargs: completion())
    old_client = server.client
    old_client.get.return_value = healthy
    server.start()

    def wait(timeout=None):
        waiting.set()
        assert release.wait(2)
        return spawned[0].poll()

    spawned[0].wait = wait
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(backend.transcribe, PCM)
        try:
            assert waiting.wait(2)
            ingest.request_restart(60)
        finally:
            release.set()
        assert result.result(timeout=1) == ""

    assert server.client is old_client
    old_client.close.assert_not_called()
    assert backend.transcribe(PCM, generation=1) == "はい。"
    old_client.close.assert_called_once()
    assert len(spawned) == 2 and server.alive()
    assert client.post.call_count == 2  # warmup and speech, both for the new generation


def test_restart_health_checks_and_cleanup_obey_original_wall_clock_budget(
    backend, owned_server, monkeypatch,
):
    server, spawned = owned_server

    def get(url, *, timeout):
        time.sleep(timeout)
        return SimpleNamespace(status_code=503)

    client = reconnect(backend, server, monkeypatch, get=get)
    begin = time.monotonic()
    budget = 0.65
    assert backend.transcribe(PCM, deadline=begin + budget) == ""
    elapsed = time.monotonic() - begin
    assert budget - 0.03 <= elapsed < budget + 0.25
    assert backend.last_rejection == "deadline"
    assert backend._restarts == 1
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert server.process is None and server._log_stream is None
    client.post.assert_not_called()


def test_restart_warmup_uses_only_remaining_wall_clock_budget(backend, owned_server, monkeypatch):
    server, spawned = owned_server
    timeouts = []

    def get(url, *, timeout):
        time.sleep(0.1)
        return SimpleNamespace(status_code=200)

    def post(url, *, json, timeout):
        timeouts.append(timeout)
        time.sleep(timeout)
        return completion()

    client = reconnect(backend, server, monkeypatch, get=get, post=post)
    begin = time.monotonic()
    budget = 0.65
    assert backend.transcribe(PCM, deadline=begin + budget) == ""
    assert budget - 0.03 <= time.monotonic() - begin < budget + 0.25
    assert backend.last_rejection == "deadline"
    assert len(timeouts) == 1 and 0 < timeouts[0] < budget - 0.08
    client.post.assert_called_once()
    assert len(spawned) == 1 and server.alive()


@pytest.mark.parametrize("operation", ["start", "reset_for_restart"])
def test_lifecycle_lock_contention_cannot_outlast_utterance(owned_server, operation):
    server, spawned = owned_server
    with ThreadPoolExecutor(max_workers=1) as pool, server._lock:
        result = pool.submit(getattr(server, operation), deadline=time.monotonic() + 0.1)
        with pytest.raises(ls.LlamaServerError, match="deadline"):
            result.result(timeout=0.75)
    assert not spawned


def test_normal_and_watchdog_recovery_are_serialized(backend, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "calls": 0}

    def reset(*, deadline, check=None):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["calls"] += 1
            call_number = state["calls"]
        try:
            if call_number == 1:
                entered.set()
                assert release.wait(2.0)
            if check is not None:
                check()
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(backend.server, "reset_for_restart", reset)
    monkeypatch.setattr(backend, "_warmup", lambda *_args: None)
    errors = []
    deadline = time.monotonic() + 3.0

    def normal_recovery():
        try:
            backend._restart(deadline=deadline, generation=backend._generation)
        except Exception as exc:
            errors.append(exc)

    def watchdog_recovery():
        try:
            backend.recover_stalled(deadline=deadline)
        except Exception as exc:
            errors.append(exc)

    normal = threading.Thread(target=normal_recovery)
    watchdog = threading.Thread(target=watchdog_recovery)
    normal.start()
    try:
        assert entered.wait(1.0)
        watchdog.start()
        time.sleep(0.03)
        assert state["calls"] == 1
    finally:
        release.set()
    normal.join(2.0)
    watchdog.join(2.0)

    assert not normal.is_alive() and not watchdog.is_alive()
    assert not errors
    assert state["calls"] == 2
    assert state["max_active"] == 1


def test_startup_unwinds_while_concurrent_close_owns_reap(owned_server):
    server, spawned = owned_server
    checking, release_health = threading.Event(), threading.Event()
    reaping, release_reap = threading.Event(), threading.Event()

    def health(*args, **kwargs):
        checking.set()
        assert release_health.wait(2)
        return SimpleNamespace(status_code=200)

    def wait_for_exit(timeout=None):
        reaping.set()
        assert release_reap.wait(2)
        return spawned[0].poll()

    server.client.close()
    server.client = Mock(get=health)
    with ThreadPoolExecutor(max_workers=2) as pool:
        startup = pool.submit(server.start, deadline=time.monotonic() + 5)
        try:
            assert checking.wait(1)
            spawned[0].wait = wait_for_exit
            closing = pool.submit(server.close)
            assert reaping.wait(1)
            release_health.set()
            with pytest.raises(CancelledError):
                startup.result(timeout=0.75)
            assert not closing.done()
        finally:
            release_health.set()
            release_reap.set()
        closing.result(timeout=1)
    assert len(spawned) == 1
    assert server.process is None and server._log_stream is None
    server.client.close.assert_called_once()


@pytest.mark.parametrize("stage", ["health", "warmup"])
@pytest.mark.parametrize("cancel", ["seek", "close"])
def test_cancelling_restart_cannot_resurrect_or_retry(
    backend, owned_server, monkeypatch, stage, cancel,
):
    server, spawned = owned_server
    ingest = start_ingest(backend)
    entered, release = threading.Event(), threading.Event()

    def get(*args, **kwargs):
        if stage == "health":
            entered.set()
            assert release.wait(2)
        return SimpleNamespace(status_code=200)

    def post(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return completion()

    client = reconnect(backend, server, monkeypatch, get=get, post=post)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.transcribe, PCM)
        try:
            assert entered.wait(2)
            if cancel == "close":
                backend.close()
            else:
                ingest.request_restart(90)
        finally:
            release.set()
        assert future.result(timeout=1) == ""

    assert len(spawned) == 1
    assert backend._restarts == 0
    assert client.post.call_count == (1 if stage == "warmup" else 0)
    assert server.alive() == (stage == "warmup" and cancel == "seek")
    if cancel == "close":
        backend.close()
        with pytest.raises(ls.LlamaServerError, match="closed"):
            server.reset_for_restart()
        client.close.assert_called_once()


def test_http_wall_deadline_cuts_off_trickling_reply():
    # Each chunk arrives faster than the request timeout. Only a wall timer
    # can stop this response before the whole body is received.
    class Trickle(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            with suppress(OSError):
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.02)

        def log_message(self, *_args):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Trickle) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        client = ls.DeadlineHttpClient()
        try:
            start = time.monotonic()
            with pytest.raises(httpx.TimeoutException):
                client.get(f"http://127.0.0.1:{httpd.server_port}", timeout=0.18)
            assert 0.15 <= time.monotonic() - start < 0.7
        finally:
            client.close()
            httpd.shutdown()
            thread.join(timeout=1)
