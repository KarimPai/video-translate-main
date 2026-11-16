"""Optional, bounded public YouTube chat evidence, isolated from audio decoding.

Only terms already heard by an ASR candidate leave this module. Chat text and
viewer identities are never placed in model prompts or application logs. Live
chat requires an explicit source-to-UTC anchor; replay offsets are authoritative.
"""
import hashlib
import json
import logging
import math
import re
import threading
from collections import deque
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx

log = logging.getLogger(__name__)
_TERM = re.compile(r"[ァ-ヶー]{2,20}|[一-鿿]{2,12}|[A-Za-z][A-Za-z0-9_-]{1,23}")
_SAFE_TERM = re.compile(r"[ぁ-ヺ一-鿿A-Za-z0-9ー_-]{2,24}\Z")
_UNTRUSTED = re.compile(
    r"https?://|www\.|discord\.gg|(?:buy|sale|subscribe|promo|giveaway)\b|"
    r"ignore|instruction|system|assistant|prompt|translate|subtitle|"
    r"指示|命令|翻訳して|無視して|広告|宣伝|登録して", re.I,
)
_MAX_RESPONSE = 2 * 1024 * 1024


def youtube_video_id(url):
    """Accept public YouTube video URLs only; never fetch arbitrary locations."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        return None
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if not candidate and parsed.path.startswith(("/live/", "/embed/", "/shorts/")):
            candidate = parsed.path.split("/")[2]
    else:
        return None
    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None


def _objects(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for item in value.values():
            yield from _objects(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item, key)


def _script_json(page, marker, *, merge=False):
    combined = {}
    for match in re.finditer(marker, page):
        try:
            value, _ = json.JSONDecoder().raw_decode(page[match.end():].lstrip())
            if isinstance(value, dict):
                if merge:
                    combined.update(value)
                else:
                    return value
        except (ValueError, RecursionError):
            pass
    return combined


def _continuation(data):
    for name in ("timedContinuationData", "invalidationContinuationData",
                 "liveChatReplayContinuationData", "reloadContinuationData"):
        for item in _objects(data, name):
            if isinstance(item, dict) and item.get("continuation"):
                return item["continuation"], item.get("timeoutMs", 5000) / 1000
    return None, 5.0


class _YouTubeChatSource:
    """One synchronous HTTP poll, called exclusively by the chat daemon.

    No cookies, credentials, browser profile, downloaded chat files, or additional
    thread pools. Network and response size bounds also cover missing/private chat.
    """
    def __init__(self, video_id, live, start_s, limit):
        self.client = httpx.Client(timeout=2.0, follow_redirects=False,
                                   headers={"User-Agent": "Mozilla/5.0"})
        self.video_id, self.live = video_id, live
        self.start_s, self.limit = start_s, limit
        self.token = None
        self.context = None
        self.key = None
        self.closed = False
        self.interval_s = 5.0

    def _request(self, method, url, **kwargs):
        with self.client.stream(method, url, **kwargs) as response:
            response.raise_for_status()
            chunks = bytearray()
            for chunk in response.iter_bytes():
                if self.closed:
                    raise RuntimeError("chat cancelled")
                chunks.extend(chunk)
                if len(chunks) > _MAX_RESPONSE:
                    raise ValueError("chat response exceeds size limit")
            return chunks.decode("utf-8")

    def poll(self):
        if self.closed:
            return []
        if self.context is None:
            page = self._request("GET", f"https://www.youtube.com/watch?v={self.video_id}")
            initial = _script_json(page, r'(?:var\s+)?ytInitialData\s*=\s*')
            renderer = next(_objects(initial, "liveChatRenderer"), None)
            if not renderer:
                raise LookupError("public chat unavailable")
            self.token, _ = _continuation(renderer)
            config = _script_json(page, r'ytcfg\.set\(\s*', merge=True)
            client = config.get("INNERTUBE_CONTEXT", {}).get("client", {})
            version = client.get("clientVersion") or config.get("INNERTUBE_CLIENT_VERSION")
            if not self.token or not isinstance(version, str):
                raise LookupError("chat continuation unavailable")
            self.context = {"client": {"clientName": "WEB", "clientVersion": version,
                                       "hl": "ja"}}
            self.key = config.get("INNERTUBE_API_KEY")
        if not self.token:
            return []
        action = "get_live_chat" if self.live else "get_live_chat_replay"
        payload = {"context": self.context, "continuation": self.token}
        if not self.live:
            payload["currentPlayerState"] = {"playerOffsetMs": str(int(self.start_s * 1000))}
        params = {"key": self.key} if self.key else None
        data = json.loads(self._request("POST", f"https://www.youtube.com/youtubei/v1/live_chat/{action}",
                                        params=params, json=payload))
        contents = data.get("continuationContents", {}).get("liveChatContinuation", {})
        self.token, self.interval_s = _continuation(contents)
        actions = contents.get("actions") or []
        messages = []
        for action in actions[:self.limit]:
            replay = action.get("replayChatItemAction")
            offset = None
            nested = [action]
            if isinstance(replay, dict):
                offset = _finite(replay.get("videoOffsetTimeMsec"))
                offset = None if offset is None else offset / 1000
                nested = replay.get("actions") or []
            for inner in nested[:4]:
                renderer = inner.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer")
                if not isinstance(renderer, dict):
                    continue
                # Ignore paid/membership/system messages and never access author fields.
                runs = renderer.get("message", {}).get("runs") or []
                text = "".join(str(run.get("text", "")) if "emoji" not in run else "😀"
                               for run in runs[:100])
                stamp = _finite(renderer.get("timestampUsec"))
                messages.append({"text": text, "source_s": offset,
                                 "unix_s": None if stamp is None else stamp / 1e6})
        return messages[:self.limit]

    def close(self):
        self.closed = True
        self.client.close()


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _clean(text):
    text = str(text)
    if not 2 <= len(text) <= 160 or _UNTRUSTED.search(text):
        return None
    if any(ord(char) < 32 for char in text):
        return None
    if re.search(r"(.)\1{5,}", text):
        return None
    letters = sum(char.isalnum() for char in text)
    if letters < 2 or letters / len(text) < .45:
        return None
    return text.strip()


@dataclass(frozen=True)
class _Message:
    source_s: float
    text: str


class ChatContext:
    def __init__(self, cfg, video_url, *, live=False, start_s=0.0, glossary=None,
                 source_factory=None):
        self.cfg, self.live, self.glossary = cfg, live, glossary
        self.video_id = youtube_video_id(video_url)
        self._factory = source_factory or _YouTubeChatSource
        self._messages = deque(maxlen=cfg.max_messages)
        self._seen = deque(maxlen=cfg.max_messages)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._source = None
        self._anchor = None
        self._generation = 0
        self._revision = 0
        self._start_s = max(0., float(start_s))
        self._position_s = self._start_s
        self.status = "disabled" if not cfg.enabled else "idle"
        self.alignment = "replay_offset" if not live else "unavailable"

    def start(self):
        if not self.cfg.enabled or not self.video_id or self._stop.is_set():
            if self.cfg.enabled and not self.video_id:
                self.status = "unsupported_source"
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="youtube-chat", daemon=True)
            self._thread.start()

    def set_clock_anchor(self, source_s, unix_s, *, uncertainty_s=0.0, alignment="measured"):
        """Map live UTC timestamps; label receipt estimates honestly for diagnostics.

        A PCM-receipt estimate includes unknown encoder/network delay. It is weak
        timing evidence, never a verified program-date-time/media-clock anchor.
        """
        values = [_finite(x) for x in (source_s, unix_s, uncertainty_s)]
        if any(x is None for x in values) or values[2] < 0:
            return
        with self._lock:
            self._anchor = tuple(values)
            self.alignment = "receipt_estimate" if alignment == "receipt_estimate" else "measured"

    def reset(self, generation, start_s=0.0):
        """Invalidate in-flight responses and rebuild replay at a new seek point."""
        with self._lock:
            if generation == self._generation:
                return
            self._generation = generation
            self._revision += 1
            self._start_s = max(0., float(start_s))
            self._position_s = self._start_s
            self._messages.clear()
            self._seen.clear()
        self._wake.set()

    def observe(self, text, *, source_s=None, unix_s=None, generation=None):
        """Accept already-timestamped public messages; intentionally ignore identities."""
        text = _clean(text)
        if text is None or not self.cfg.enabled or self._stop.is_set():
            return False
        with self._lock:
            if generation is not None and generation != self._generation:
                return False
            stamp = _finite(source_s)
            if self.live:
                unix = _finite(unix_s)
                if self._anchor is None or unix is None:
                    return False
                source, wall, uncertainty = self._anchor
                # Shift late by the uncertainty so a possibly-future message
                # cannot be admitted before its conservative upper time bound.
                stamp = source + unix - wall + uncertainty
            if stamp is None or stamp < 0:
                return False
            fingerprint = hashlib.blake2b(text.casefold().encode(), digest_size=12).digest()
            if fingerprint in self._seen:
                return False
            self._seen.append(fingerprint)
            self._messages.append(_Message(stamp, text))
        return True

    def evidence(self, start_s, end_s, primary, secondary, generation=None):
        """Return lexical support for a disagreement, never new transcript content."""
        if not self.cfg.enabled or self._stop.is_set() or primary == secondary:
            return []
        sources = []
        if self.glossary is not None:
            sources.extend(source for source, _ in self.glossary.find(primary))
            sources.extend(source for source, _ in self.glossary.find(secondary))
        sources.extend(_TERM.findall(primary))
        sources.extend(_TERM.findall(secondary))
        candidates = [term for term in dict.fromkeys(sources)
                      if _SAFE_TERM.fullmatch(term) and not _UNTRUSTED.search(term)
                      and ((term in primary) != (term in secondary))][:24]
        with self._lock:
            if generation is not None and generation != self._generation:
                return []
            # Feed replay's playerOffsetMs without blocking the speech path.
            position = _finite(end_s)
            if position is not None:
                self._position_s = max(self._position_s, position)
            recent = [message.text for message in self._messages
                      if start_s - self.cfg.window_s <= message.source_s <= end_s]
        evidence = []
        for term in candidates:
            count = sum(term in text for text in recent)
            if count:
                evidence.append({"term": term, "mentions": min(3, count),
                                 "strength": "weak_public_chat"})
        return evidence[:6]

    def _run(self):
        source = None
        revision = -1
        try:
            while not self._stop.is_set():
                with self._lock:
                    requested_revision = self._revision
                    generation = self._generation
                    start_s = self._start_s
                if source is None or revision != requested_revision:
                    if source is not None:
                        self._close_source(source)
                    source = self._factory(self.video_id, self.live, start_s, self.cfg.max_messages)
                    revision = requested_revision
                    with self._lock:
                        self._source = source
                    if self._stop.is_set():
                        return
                if not self.live and hasattr(source, "start_s"):
                    with self._lock:
                        source.start_s = self._position_s
                self.status = "polling"
                batch = source.poll()
                with self._lock:
                    current = revision == self._revision
                if current and not self._stop.is_set():
                    for message in batch[:self.cfg.max_messages]:
                        self.observe(message.get("text", ""), source_s=message.get("source_s"),
                                     unix_s=message.get("unix_s"), generation=generation)
                self.status = "available"
                interval = min(30., max(self.cfg.poll_interval_s,
                                        getattr(source, "interval_s", 0.)))
                self._wake.wait(interval)
                self._wake.clear()
        except Exception as exc:
            if not self._stop.is_set():
                self.status = "unavailable"
                # Exception strings may contain signed URLs or raw network/chat data.
                log.info("Public YouTube chat unavailable (%s); speech translation continues",
                         type(exc).__name__)
        finally:
            if source is not None:
                self._close_source(source)
            with self._lock:
                self._source = None

    @staticmethod
    def _close_source(source):
        try:
            source.close()
        except Exception as exc:
            log.debug("Chat cleanup unavailable (%s)", type(exc).__name__)

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            source = self._source
            self._messages.clear()
            self._seen.clear()
        if source is not None:
            self._close_source(source)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(.5)
        self.status = "closed"
