"""HY-MT2 Japanese-to-English subtitle translation over a local chat endpoint."""

import json
import time
import threading
import urllib.request

from ..config import TranslateConfig
from ..glossary import Glossary


class HyMt2Translator:
    def __init__(self, cfg: TranslateConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0

        # Warm up the selected HY-MT2 model before processing subtitles.
        self._chat(
            self.build_prompt("こんにちは", []),
            timeout=cfg.load_timeout_s,
        )

    def build_prompt(self, ja: str, history: list[tuple[str, str]]) -> str:
        parts = []

        if hits := self.glossary.find(ja):
            parts.append(
                "Reference translations:\n"
                + "\n".join(
                    f"{term} translates to {en}"
                    for term, en in hits
                )
            )

        if history:
            parts.append(
                "Context (do not translate):\n"
                + "\n".join(
                    f"JA: {j}\nEN: {e}"
                    for j, e in history
                )
            )

        parts.append(
            "Translate the following text into English. "
            "Only output the translated result without any additional explanation:\n\n"
            + ja
        )
        return "\n\n".join(parts)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        result = self._chat(
            self.build_prompt(ja, history),
            timeout=self.cfg.timeout_s,
        )
        return _clean(result)

    def _pace_request(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            wait = self.cfg.min_request_interval_s - (
                now - self._last_request_at
            )
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _chat(self, user_msg: str, timeout: float) -> str:
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            **self.cfg.extra_body,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        self._pace_request()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
            return data["choices"][0]["message"]["content"]


def _clean(text: str) -> str:
    """Keep the first non-empty line, minus an EN prefix and surrounding quotes."""
    line = next(
        (ln.strip() for ln in text.strip().splitlines() if ln.strip()),
        "",
    )

    if line.upper().startswith("EN:"):
        line = line[3:].strip()

    return line.strip('"“”「」')
