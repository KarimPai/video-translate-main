"""PLaMo 3 Translate via a local llama.cpp OpenAI-compatible server."""

import json
import urllib.request

from ..config import TranslateConfig
from ..glossary import Glossary


class PlamoTranslator:
    """Japanese-to-English PLaMo translator; inference stays on the local machine."""

    def __init__(self, cfg: TranslateConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.url = cfg.plamo_base_url.rstrip("/") + "/chat/completions"
        # Warm up the model before ingest starts so first subtitles do not pay load cost.
        self._chat("こんにちは", [], timeout=cfg.load_timeout_s)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return _clean(self._chat(ja, history, timeout=self.cfg.timeout_s))

    def _chat(
        self,
        ja: str,
        history: list[tuple[str, str]],
        timeout: float,
    ) -> str:
        messages: list[dict[str, str]] = []
        for source, translation in history[-self.cfg.context_lines:]:
            messages.append({"role": "user", "content": source})
            messages.append({"role": "assistant", "content": translation})
        messages.append({"role": "user", "content": ja})

        options: dict = {
            "source_lang": "ja",
            "target_lang": "en",
        }
        if hits := self.glossary.find(ja):
            options["glossary"] = [
                {"source": source, "target": target}
                for source, target in hits
            ]

        body = {
            "model": self.cfg.plamo_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.cfg.max_tokens,
            "stop": ["<|plamo:tag|>", "<|plamo:bos|>"],
            "chat_template_kwargs": options,
            "stream": False,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]


def _clean(text: str) -> str:
    """Remove empty lines, surrounding quotes, and accidental role prefixes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    if line.upper().startswith(("EN:", "ENGLISH:")):
        line = line.split(":", 1)[1].strip()
    return line.strip('"“”「」')
