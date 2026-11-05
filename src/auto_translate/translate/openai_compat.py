
"""LLM backend over an OpenAI-style chat API (Ollama, llama.cpp llama-server)."""

import json
import os
import time
import threading
import urllib.error
import urllib.request

from ..config import OpenAICompatConfig
from ..glossary import Glossary


SYSTEM_PROMPT = """\
You are a highly accurate Japanese-to-English subtitle translator specializing in Hololive, VTubers, livestreams, gaming and Japanese internet culture.

Your priority is faithful meaning, natural English and consistency.

TRANSLATION PRIORITIES:
1. Preserve the original Japanese meaning.
2. Preserve all meaningful information and speaker intent.
3. Use natural conversational English.
4. Keep subtitles concise without removing important details.
5. Maintain consistency with previous translations and the glossary.

JAPANESE UNDERSTANDING:
- Interpret casual Japanese, contractions, slang, incomplete sentences and conversational speech.
- Understand implied subjects, omitted pronouns and context-dependent expressions.
- Use conversation context to resolve ambiguous words and expressions.
- Treat the supplied Japanese transcription as the primary source.
- Correct a likely transcription error only when the surrounding context strongly supports the correction.
- Never replace unclear words with guesses merely to make a sentence sound natural.
- If the meaning is uncertain, preserve the closest reasonable interpretation.

HOLOLIVE AND INTERNET CULTURE:
- Nakiri Ayame (百鬼あやめ) sometimes refers to herself as 「余」(よ / yo). Interpret this as her first-person pronoun, meaning "I", "me" or "my" according to context. Do not translate it literally as "surplus" or "remainder".
- Recognize Hololive talents, generations, groups, managers, nicknames and fan names.
- Preserve official English spellings of talent names.
- Recognize Hololive terminology, memes, inside jokes and recurring references.
- Understand VTuber expressions, gaming terminology, streamer slang and Japanese internet abbreviations.
- Do not translate proper names literally.
- Prefer established English terminology over literal dictionary translations.

SPEECH AND PERSONALITY:
- Preserve humor, sarcasm, excitement, frustration and emotional tone.
- Translate reactions and interjections naturally.
- Preserve meaningful repetitions and speech patterns.
- Do not unnecessarily remove short responses or conversational details.
- Do not add explanations or information that the speaker did not express.

CONTEXT AND GLOSSARY:
- Earlier JA/EN lines are context only.
- Use context to understand pronouns, subjects, references and ongoing jokes.
- Follow glossary translations when the Japanese term matches the glossary entry.
- Preserve consistent translations for recurring names and terminology.
- Do not force glossary entries when the context clearly indicates a different meaning.

INPUT:
- Translate ONLY the Japanese text following "Translate:".
- Do not translate or repeat previous context lines.

OUTPUT:
- Output ONLY the English translation.
- Use natural, conversational English suitable for subtitles.
- Keep the translation concise but preserve the complete meaning.
- No explanations, notes, alternatives, romaji or quotation marks.
- No speaker labels.
- Do not output Japanese text.
"""


class OpenAICompatTranslator:
    def __init__(self, cfg: OpenAICompatConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self.is_hymt2 = "hy-mt2" in cfg.model.lower()

        # Warm up the selected model before processing subtitles.
        self._chat(
            self.build_prompt("こんにちは", []),
            timeout=cfg.load_timeout_s,
        )

    def build_prompt(self, ja: str, history: list[tuple[str, str]]) -> str:
        parts = []

        # Include only glossary entries matching the current Japanese text.
        if hits := self.glossary.find(ja):
            if self.is_hymt2:
                parts.append(
                    "Reference translations:\n"
                    + "\n".join(
                        f"{term} translates to {en}"
                        for term, en in hits
                    )
                )
            else:
                parts.append(
                    "Glossary:\n"
                    + "\n".join(
                        f"{term} = {en}"
                        for term, en in hits
                    )
                )

        # Previous subtitle lines are context, not translation input.
        if history:
            parts.append(
                "Context (do not translate):\n"
                + "\n".join(
                    f"JA: {j}\nEN: {e}"
                    for j, e in history
                )
            )

        if self.is_hymt2:
            parts.append(
                "Translate the following text into English. "
                "Only output the translated result without any additional explanation:\n\n"
                + ja
            )
        else:
            parts.append(f"Translate:\nJA: {ja}")

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
        if self.is_hymt2:
            messages = [
                {"role": "user", "content": user_msg}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            **self.cfg.extra_body,
        }

        headers = {"Content-Type": "application/json"}

        if self.cfg.api_key_env:
            api_key = os.getenv(self.cfg.api_key_env)

            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self.cfg.api_key_env} is not set"
                )

            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )

        # Local models get one attempt to avoid long subtitle stalls.
        # Cloud APIs retain retries for temporary server errors.
        max_attempts = 1 if not self.cfg.api_key_env else 5

        for attempt in range(max_attempts):
            self._pace_request()

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.load(resp)
                    return data["choices"][0]["message"]["content"]

            except urllib.error.HTTPError as exc:
                if (
                    exc.code not in (429, 500, 502, 503, 504)
                    or attempt == max_attempts - 1
                ):
                    raise

                delay = min(2 ** attempt, 8)

                print(
                    f"Translation API HTTP {exc.code}; retrying in {delay}s "
                    f"({attempt + 1}/{max_attempts - 1})",
                    flush=True,
                )

                time.sleep(delay)

        raise RuntimeError("Translation API request failed")


def _clean(text: str) -> str:
    """Keep the first non-empty line, minus an EN prefix and surrounding quotes."""
    line = next(
        (ln.strip() for ln in text.strip().splitlines() if ln.strip()),
        "",
    )

    if line.upper().startswith("EN:"):
        line = line[3:].strip()

    return line.strip('"“”「」')
