"""Tencent Hy-MT3 1.8B translation prompts and shared llama.cpp backend."""
from __future__ import annotations

import re

from .hymt import HymtTranslator, build_context, clean_output


def build_hymt3_prompt(text: str, terms: list[tuple[str, str]], context: str,
                       style: str = "") -> str:
    """Build the official Hy-MT3 English translation instruction."""
    del style  # Hy-MT3 uses its native translation style.
    parts = []
    if context:
        parts.append(f"[Background Information]{context}")
    if terms:
        term_lines = "\n".join(f"{ja} translates to {en}" for ja, en in terms)
        parts.append(f"Reference the following translations:\n{term_lines}")
    if context:
        parts.append(
            "Please translate the following text into English, taking the provided "
            "background information into consideration.\n"
            f"[Source Text]{text}"
        )
    else:
        parts.append(
            "Translate the following text into English. Note that you should only "
            "output the translated result without any additional explanation:"
            f"{text}"
        )
    return "\n\n".join(parts)


def build_hymt3_context(header: str, history: list[tuple[str, str]], lines: int,
                        max_chars: int, *, english: bool = False) -> str:
    return build_context(
        header, history, lines, max_chars, english=english,
        dialogue_label="Previous dialogue:",
        english_translation_label="English translation: ",
    )


def clean_hymt3_output(text: str) -> str:
    """Remove an echoed Hy-MT3 prompt prefix before subtitle cleaning."""
    value = clean_output(text)
    source_marker = re.search(
        r"\[(?:Linguistic\s+)?Source Text\]\s*", value, flags=re.IGNORECASE
    )
    return value[source_marker.end():].strip() if source_marker else value


class HyMT3Translator(HymtTranslator):
    """Hy-MT3 translator reusing Hy-MT2's guarded llama.cpp lifecycle."""

    model_description = "Hy-MT3 1.8B"
    prompt_builder = staticmethod(build_hymt3_prompt)
    context_builder = staticmethod(build_hymt3_context)
    output_cleaner = staticmethod(clean_hymt3_output)


__all__ = [
    "HyMT3Translator",
    "build_hymt3_context",
    "build_hymt3_prompt",
    "clean_hymt3_output",
]