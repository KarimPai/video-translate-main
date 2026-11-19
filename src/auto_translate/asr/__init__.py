"""The application's single local Japanese ASR backend."""
from ..config import AsrConfig
from ..glossary import Glossary
from .base import ASR
from .haas import HaasASR


def asr_context(cfg: AsrConfig, glossary: Glossary | None = None, title: str = "") -> str:
    """User context and bounded canonical-name/alias hints selected by metadata."""
    text = cfg.context.strip()
    names = glossary.recognition_names("\n".join((title, text))) if glossary else []
    # Keep metadata-selected hints small to avoid echoing the talent dictionary.
    budget = cfg.max_context_chars - len(text) - 4
    kept = []
    for name in names:
        if budget - len(name) - 1 < 0:
            break
        kept.append(name)
        budget -= len(name) + 1
    if kept:
        text = (text + "\n" if text else "") + "人名：" + "、".join(kept)
    words = []
    # A stream tag alone is weak evidence for any particular utterance. Keep
    # the default shortlist small and give terms actually in the title priority.
    vocabulary = glossary.vocabulary if glossary else ()
    for word in sorted(vocabulary, key=lambda word: word not in title)[:4]:
        if budget - len(word) - 5 < 0:
            break
        words.append(word)
        budget -= len(word) + 1
    if words:
        text = (text + "\n" if text else "") + "用語：" + "、".join(words)
    return text[: cfg.max_context_chars]


def create_asr(cfg: AsrConfig, glossary: Glossary | None = None, title: str = "") -> ASR:
    return HaasASR(cfg, asr_context(cfg, glossary, title))
