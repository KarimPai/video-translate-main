"""The application's single local Japanese ASR backend."""
from ..config import AsrConfig
from ..glossary import Glossary
from .base import ASR
from .qwen3 import Qwen3ASR


def asr_context(cfg: AsrConfig, glossary: Glossary | None = None, title: str = "") -> str:
    """User context and canonical names selected by stream metadata."""
    text = cfg.context.strip()
    names = glossary.proper_names("\n".join((title, text))) if glossary else []
    # Raw titles and nickname lists can be echoed as speech by Qwen3-ASR.
    budget = cfg.max_context_chars - len(text) - 4
    kept = []
    for name in names:
        if budget - len(name) - 1 < 0:
            break
        kept.append(name)
        budget -= len(name) + 1
    if kept:
        text = (text + "\n" if text else "") + "人名：" + "、".join(kept)
    return text[: cfg.max_context_chars]


def create_asr(cfg: AsrConfig, glossary: Glossary | None = None, title: str = "") -> ASR:
    return Qwen3ASR(cfg, asr_context(cfg, glossary, title))
