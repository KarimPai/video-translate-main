"""Qwen3.5-4B is the sole local Japanese-to-English translation engine."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .qwen import QwenTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> Translator:
    """Create the local Qwen translator with the existing terminology glossary."""
    return QwenTranslator(cfg, glossary, vod_title=vod_title)
