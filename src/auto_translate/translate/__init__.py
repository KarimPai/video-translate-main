"""Index-Translate-2B is the sole local Japanese-to-English translator."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .index import IndexTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> Translator:
    return IndexTranslator(cfg, glossary, vod_title=vod_title)
