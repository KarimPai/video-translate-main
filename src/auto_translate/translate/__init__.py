"""CAT-Translate-3.3B: sole offline Japanese-to-English translator."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .cat import CatTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> Translator:
    return CatTranslator(cfg, glossary, vod_title=vod_title)
