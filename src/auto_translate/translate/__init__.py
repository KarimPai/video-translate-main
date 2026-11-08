"""PLaMo 3 Translate 8B Q3_K_M: sole offline Japanese-to-English translator."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .plamo import PlamoTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> Translator:
    return PlamoTranslator(cfg, glossary, vod_title=vod_title)
