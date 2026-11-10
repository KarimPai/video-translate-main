"""PLaMo 3 Translate 2B is the only Japanese-to-English translation engine."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .plamo import PlamoTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "") -> Translator:
    """Construct the local PLaMo translator with video context and glossary."""
    return PlamoTranslator(cfg, glossary, vod_title=vod_title)
