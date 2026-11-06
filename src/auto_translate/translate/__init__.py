from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .plamo import PlamoTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary) -> Translator:
    """Build the application's only supported translator: PLaMo 3 Translate."""
    return PlamoTranslator(cfg, glossary)
