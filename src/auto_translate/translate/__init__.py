from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .hymt2 import HyMt2Translator


def create_translator(cfg: TranslateConfig, glossary: Glossary) -> Translator:
    """Build the application's single supported translator: HY-MT2."""
    if "hy-mt2" not in cfg.model.lower():
        raise SystemExit(
            f"only HY-MT2 is supported for translation; configured model is {cfg.model!r}"
        )
    return HyMt2Translator(cfg, glossary)
