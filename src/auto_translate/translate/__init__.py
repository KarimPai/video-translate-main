from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .hymt2 import HyMt2Translator
from .plamo import PlamoTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary) -> Translator:
    """Build the configured local translation backend."""
    backend = cfg.backend.strip().lower()
    if backend == "hymt2":
        if "hy-mt2" not in cfg.model.lower():
            raise SystemExit(
                f"HY-MT2 backend requires a HY-MT2 model, got {cfg.model!r}"
            )
        return HyMt2Translator(cfg, glossary)
    if backend == "plamo":
        return PlamoTranslator(cfg, glossary)
    raise SystemExit(
        f"unsupported translation backend {cfg.backend!r}; choose 'hymt2' or 'plamo'"
    )
