from ..config import HyMt2Config, TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .hymt2 import HyMt2Translator


def create_translator(cfg: TranslateConfig, glossary: Glossary) -> Translator | None:
    """Build the sole supported translation engine: HY-MT2."""
    if cfg.preset == "none":
        return None
    if cfg.preset != "hymt2":
        raise SystemExit(
            f"unsupported translation preset {cfg.preset!r}; "
            "only 'hymt2' and 'none' are supported"
        )

    settings = dict(cfg.presets.get("hymt2", {}))
    backend = settings.pop("backend", "hymt2")
    settings.pop("mpv_delay_s", None)
    settings.pop("api_key_env", None)  # HY-MT2 is local-only; no cloud API keys.

    # Accept the previous backend label in existing local config files,
    # but never permit a non-HY-MT2 model.
    if backend not in ("hymt2", "openai-compat"):
        raise SystemExit(f"unsupported HY-MT2 backend {backend!r}")
    model = str(settings.get("model", "kaelri/hy-mt2:1.8b-q4_K_M"))
    if "hy-mt2" not in model.lower():
        raise SystemExit(
            f"only HY-MT2 is supported for translation; configured model is {model!r}"
        )

    return HyMt2Translator(HyMt2Config(**settings), glossary)
