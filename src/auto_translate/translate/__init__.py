"""Tencent HY-MT1.5-7B is the only Japanese-to-English translation engine."""

from ..config import TranslateConfig
from ..glossary import Glossary
from .base import Translator
from .hymt import HymtTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, vod_title: str = "",
                      gpu_layers: int | None = None) -> Translator:
    return HymtTranslator(cfg, glossary, vod_title=vod_title, gpu_layers=gpu_layers)
