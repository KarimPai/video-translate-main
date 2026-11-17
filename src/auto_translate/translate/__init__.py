"""Tencent Hy-MT2-7B is the only Japanese-to-English translation engine."""

from ..config import TranslateConfig
from ..glossary import Glossary
from ..stream_context import StreamContext
from .base import Translator
from .hymt import HymtTranslator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, stream: StreamContext | None = None,
                      gpu_layers: int | None = None) -> Translator:
    return HymtTranslator(cfg, glossary, stream=stream, gpu_layers=gpu_layers)
