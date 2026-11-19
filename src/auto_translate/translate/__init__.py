"""Gemma 4 E4B is the Japanese-to-English translation engine with reasoning support."""

from ..config import TranslateConfig
from ..glossary import Glossary
from ..stream_context import StreamContext
from .base import Translator
from .gemma4 import Gemma4Translator


def create_translator(cfg: TranslateConfig, glossary: Glossary, *, stream: StreamContext | None = None,
                      gpu_layers: int | None = None) -> Translator:
    return Gemma4Translator(cfg, glossary, stream=stream, gpu_layers=gpu_layers)
