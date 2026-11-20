"""Translation package — backends and factory."""

from .base import TranslationBackend, TranslatorUnavailable
from .gemma4 import Gemma4Translator, build_prompt, clean_output, useful_context
from .hymt import HymtTranslator, build_hymt_prompt
from .plamo import PlamoTranslator, build_plamo_prompt

__all__ = [
           "Gemma4Translator",
           "HymtTranslator",
           "PlamoTranslator",
           "TranslationBackend",
           "TranslatorUnavailable",
           "build_hymt_prompt",
           "build_plamo_prompt",
           "build_prompt",
           "clean_output",
           "useful_context",
]
