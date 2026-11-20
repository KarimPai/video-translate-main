"""Translation package — backends and factory."""

from .base import TranslationBackend, TranslatorUnavailable
from .gemma4 import Gemma4Translator, useful_context, build_prompt, clean_output
from .plamo import PlamoTranslator, build_plamo_prompt
from .hymt import HymtTranslator, build_hymt_prompt

__all__ = ["TranslationBackend", "TranslatorUnavailable", "Gemma4Translator", "PlamoTranslator",
           "HymtTranslator", "useful_context", "build_prompt", "clean_output", "build_plamo_prompt",
           "build_hymt_prompt"]
