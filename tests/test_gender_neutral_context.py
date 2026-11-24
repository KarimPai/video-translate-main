"""Context prompt gender is included only when explicitly configured."""
from __future__ import annotations

from auto_translate.glossary import Glossary


def test_host_context_does_not_assume_a_pronoun():
    glossary = Glossary(
        {"さくらみこ": "Sakura Miko"},
        talent_aliases={"みこ": "さくらみこ"},
        talents={"さくらみこ": {}},
    )

    assert glossary.host_line("さくらみこ") == "配信者：さくらみこ（Sakura Miko）"


def test_host_context_uses_only_an_explicit_profile_pronoun():
    glossary = Glossary(
        {"さくらみこ": "Sakura Miko"},
        talents={"さくらみこ": {"pronoun": "she"}},
    )

    assert "女性" in glossary.host_line("さくらみこ")