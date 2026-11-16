"""JA → EN fixed terms from glossary.toml."""

import tomllib
import re
import logging
from functools import lru_cache
from pathlib import Path

from .name_context import (MikoMention, NameMention, is_miko_target, is_name_target,
                           resolve_miko_mentions, resolve_matsuri_mentions)


# Japanese has no whitespace between most words. Favor missed hints over
# potentially matching stems inside compounds. Only common post-name syntax
# is admitted; the original (unmasked) text determines each boundary.
_PARTICLE = re.compile(r"(?:が|は|を|に|も|の|と|で)(?=$|[^ぁ-ん]|いる|いた|きた|どこ|いっしょ)")
_FOLLOW = re.compile(r"来[たてるな]|いた|いる|すご[いく]|かわいい|おはよう|どこ|って|だよ|です|だね|だ[。！？!?]?$")
_SHIP = re.compile(r"海賊船|船員|乗組員|乗客|出航|航海|操舵|操縦|甲板|港|この船|船の|フェリー")


def nickname_boundary(text: str, start: int, end: int, *, joined=False, marine=False) -> bool:
    before, after = text[:start], text[end:]
    left = (not before or not before[-1].isalnum()
            or before[-1] in "がはをと" or marine)
    return left and (joined or not after or not after[0].isalnum()
                     or bool(_PARTICLE.match(after) or _FOLLOW.match(after)))


class Glossary:
    def __init__(self, terms: dict[str, str], *, scene_pairs=(), name_guards=(), nicknames=None) -> None:
        # Longest first, so "てぇてぇ" is matched before any shorter term inside it.
        self.nicknames = dict(nicknames or {})
        self.terms = sorted((terms | self.nicknames).items(), key=lambda kv: len(kv[0]), reverse=True)
        self._miko_aliases = {source: target for source, target in self.terms if is_miko_target(target)}
        self._configured_terms = dict(self.terms)
        # Optional ASR-only rules: acoustic peer comparison, not inferred words.
        self.scene_pairs = tuple(
            (rule["heard"], rule["correct"], tuple(rule.get("anchors", ())))
            for rule in scene_pairs
        )
        self.name_guards = tuple(
            (re.compile(rule["full"]), re.compile(rule["mention"]))
            for rule in name_guards
        )

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            logging.getLogger(__name__).warning("Glossary missing: %s", path)
            return cls({})
        with path.open("rb") as f:
            data = tomllib.load(f)
            rules = data.get("asr_rules", {})
            return cls(data.get("terms", {}),
                       nicknames=data.get("nicknames"),
                       scene_pairs=rules.get("scene_pairs", ()),
                       name_guards=rules.get("name_guards", ()))

    def miko_mentions(self, ja: str, *, previous_ja: str = "", miko_context: bool = False,
                      video_context: str = "") -> tuple[MikoMention, ...]:
        return resolve_miko_mentions(ja, self._miko_aliases, previous_ja=previous_ja,
                                     miko_context=miko_context, video_context=video_context)

    def find(self, ja: str, *, previous_ja: str = "", miko_context: bool = False,
             video_context: str = "") -> list[tuple[str, str]]:
        """Entries that appear in `ja` (non-overlapping, longest match wins)."""
        return list(self._matches(ja, previous_ja, miko_context, video_context))

    def matsuri_mentions(self, ja: str, *, previous_ja: str = "",
                         video_context: str = "") -> tuple[NameMention, ...]:
        return resolve_matsuri_mentions(ja, self._configured_terms,
                                       previous_ja=previous_ja, video_context=video_context)

    def name_mentions(self, ja: str, *, previous_ja: str = "", miko_context: bool = False,
                       video_context: str = "") -> tuple[NameMention, ...]:
        mentions = self.miko_mentions(ja, previous_ja=previous_ja, miko_context=miko_context,
                                      video_context=video_context)
        mentions += self.matsuri_mentions(ja, previous_ja=previous_ja, video_context=video_context)
        return tuple(sorted(mentions, key=lambda item: item.start))

    @lru_cache(maxsize=128)
    def _matches(self, ja: str, previous_ja: str, miko_context: bool,
                 video_context: str) -> tuple[tuple[str, str], ...]:
        source = ja
        hits = []
        mentions = self.name_mentions(source, previous_ja=previous_ja,
                                      miko_context=miko_context, video_context=video_context)
        for mention in sorted(mentions, key=lambda item: item.end - item.start, reverse=True):
            pair = (mention.source, mention.target)
            # A term glossary applies to every occurrence. Never give it a
            # global name mapping when another occurrence denotes the role.
            occurrences = list(re.finditer(re.escape(mention.source), source))
            if pair not in hits and all(any(
                item.start <= match.start() and match.end() <= item.end for item in mentions
            ) for match in occurrences):
                hits.append(pair)
            ja = ja[:mention.start] + "\0" * (mention.end - mention.start) + ja[mention.end:]
        for term, en in self.terms:
            if is_name_target(en):
                continue  # The shared span resolver owns these ambiguous names.
            if term not in ja:
                continue
            if term in self.nicknames:
                if term == "船長" and _SHIP.search(source):
                    en = "captain"
                elif term == "船長" and not (
                    "マリン船長" in source or "宝鐘マリン" in source
                    or ("宝鐘マリン" in previous_ja and not _SHIP.search(previous_ja))
                ):
                    continue
                matches = []
                for m in re.finditer(re.escape(term), ja):
                    repeated = source.strip("。！？!? ") == term * 2
                    marine_title = term == "船長" and source[:m.start()].endswith("マリン")
                    following = source[m.end():]
                    joined_names = (following.startswith("と") and any(
                        following.startswith(other, 1) for other in self.nicknames
                    ))
                    if repeated or nickname_boundary(source, m.start(), m.end(),
                                                     joined=joined_names, marine=marine_title):
                        matches.append(m)
                if matches:
                    hits.append((term, en))
                    for m in reversed(matches):
                        ja = ja[:m.start()] + "\0" * len(term) + ja[m.end():]
                continue
            if re.fullmatch(r"[一-鿿]", term):
                # A one-kanji hint may be useful with particles, but changes
                # meaning inside compounds such as 素敵, 石油 or 草原.
                pattern = rf"(?<![一-鿿]){re.escape(term)}(?![一-鿿])"
                ja, count = re.subn(pattern, "\0", ja)
                if count:
                    hits.append((term, en))
            else:
                hits.append((term, en))
                ja = ja.replace(term, "\0" * len(term))  # mask it so shorter terms can't match inside
        return tuple(hits)
