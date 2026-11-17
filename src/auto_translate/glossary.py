"""JA → EN fixed terms from glossary.toml."""

import tomllib
import re
import logging
from functools import lru_cache
from pathlib import Path

from .proper_names import name_glossary, resolve_names


# Japanese has no whitespace between most words. Favor missed hints over
# potentially matching stems inside compounds. Only common post-name syntax
# is admitted; the original (unmasked) text determines each boundary.
_PARTICLE = re.compile(r"(?:が|は|を|に|も|の|と|で)(?=$|[^ぁ-ん]|いる|いた|きた|どこ|いっしょ)")
_FOLLOW = re.compile(r"来[たてるな]|いた|いる|すご[いく]|かわいい|おはよう|どこ|って|だよ|です|だね|だ[。！？!?]?$")
_SHIP = re.compile(r"海賊船|船員|乗組員|乗客|出航|航海|操舵|操縦|甲板|港|この船|船の|フェリー")
_SHRINE = re.compile(r"神社|神主|神職|巫女|お祓い")


def nickname_boundary(text: str, start: int, end: int, *, joined=False, marine=False) -> bool:
    before, after = text[:start], text[end:]
    left = (not before or not before[-1].isalnum()
            or before[-1] in "がはをと" or marine)
    return left and (joined or not after or not after[0].isalnum()
                     or bool(_PARTICLE.match(after) or _FOLLOW.match(after)))


class Glossary:
    def __init__(self, terms: dict[str, str], *, nicknames=None) -> None:
        # Longest first, so "てぇてぇ" is matched before any shorter term inside it.
        self.nicknames = dict(nicknames or {})
        self.terms = sorted((terms | self.nicknames).items(), key=lambda kv: len(kv[0]), reverse=True)

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            logging.getLogger(__name__).warning("Glossary missing: %s", path)
            return cls({})
        with path.open("rb") as f:
            data = tomllib.load(f)
            return cls(data.get("terms", {}),
                       nicknames=data.get("nicknames"))

    def find(self, ja: str, *, previous_ja: str = "", miko_context: bool = False,
             name_mentions=None) -> list[tuple[str, str]]:
        """Entries that appear in `ja` (non-overlapping, longest match wins)."""
        # `miko_context` remains accepted for settings/API compatibility. A
        # boolean derived from a title is no longer proof of a spoken person.
        mentions = (resolve_names(ja, previous_ja=previous_ja)
                    if name_mentions is None else name_mentions)
        matches = list(self._matches(ja, previous_ja if "船長" in ja else "", False))
        configured = {source for source, _ in self.terms}
        matches.extend(pair for pair in name_glossary(mentions)
                       if pair[0] in configured and pair not in matches)
        return sorted(matches, key=lambda pair: len(pair[0]), reverse=True)

    @lru_cache(maxsize=128)
    def _matches(self, ja: str, previous_ja: str, miko_context: bool) -> tuple[tuple[str, str], ...]:
        source = ja
        hits = []
        for term, en in self.terms:
            if term not in ja:
                continue
            if re.fullmatch(r"(?:Natsuiro )?Matsuri(?:-(?:chan|san|sama))?|(?:Sakura )?Miko(?:chi)?(?:-(?:chan|san|sama))?", en):
                # Central resolution owns these ambiguous families, including
                # kanji spellings and mixed person/common-noun occurrences.
                continue
            if term in self.nicknames:
                if term == "船長" and _SHIP.search(source):
                    en = "captain"
                elif term == "船長" and not (
                    "マリン船長" in source or "宝鐘マリン" in source
                    or ("宝鐘マリン" in previous_ja and not _SHIP.search(previous_ja))
                ):
                    continue
                if en == "Miko" and (not miko_context or _SHRINE.search(source)):
                    continue
                matches = []
                for m in re.finditer(re.escape(term), ja):
                    repeated = source.strip("。！？!? ") == term * 2
                    marine_title = term == "船長" and source[:m.start()].endswith("マリン")
                    following = source[m.end():]
                    joined_names = (following.startswith("と") and any(
                        following.startswith(other, 1) for other in self.nicknames
                    ))
                    if term == "みこ" and following.startswith("と") and not joined_names:
                        continue  # みこと is also a complete name/word.
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
