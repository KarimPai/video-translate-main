"""Context-gated Japanese terminology and source-backed talent aliases."""

import logging
import re
import tomllib
from functools import lru_cache
from pathlib import Path


_PARTICLE = re.compile(r"(?:が|は|を|に|も|の|と|で)(?=$|[^ぁ-ん]|いる|いた|きた|どこ|いっしょ)")
_FOLLOW = re.compile(r"来[たてるな]|いた|いる|すご[いく]|かわいい|おはよう|どこ|って|だよ|です|だね|だ[。！？!?]?$")
_SHIP = re.compile(r"海賊船|船員|乗組員|乗客|出航|航海|操舵|操縦|甲板|港|この船|船の|フェリー")
_SHRINE = re.compile(r"神社|神主|神職|巫女|お祓い")
_BUTTON = re.compile(r"ぼたん(?:を|は|が|に)?(?:押|おす|くりっく|連打)|"
                     r"(?:電源|再生|停止|決定|戻る|すたーと|りせっと|[ABXYLR])ぼたん")
_FESTIVAL = re.compile(r"(?:まつり|祭り)(?:に|へ)(?:行|い[くこ])")
AMBIGUOUS = frozenset({"ぼたん", "みこ", "まつり", "祭り"})
_HONORIFIC = re.compile(r"(?:ちゃん|さん|様|さま)$")


def kana_key(text: str) -> str:
    # Character-for-character folding preserves source offsets and kanji meanings.
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def nickname_boundary(text: str, start: int, end: int, *, joined=False, marine=False,
                      distinctive=False) -> bool:
    before, after = text[:start], text[end:]
    left = (not before or not before[-1].isalnum()
            or before[-1] in "がはをと" or marine or distinctive)
    return left and (joined or not after or not after[0].isalnum()
                     or bool(_PARTICLE.match(after) or _FOLLOW.match(after)))


def _given_name(en: str) -> str:
    return re.split(r"[- ]", en)[-1] if " " in en else re.split(r"-", en)[0]


class Glossary:
    def __init__(self, terms: dict[str, str], *, nicknames=None, talent_aliases=None) -> None:
        self.nicknames = dict(nicknames or {})
        self.talent_aliases = dict(talent_aliases or {})
        self._name_ids = {kana_key(alias): canonical for alias, canonical in self.talent_aliases.items()}
        self._canonical_names = frozenset(self.talent_aliases.values())
        self._name_ids.update((kana_key(name), name) for name in self._canonical_names)
        self.terms = sorted((terms | self.nicknames).items(), key=lambda kv: len(kv[0]), reverse=True)
        self._nickname_keys = frozenset(kana_key(term) for term in self.nicknames)

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            logging.getLogger(__name__).warning("Glossary missing: %s", path)
            return cls({})
        with path.open("rb") as f:
            data = tomllib.load(f)
        aliases = data.get("talent_aliases", {})
        terms = data.get("terms", {})
        nicknames = data.get("nicknames", {})
        for alias, canonical in aliases.items():
            if (alias not in terms and alias not in nicknames) or canonical not in terms:
                raise ValueError(f"Invalid talent alias: {alias} -> {canonical}")
        return cls(terms, nicknames=nicknames, talent_aliases=aliases)

    def _identity(self, term: str, en: str) -> str:
        key = kana_key(term)
        return (self._name_ids.get(key) or self._name_ids.get(_HONORIFIC.sub("", key))
                or _given_name(en))

    def proper_names(self, context: str | None = None) -> list[str]:
        talents = self._talents(context) if context is not None else None
        names, seen = [], set()
        for term, en in self.terms:
            if kana_key(term) in AMBIGUOUS or not en[:1].isupper():
                continue
            talent = self._identity(term, en)
            if talents is not None and (term in self.nicknames or talent not in talents or talent in seen):
                continue
            if talent in self._canonical_names and term != talent:
                continue
            names.append(term)
            seen.add(talent)
        return names

    def recognition_names(self, context: str) -> list[str]:
        canonical = self.proper_names(context)[:4]
        names = list(canonical)
        for name in canonical:
            for alias, identity in self.talent_aliases.items():
                if identity == name and kana_key(alias) not in AMBIGUOUS and alias != "船長":
                    names.append(alias)
                    break
        return names

    def find(self, ja: str, *, context: str = "") -> list[tuple[str, str]]:
        return list(self._matches(ja, context))

    def name_terms(self, ja: str, *, context: str = "") -> list[tuple[str, str]]:
        return [(term, en) for term, en in self._matches(ja, context)
                if self._identity(term, en) in self._canonical_names and en[:1].isupper()]

    def _talents(self, context: str) -> frozenset[str]:
        return frozenset(self._identity(term, en) for term, en in self._matches(context, "")
                         if en[:1].isupper())

    @lru_cache(maxsize=256)
    def _matches(self, ja: str, context: str) -> tuple[tuple[str, str], ...]:
        source = ja
        talents = self._talents(context) if context else frozenset()
        hits = []
        for term, en in self.terms:
            key = kana_key(term)
            is_name = term in self.nicknames or self._identity(term, en) in self._canonical_names
            text = kana_key(ja) if is_name else ja
            needle = key if is_name else term
            if needle not in text:
                continue
            if term in self.nicknames:
                if key == "船長" and _SHIP.search(source):
                    en = "captain"
                elif key == "船長" and not ("マリン" in source or self._identity(term, en) in talents):
                    continue
                if key in AMBIGUOUS and (self._identity(term, en) not in talents
                                         or (en == "Miko" and _SHRINE.search(source))):
                    continue
                if ((key == "ぼたん" and _BUTTON.search(kana_key(source)))
                        or (key in {"まつり", "祭り"} and _FESTIVAL.search(kana_key(source)))):
                    continue
                matches = []
                for m in re.finditer(re.escape(needle), text):
                    repeated = kana_key(source.strip("。！？!? ")) == key * 2
                    marine_title = key == "船長" and source[:m.start()].endswith("マリン")
                    following = kana_key(source[m.end():])
                    joined_names = any(following.startswith(other) or following.startswith("と" + other)
                                       for other in self._nickname_keys)
                    if key == "みこ" and following.startswith("と") and not joined_names:
                        continue
                    distinctive = len(key) >= 3 and key not in AMBIGUOUS
                    if repeated or nickname_boundary(source, m.start(), m.end(), joined=joined_names,
                                                      marine=marine_title, distinctive=distinctive):
                        matches.append(m)
                if matches:
                    hits.append((source[matches[0].start():matches[0].end()], en))
                    for m in reversed(matches):
                        ja = ja[:m.start()] + "\0" * len(needle) + ja[m.end():]
                continue
            if re.fullmatch(r"[一-鿿]", term):
                pattern = rf"(?<![一-鿿]){re.escape(term)}(?![一-鿿])"
                ja, count = re.subn(pattern, "\0", ja)
                if count:
                    hits.append((term, en))
            else:
                matches = list(re.finditer(re.escape(needle), text))
                hits.append((source[matches[0].start():matches[0].end()], en))
                for m in reversed(matches):
                    ja = ja[:m.start()] + "\0" * len(needle) + ja[m.end():]
        return tuple(hits)
