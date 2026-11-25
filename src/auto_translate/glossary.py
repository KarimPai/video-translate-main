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
_PLANET = re.compile(r"水星の魔女|惑星|太陽系|金星|火星|木星|土星|天王星|海王星|宇宙|ロケット")
AMBIGUOUS = frozenset({"ぼたん", "みこ", "まつり", "水星"})
_HONORIFIC = re.compile(r"(?:ちゃん|さん|様|さま)$")
_SUFFIXED = re.compile(r"(?:ちゃん|さん|様|さま|先輩|せんぱい|くん|君)$")


def kana_key(text: str) -> str:
    # Character-for-character folding preserves source offsets and kanji meanings.
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def _script_group(char: str) -> str | None:
    if "ぁ" <= char <= "ゖ":
        return "hiragana"
    if "ァ" <= char <= "ヶ" or "ｦ" <= char <= "ﾟ":
        return "katakana"
    if "一" <= char <= "鿿":
        return "kanji"
    if char.isascii() and char.isalnum():
        return "latin"
    return None


def _term_boundary(text: str, start: int, end: int, term: str) -> bool:
    first_group, last_group = _script_group(term[0]), _script_group(term[-1])
    left_group = _script_group(text[start - 1]) if start else None
    right_group = _script_group(text[end]) if end < len(text) else None
    left_matches_word = first_group is not None and left_group == first_group
    right_matches_word = last_group is not None and right_group == last_group
    if right_matches_word and last_group == "hiragana" and text[end] in "がはをにものとでへやも":
        right_matches_word = False
    return not left_matches_word and not right_matches_word


def nickname_boundary(text: str, start: int, end: int, *, joined=False, marine=False,
                      distinctive=False) -> bool:
    before, after = text[:start], text[end:]
    left = (not before or not before[-1].isalnum()
            or before[-1] in "がはをと" or marine or distinctive)
    return left and (joined or not after or not after[0].isalnum()
                     or bool(_PARTICLE.match(after) or _FOLLOW.match(after)))


def _given_name(en: str) -> str:
    return re.split(r"[- ]", en)[-1] if " " in en else re.split(r"-", en)[0]


_PRONOUNS = {"she": "女性", "he": "男性"}


class Glossary:
    def __init__(self, terms: dict[str, str], *, nicknames=None, talent_aliases=None,
                 talents=None, topics=None, vocabulary=(), active_topics=()) -> None:
        self.nicknames = dict(nicknames or {})
        self.talent_aliases = dict(talent_aliases or {})
        self.talents = {name: dict(profile) for name, profile in (talents or {}).items()}
        self.topics = dict(topics or {})
        self.vocabulary = tuple(vocabulary)
        self.active_topics = tuple(active_topics)
        self._base_terms = dict(terms)
        self._english = terms | self.nicknames
        # Every spelling whose English matches a profiled talent identifies that talent.
        by_english = {self._english[name]: name for name in self.talents if name in self._english}
        self._talent_spellings = sorted(
            ((kana_key(ja), by_english[en]) for ja, en in terms.items() if en in by_english),
            key=lambda pair: len(pair[0]), reverse=True)
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
        talents = data.get("talents", {})
        for name, profile in talents.items():
            if (name not in terms or not isinstance(profile, dict)
                    or set(profile) - {"self", "pronoun"}
                    or not isinstance(profile.get("self", []), list)
                    or any(not isinstance(word, str) or not word.strip() for word in profile.get("self", []))
                    or profile.get("pronoun", "she") not in _PRONOUNS):
                raise ValueError(f"Invalid talent profile: {name}")
        topics = data.get("topics", {})
        for name, topic in topics.items():
            words = [*topic.get("match", []), *topic.get("asr", [])] if isinstance(topic, dict) else [None]
            if (not isinstance(topic, dict) or set(topic) - {"match", "asr", "terms"}
                    or not topic.get("match") or not isinstance(topic.get("terms", {}), dict)
                    or any(not isinstance(word, str) or not word.strip() for word in words)
                    or any(not isinstance(v, str) for v in topic.get("terms", {}).values())):
                raise ValueError(f"Invalid glossary topic: {name}")
        return cls(terms, nicknames=nicknames, talent_aliases=aliases, talents=talents, topics=topics)

    def for_stream(self, metadata_text: str) -> "Glossary":
        """Enable topic vocabulary (e.g. GTA role-play) named in the stream title or tags."""
        folded = metadata_text.casefold()
        matched = {
            name: [word.casefold() for word in topic["match"] if word.casefold() in folded]
            for name, topic in self.topics.items()
        }
        active = [name for name, words in matched.items() if words]
        active = [name for name in active if not any(
            other != name and len(other_word) > len(word) and word in other_word
            for other in active for word in matched[name] for other_word in matched[other]
        )]
        if not active:
            return self
        terms, vocabulary = dict(self._base_terms), []
        for name in active:
            terms |= self.topics[name].get("terms", {})
            vocabulary += [word for word in self.topics[name].get("asr", []) if word not in vocabulary]
        return Glossary(terms, nicknames=self.nicknames, talent_aliases=self.talent_aliases,
                        talents=self.talents, topics=self.topics, vocabulary=vocabulary,
                        active_topics=active)

    def talents_in(self, text: str) -> list[str]:
        """Profiled talents named in Japanese or romanized text, by first mention."""
        folded, lowered = kana_key(text), text.casefold()
        found: dict[str, int] = {}
        for spelling, name in self._talent_spellings:
            index = folded.find(spelling)
            if index >= 0:
                found[name] = min(index, found.get(name, index))
        for name in self.talents:
            english = self._english[name]
            match = re.search(r"(?<![a-z])" + re.escape(english.casefold()) + r"(?![a-z])", lowered)
            if match:
                found[name] = min(match.start(), found.get(name, match.start()))
        return sorted(found, key=found.get)

    def host_line(self, name: str) -> str:
        """Profile the streamer so dropped subjects and self-references resolve."""
        profile = self.talents[name]
        line = f"配信者：{name}（{self._english[name]}"
        pronoun = profile.get("pronoun")
        if pronoun in _PRONOUNS:
            line += f"、{_PRONOUNS[pronoun]}"
        line += "）"
        if profile.get("self"):
            line += "。自称「" + "」「".join(profile["self"]) + "」は配信者本人"
        return line

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
                if (self._identity(term, en) in self._canonical_names or term in self.nicknames)
                and en[:1].isupper()]

    def _talents(self, context: str) -> frozenset[str]:
        return frozenset(self._identity(term, en) for term, en in self._matches(context, "")
                         if en[:1].isupper())

    @lru_cache(maxsize=256)  # noqa: B019
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
                # Specific context checks first: skip only when the ordinary meaning is clear.
                if ((key == "ぼたん" and _BUTTON.search(kana_key(source)))
                        or (key == "まつり" and _FESTIVAL.search(kana_key(source)))
                        or (key.startswith("水星") and _PLANET.search(source))):
                    continue
                # Ambiguous nicknames without a specific guard still need talent context.
                if key in AMBIGUOUS and self._identity(term, en) not in talents:
                    if en == "Miko" and _SHRINE.search(source):
                        continue
                    # Botan can be a name when no button context is present; Matsuri
                    # and Suisei require a confirmed talent identity.
                    if key != "ぼたん":
                        continue
                matches = []
                for m in re.finditer(re.escape(needle), text):
                    repeated = kana_key(source.strip("。！？!? ")) == key * 2
                    marine_title = key == "船長" and source[:m.start()].endswith("マリン")
                    following = kana_key(source[m.end():])
                    joined_names = any(following.startswith((other, "と" + other))
                                       for other in self._nickname_keys)
                    if key == "みこ" and following.startswith("と") and not joined_names:
                        continue
                    distinctive = len(key) >= 3 and key not in AMBIGUOUS
                    if repeated or nickname_boundary(source, m.start(), m.end(),
                                                      joined=joined_names or bool(_SUFFIXED.search(key)),
                                                      marine=marine_title, distinctive=distinctive):
                        matches.append(m)
                if matches:
                    hits.append((source[matches[0].start():matches[0].end()], en))
                    for m in reversed(matches):
                        ja = ja[:m.start()] + "\0" * len(needle) + ja[m.end():]
                continue
            matches = [m for m in re.finditer(re.escape(needle), text)
                       if _term_boundary(text, m.start(), m.end(), term)]
            if matches:
                hits.append((source[matches[0].start():matches[0].end()], en))
                for m in reversed(matches):
                    ja = ja[:m.start()] + "\0" * len(needle) + ja[m.end():]
        return tuple(hits)
