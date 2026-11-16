"""Source-span evidence for Miko's name versus the homophone shrine maiden.

The glossary supplies the aliases. Metadata/recent Japanese can disambiguate a
mention already in the current speech; they never add a missing name. No state
is kept here, so callers can discard recent evidence on a seek or new session.
"""

from dataclasses import dataclass
import re
from typing import Mapping


_MIKO_TARGET = re.compile(r"(?:Sakura )?Miko(?:chi)?(?:-(?:chan|san|sama))?\Z", re.I)
_HOLO = re.compile(r"ホロ|holo", re.I)
_TITLE_IDENTITY = re.compile(r"さくら[みミ][こコ]|サクラ[みミ][こコ]|Sakura\s+Miko|\bMiko\s+Ch\b", re.I)
_BARE = frozenset({"みこ", "ミコ", "巫女"})
_PUNCTUATION = " \r\n\t。、，,.!！?？「」『』（）()：:"
_DIRECT_ADDRESS = re.compile(r"(?:ねえ|ねぇ|おい|ほら|あの|もしもし)[、，, ]*\Z")
_FOLLOW = re.compile(r"来[たてるな]|いた|いる|すご[いく]|かわいい|おはよう|どこ|って|だよ|です|だね|だ[。！？!?]?$")
_WITH_PERSON = re.compile(r"と(?:話|遊|会|通話|一緒|いっしょ)")
_LITERAL_BEFORE = re.compile(r"(?:神社|神主|神職|お祓い|神楽)(?:の|にいる|で働く)[^。、！？!?\n]{0,5}\Z")
_LITERAL_AFTER = re.compile(r"(?:アイドル|装束|衣装|舞|服|として|にな[るりっ])")
_SHRINE_TOPIC = re.compile(
    r"神社|神主|神職|お祓い|神楽|"
    r"巫女(?:さん|ちゃん|様|さま|アイドル|装束|衣装|舞|とは|として|の(?:仕事|役目|役割|職業|衣装|装束|舞))"
)
_SOURCE_TOPIC_WINDOW = 96
_RECENT_TOPIC_WINDOW = 240
_KANJI_PERSON = re.compile(
    r"(?:(?:が|は|も)(?:来|きた|来ない|い[るた]|どこ|どう|何して|言|話|喋|しゃべ|配信|ログイン|帰|戻|やっ)"
    r"|(?:に|と)(?:会|話|遊|聞|通話)|どこ|来[たてるな]|って(?:言|話))"
)


@dataclass(frozen=True)
class MikoMention:
    start: int
    end: int
    source: str
    target: str
    evidence: str


def is_miko_target(target: str) -> bool:
    return bool(_MIKO_TARGET.fullmatch(target))


def _distinctive_name(mention: MikoMention) -> bool:
    # Kana + an honorific can still politely address a literal shrine maiden.
    # Only the full stage name or distinctive Mikochi nickname can establish
    # identity without Hololive/name metadata.
    target = mention.target.casefold()
    return mention.source != "巫女" and (target.startswith("sakura miko") or target.startswith("mikochi"))


def _left_boundary(text: str, start: int) -> bool:
    before = text[:start]
    return (not before or not before[-1].isalnum()
            or before[-1] in "がはをとにものでも"
            or bool(_DIRECT_ADDRESS.search(before)))


def _right_boundary(text: str, end: int, source: str, aliases: Mapping[str, str]) -> bool:
    after = text[end:]
    if not after or not after[0].isalnum():
        return True
    if _FOLLOW.match(after):
        return True
    if after[0] in "がはをにもので":
        return True
    if after.startswith("と"):
        # Bare みこと can be the complete name Mikoto. Admit only an
        # identifiable conjunction or a verb taking a named companion.
        return (source not in _BARE or bool(_WITH_PERSON.match(after))
                or any(after.startswith(other, 1) for other in aliases))
    return False


def _literal_use(text: str, start: int, end: int, source: str) -> bool:
    before, after = text[:start], text[end:]
    if _LITERAL_BEFORE.search(before) or _LITERAL_AFTER.match(after):
        return True
    if source == "巫女":
        # 職業/role compounds and the ordinary polite role 巫女さん must
        # survive even when the stream title names Sakura Miko.
        return (after.startswith(("さん", "ちゃん", "様", "さま"))
                or bool(before and re.match(r"[一-鿿]", before[-1]))
                or bool(after and re.match(r"[一-鿿]", after[0])))
    return False


def _alias_mentions(text: str, aliases: Mapping[str, str]) -> tuple[MikoMention, ...]:
    candidates = []
    for source, target in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(re.escape(source), text):
            repeated = text.strip(_PUNCTUATION) in (source * 2, source * 3)
            if ((repeated or (_left_boundary(text, match.start())
                              and _right_boundary(text, match.end(), source, aliases)))
                    and (target.startswith("Sakura ")
                         or not _literal_use(text, match.start(), match.end(), source))):
                if not any(match.start() < item.end and match.end() > item.start for item in candidates):
                    candidates.append(MikoMention(match.start(), match.end(), source, target, "alias"))
    return tuple(sorted(candidates, key=lambda item: item.start))


def _nearby_shrine_topic(text: str, start: int, end: int, previous_ja: str) -> bool:
    # A neighboring sentence or recent shrine discussion can establish the
    # literal role even when the current name-shaped clause is short. Limit
    # the negative evidence so old topics do not suppress names indefinitely.
    nearby = text[max(0, start - _SOURCE_TOPIC_WINDOW):start] + text[end:end + _SOURCE_TOPIC_WINDOW]
    return bool(_SHRINE_TOPIC.search(nearby)
                or _SHRINE_TOPIC.search(previous_ja[-_RECENT_TOPIC_WINDOW:]))


def resolve_miko_mentions(
    text: str,
    aliases: Mapping[str, str],
    *,
    previous_ja: str = "",
    miko_context: bool = False,
    video_context: str = "",
) -> tuple[MikoMention, ...]:
    """Return only evidenced current-source names, preserving source offsets.

    A Hololive context permits familiar kana spellings. The kanji homophone
    requires the stronger identity evidence of an explicit full name/nickname
    plus name-like syntax; a general Hololive title alone is insufficient.
    Nearby shrine-topic evidence keeps this ambiguous kanji reading literal.
    """
    aliases = {source: target for source, target in aliases.items()
               if source and is_miko_target(target)}
    if not aliases:
        return ()
    current = _alias_mentions(text, aliases)
    recent = _alias_mentions(previous_ja, aliases)
    identity = bool(_TITLE_IDENTITY.search(video_context)
                    or any(_distinctive_name(mention) for mention in current + recent))
    honorific_context = bool(miko_context or _HOLO.search(video_context) or identity)
    recent_bare_name = any(
        item.source in ("みこ", "ミコ")
        and _KANJI_PERSON.match(previous_ja[item.end:])
        and not _nearby_shrine_topic(previous_ja, item.start, item.end, "")
        for item in recent
    )
    named_context = honorific_context or recent_bare_name
    mentions = [MikoMention(item.start, item.end, item.source, item.target,
                            "explicit_name" if _distinctive_name(item) else "name_context")
                for item in current
                if (_distinctive_name(item)
                    or (item.source in ("みこ", "ミコ") and named_context)
                    or (item.source not in _BARE and honorific_context))]

    # Do not derive a short target when the configured glossary only contains
    # a full name, or when the user has removed/renamed the Miko aliases.
    if identity and "Miko" in aliases.values():
        for match in re.finditer("巫女", text):
            if (not _left_boundary(text, match.start())
                    or _literal_use(text, match.start(), match.end(), "巫女")
                    or _nearby_shrine_topic(text, match.start(), match.end(), previous_ja)
                    or any(match.start() < item.end and match.end() > item.start for item in mentions)):
                continue
            after = text[match.end():]
            isolated = text.strip(_PUNCTUATION) == "巫女"
            if isolated or _KANJI_PERSON.match(after):
                mentions.append(MikoMention(match.start(), match.end(), "巫女", "Miko", "homophone_identity"))
    return tuple(sorted(mentions, key=lambda item: item.start))
