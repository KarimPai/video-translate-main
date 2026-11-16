"""Source-span evidence for Miko and Matsuri versus their ordinary meanings.

The glossary supplies the aliases. Metadata/recent Japanese can disambiguate a
mention already in the current speech; they never add a missing name. No state
is kept here, so callers can discard recent evidence on a seek or new session.
"""

from dataclasses import dataclass
import re
from typing import Mapping


_MIKO_TARGET = re.compile(r"(?:Sakura )?Miko(?:chi)?(?:-(?:chan|san|sama))?\Z", re.I)
_MATSURI_TARGET = re.compile(r"(?:Natsuiro )?Matsuri(?:-(?:chan|san|sama))?\Z", re.I)
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
class NameMention:
    start: int
    end: int
    source: str
    target: str
    evidence: str


# Retain the original annotation API for callers interested only in Miko.
MikoMention = NameMention


def is_miko_target(target: str) -> bool:
    return bool(_MIKO_TARGET.fullmatch(target))


def is_matsuri_target(target: str) -> bool:
    return bool(_MATSURI_TARGET.fullmatch(target))


def is_name_target(target: str) -> bool:
    return is_miko_target(target) or is_matsuri_target(target)


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


_MATSURI_TITLE = re.compile(r"夏色(?:まつり|マツリ|祭り)|Natsuiro\s+Matsuri|\bMatsuri\s+Ch\b", re.I)
_MATSURI_KANJI = frozenset({"祭り", "祭"})
_MATSURI_SUFFIXES = {"": "", "ちゃん": "-chan", "さん": "-san", "様": "-sama", "さま": "-sama"}
_MATSURI_EVENT_PREFIX = ("お", "御", "夏", "秋", "春", "冬", "ひな", "雛", "血", "雪", "文化", "学園", "収穫")
_MATSURI_DEMONSTRATIVE = ("この", "その", "あの", "どの")
_MATSURI_EVENT_BEFORE = re.compile(
    r"(?:神社|街|町|村|地元|学校|地域|伝統|毎年)(?:の|で行われる)[^。、！？!?\n]{0,5}\Z"
)
_MATSURI_EVENT_AFTER = re.compile(
    r"会場|期間|当日|本番|実行委員|委員会|囃子|ばやし|騒ぎ|ごと|事|"
    r"(?:の)(?:開催|会場|屋台|日程|準備|季節|出店|実行委員|委員会)|"
    r"(?:が|は|を|に|も)?(?:開催|始ま|終わ|延期|中止|盛大|近づ|開か|行われ|ある|あった|あります|あって)|"
    r"に(?:行|いく|行く|参加)"
)
_MATSURI_EVENT_PREDICATE = re.compile(
    r"(?:が|は|も)?(?:楽しみ|たのしみ|楽しい|たのしい)|"
    r"(?:が|は|も)?(?:今日|明日|あした|あす|明後日|来週|来月|週末|[0-9０-９]+[月日])"
    r"(?:だ|です|から|まで|に|[。！？!?]|$)"
)
_MATSURI_EVENT_TOPIC = re.compile(
    r"お祭|おまつり|御祭|夏祭|秋祭|春祭|冬祭|ひなまつり|雛祭|文化祭|学園祭|収穫祭|"
    r"祭典|神社|屋台|縁日|花火大会|開催|フェス"
)
_MATSURI_PERSON = re.compile(
    r"(?:(?:が|は|も)(?:来|きた|い[るた]|どこ|どう|何して|言|話|喋|しゃべ|歌|配信|ログイン|帰|戻|やっ)"
    r"|(?:に|と)(?:会|話|遊|聞|通話|一緒|いっしょ|よろしく)|の(?:配信|声|視点|チャンネル)"
    r"|(?:が|は|も)(?:この|その|あの|どの)?(?:神社|会場|お祭り|夏祭り|秋祭り|まつり|マツリ|祭り|祭)"
    r"(?:に|へ)(?:行|いく|向か|来|くる)"
    r"|と(?:お祭り|夏祭り|秋祭り|神社)(?:に|へ)(?:行|いく)"
    r"|どこ|来[たてるな]|って(?:言|話))"
)
# Reviewed HoloGTA utterances with a missing ASR word boundary. Two named
# neighbors establish an enumeration; a tactical self-correction establishes
# the second cluster. Neither is a general kanji-to-name substitution.
_MATSURI_ROSTER = re.compile(
    r"(?:いろは|イロハ)[、，,\s]*(?P<name>祭り|祭)[、，,\s]*(?:あと|それから|次|と)?"
    r"[、，,\s]*(?:ビジュー|びじゅー)"
)
_MATSURI_CORRECTION = re.compile(
    r"(?P<first>祭り|祭)[、，,\s]*(?:あ[ーあぁ〜～]*|あれ)[、，,\s]*"
    r"(?P<second>祭り|祭)(?:じゃない|ではない)[、，,\s]*(?:まとり|マトリ)"
)
_MATSURI_TACTICAL = re.compile(r"サーマル(?:を)?使(?:って|え)")


def _matsuri_full_name(mention: NameMention) -> bool:
    return mention.target.casefold().startswith("natsuiro matsuri")


def _matsuri_literal(text: str, start: int, end: int, *, honorific: bool = False) -> bool:
    before, after = text[:start], text[end:]
    return (before.endswith(_MATSURI_EVENT_PREFIX)
            or (not honorific and before.endswith(_MATSURI_DEMONSTRATIVE))
            or bool(_MATSURI_EVENT_BEFORE.search(before) or _MATSURI_EVENT_AFTER.match(after)))


def _matsuri_candidates(text: str, aliases: Mapping[str, str]) -> tuple[NameMention, ...]:
    candidates = []
    for source, target in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(re.escape(source), text):
            full = target.casefold().startswith("natsuiro matsuri")
            repeated = text.strip(_PUNCTUATION) in (source * 2, source * 3)
            if not (repeated or (_left_boundary(text, match.start())
                                 and _right_boundary(text, match.end(), source, aliases))):
                continue
            if not full and _matsuri_literal(text, match.start(), match.end(), honorific="-" in target):
                continue
            if not full and "-" not in target and _MATSURI_EVENT_PREDICATE.match(text[match.end():]):
                continue
            if any(match.start() < item.end and match.end() > item.start for item in candidates):
                continue
            candidates.append(NameMention(match.start(), match.end(), source, target, "alias"))
    return tuple(sorted(candidates, key=lambda item: item.start))


def _matsuri_event_nearby(text: str, start: int, end: int, previous_ja: str) -> bool:
    nearby = text[max(0, start - _SOURCE_TOPIC_WINDOW):start] + text[end:end + _SOURCE_TOPIC_WINDOW]
    return bool(_MATSURI_EVENT_TOPIC.search(nearby)
                or _MATSURI_EVENT_TOPIC.search(previous_ja[-_RECENT_TOPIC_WINDOW:]))


def _matsuri_reviewed_clusters(text: str, aliases: Mapping[str, str], video_context: str):
    if not _HOLO.search(video_context):
        return ()
    mentions = []
    for cluster in _MATSURI_ROSTER.finditer(text):
        source = cluster.group("name")
        start, end = cluster.span("name")
        if (aliases.get(source) == "Matsuri"
                and not _matsuri_literal(text, start, end)
                and not _matsuri_event_nearby(text, start, end, "")):
            mentions.append(NameMention(start, end, source, "Matsuri", "roster_names"))
    for cluster in _MATSURI_CORRECTION.finditer(text):
        if not _MATSURI_TACTICAL.search(text[cluster.end():cluster.end() + 32]):
            continue
        if any(_matsuri_literal(text, *cluster.span(group))
               or _matsuri_event_nearby(text, *cluster.span(group), "")
               for group in ("first", "second")):
            continue
        for group in ("first", "second"):
            source = cluster.group(group)
            if aliases.get(source) == "Matsuri":
                start, end = cluster.span(group)
                mentions.append(NameMention(start, end, source, "Matsuri", "spoken_self_correction"))
    return tuple(mentions)


def resolve_matsuri_mentions(
    text: str,
    configured: Mapping[str, str],
    *,
    previous_ja: str = "",
    video_context: str = "",
) -> tuple[NameMention, ...]:
    """Resolve current Matsuri mentions without mapping ordinary festivals.

    Kana needs Hololive/name context or an explicit honorific/full name.
    Bare kanji needs stronger identity and cannot override a nearby event
    topic. Local literal exclusions permit a person and festival together.
    """
    aliases = {source: target for source, target in configured.items()
               if source and is_matsuri_target(target)}
    if not aliases:
        return ()
    # Kanji ASR spellings are admitted only for canonical targets the user's
    # configuration actually contains. Explicit custom source mappings win.
    for spelling in _MATSURI_KANJI:
        for suffix, english_suffix in _MATSURI_SUFFIXES.items():
            source, target = spelling + suffix, "Matsuri" + english_suffix
            if source not in configured and target in aliases.values():
                aliases[source] = target
    current = _matsuri_candidates(text, aliases)
    recent = _matsuri_candidates(previous_ja, aliases)
    identity = bool(_MATSURI_TITLE.search(video_context)
                    or any(_matsuri_full_name(item) for item in current + recent)
                    or any(item.source not in _MATSURI_KANJI and "-" in item.target
                           and not _matsuri_event_nearby(previous_ja, item.start, item.end, "")
                           for item in recent))
    named_context = bool(_HOLO.search(video_context) or identity)
    mentions = []
    for item in current:
        full = _matsuri_full_name(item)
        honorific = "-" in item.target
        if full or honorific:
            evidence = "explicit_name"
        elif item.source in _MATSURI_KANJI:
            if not identity or _matsuri_event_nearby(text, item.start, item.end, previous_ja):
                continue
            if not (text.strip(_PUNCTUATION) == item.source
                    or _MATSURI_PERSON.match(text[item.end:])):
                continue
            evidence = "homophone_identity"
        else:
            if not named_context:
                continue
            if (_matsuri_event_nearby(text, item.start, item.end, previous_ja)
                    and not _MATSURI_PERSON.match(text[item.end:])):
                continue
            evidence = "name_context"
        mentions.append(NameMention(item.start, item.end, item.source, item.target, evidence))
    for mention in _matsuri_reviewed_clusters(text, aliases, video_context):
        if not any(mention.start < item.end and mention.end > item.start for item in mentions):
            mentions.append(mention)
    return tuple(sorted(mentions, key=lambda item: item.start))
