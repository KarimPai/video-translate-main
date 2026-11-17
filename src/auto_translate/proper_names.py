"""Occurrence-level, evidence-limited Japanese proper-name resolution.

Labels describe the evidence available to the pipeline, not calibrated
probabilities. Metadata and chat can never create a mention or establish that
an ambiguous noun is a person. The Japanese source is never rewritten here.
"""
from dataclasses import asdict, dataclass
import re

PERSON = "PERSON"
COMMON_NOUN = "COMMON_NOUN"
UNKNOWN = "UNKNOWN"
_HONORIFICS = {"ちゃん": "-chan", "さん": "-san", "さま": "-sama",
               "様": "-sama", "先輩": "-senpai", "せんぱい": "-senpai",
               "くん": "-kun", "君": "-kun"}

# Reviewed spoken forms only. Short forms keep their own English identity;
# a surname is never inferred from the talent associated with a nickname.
_FORMS = {
    "夏色まつり": ("Natsuiro Matsuri", "Matsuri", "full"),
    "夏色マツリ": ("Natsuiro Matsuri", "Matsuri", "full"),
    "夏色祭り": ("Natsuiro Matsuri", "Matsuri", "full"),
    "なついろまつり": ("Natsuiro Matsuri", "Matsuri", "full"),
    "さくらみこ": ("Sakura Miko", "Miko", "full"),
    "さくらミコ": ("Sakura Miko", "Miko", "full"),
    "さくら巫女": ("Sakura Miko", "Miko", "full"),
    "桜みこ": ("Sakura Miko", "Miko", "full"),
    "桜ミコ": ("Sakura Miko", "Miko", "full"),
    "桜巫女": ("Sakura Miko", "Miko", "full"),
    "まつり": ("Matsuri", "Matsuri", "ambiguous"),
    "マツリ": ("Matsuri", "Matsuri", "ambiguous"),
    "祭り": ("Matsuri", "Matsuri", "ambiguous"),
    "祭": ("Matsuri", "Matsuri", "ambiguous"),
    "みこ": ("Miko", "Miko", "ambiguous"),
    "ミコ": ("Miko", "Miko", "ambiguous"),
    "巫女": ("Miko", "Miko", "ambiguous"),
    "みこち": ("Mikochi", "Miko", "nickname"),
    "ミコチ": ("Mikochi", "Miko", "nickname"),
    "ラプ": ("Lap", "Lap", "nickname"),
    "らぷ": ("Lap", "Lap", "nickname"),
    "ラプラス": ("Laplus", "Lap", "nickname"),
    "らぷらす": ("Laplus", "Lap", "nickname"),
    "わため": ("Watame", "Watame", "nickname"),
    "ワタメ": ("Watame", "Watame", "nickname"),
    "わためぇ": ("Watamee", "Watame", "nickname"),
    # Short forms independently used in the official talent profile text:
    # hololive.hololivepro.com/en/talents/{moona-hoshinova,kaela-kovalskia,mococo-abyssgard}/
    "ムーナ": ("Moona", "Moona", "nickname"),
    "カエラ": ("Kaela", "Kaela", "nickname"),
    "モココ": ("Mococo", "Mococo", "nickname"),
}
_SURFACES = {spelling + suffix for spelling in _FORMS for suffix in ("", *_HONORIFICS)}
# Sort complete surfaces: みこちゃん must beat the shorter みこち prefix.
_PATTERN = re.compile("|".join(map(re.escape, sorted(_SURFACES, key=len, reverse=True))))
_PERSON_SYNTAX = re.compile(
    r"(?:が|は|も)?(?:来[たてるな]|きた|いる|いた|どこ|喋|しゃべ|言っ|歌っ|"
    r"配信|ログイン|話[すした]|遊[ぶん]|やっ|やる|待っ|見[たて])|"
    r"と(?:話|遊|通話|一緒|いっしょ)|の(?:配信|声|視点|チャンネル)|"
    r"に(?:会|聞|話|言|渡|あげ)|"
    r"(?:が|は|も)[^。！？!?、]{0,24}(?:来た|来る|いる|いた|行く|遊[ぶん]|話[すし]|配信)")
_EVENT_PREFIX = re.compile(r"(?:お|御|夏|秋|春|冬|文化|学園|ひな|雛|血|雪|収穫|体育|前夜)$")
_EVENT_PREDICATE = re.compile(
    r"(?:に|へ)(?:行[くっかき]|い[くっかき]|来[たるて]|きた|向か|参加|出かけ|到着)|"
    r"(?:を|が|は)(?:開催|開[くかき]|中止|延期)|(?:の)(?:屋台|開催|会場|日程|準備)")
_SHRINE = re.compile(r"神社|神主|神職|お祓い|御祓|神楽|境内|神殿|装束|職業")
_END = re.compile(r"^(?:$|[\s。、！？!?「」『』：:、]|(?:が|は|を|に|へ|も|の|と|で)|"
                  r"から|まで|より|だけ|なら|来[たてるな]|いた|いる|どこ|すご|かわいい|って|だ|です|おはよう)")


@dataclass(frozen=True)
class NameMention:
    start: int
    end: int
    source: str
    target: str | None
    classification: str
    evidence: tuple[str, ...]
    family: str

    def as_dict(self) -> dict:
        return asdict(self)


def _parts(value):
    for spelling in sorted(_FORMS, key=len, reverse=True):
        if value.startswith(spelling) and value[len(spelling):] in ("", *_HONORIFICS):
            return spelling, value[len(spelling):], _FORMS[spelling]
    raise ValueError("unreviewed name form")


def _boundary(text, match, spelling, family):
    before, after = text[:match.start()], text[match.end():]
    # Literal festival compounds are retained for diagnostics and excluded
    # from the glossary individually, rather than hiding a whole subtitle.
    if family == "Matsuri" and _EVENT_PREFIX.search(before):
        return True
    left = (not before or not before[-1].isalnum() or before[-1] in "がはをとにでのも"
            or before.endswith(spelling))
    if not left:
        return False
    if spelling in {"みこ", "ミコ"} and after.startswith(("と", "ト")):
        # みこと is also a different name/word. Accept an actual coordination
        # only when another reviewed name follows, or a clear human action.
        return any(after.startswith(form, 1) for form in _FORMS) or bool(
            re.match(r"と(?:遊|話|通話|一緒|いっしょ)", after))
    return bool(_END.match(after) or after.startswith(spelling))


def _explicit_context(previous, family):
    for match in _PATTERN.finditer(previous[-600:]):
        spelling, honorific, (_, group, kind) = _parts(match[0])
        if group != family or not _boundary(previous[-600:], match, spelling, group):
            continue
        if kind in {"full", "nickname"}:
            return True
        if honorific and spelling not in {"祭", "巫女"} and not _EVENT_PREFIX.search(previous[-600:][:match.start()]):
            return True
    return False


def _participant(family, participants):
    aliases = {"Matsuri": r"夏色まつり|Natsuiro Matsuri|^Matsuri$|^まつり$",
               "Miko": r"さくらみこ|Sakura Miko|^Miko$|^みこ$"}
    return any(re.search(aliases.get(family, r"(?!)"), str(value), re.I)
               for value in participants)


def resolve_names(japanese: str, *, previous_ja: str = "", video_title: str = "",
                  confirmed_participants=()) -> list[NameMention]:
    """Resolve only reviewed forms present in the current Japanese source.

    Only spoken syntax and confirmed Japanese dialogue can resolve ambiguous
    forms. `video_title` is diagnostic context and never person evidence.
    """
    if not isinstance(japanese, str) or not japanese:
        return []
    previous_ja = previous_ja if isinstance(previous_ja, str) else ""
    if isinstance(confirmed_participants, str):
        confirmed_participants = (confirmed_participants,)
    elif not isinstance(confirmed_participants, (tuple, list, set, frozenset)):
        confirmed_participants = ()
    matches = []
    for match in _PATTERN.finditer(japanese):
        spelling, suffix, (target, family, kind) = _parts(match[0])
        if _boundary(japanese, match, spelling, family):
            matches.append((match, spelling, suffix, target, family, kind))
    explicit_families = {
        family for _, spelling, suffix, _, family, kind in matches
        if kind in {"full", "nickname"} or (suffix and spelling not in {"祭", "巫女"})
    }
    result = []
    for match, spelling, suffix, target, family, kind in matches:
        before, after = japanese[:match.start()], japanese[match.end():]
        # Clause-local literal evidence must not suppress a name elsewhere.
        clause = re.split(r"[。！？!?、]", before)[-1] + match[0] + re.split(r"[。！？!?、]", after)[0]
        evidence = []
        person_context = False
        if _explicit_context(previous_ja, family):
            evidence.append("recent_confirmed_name")
            person_context = True
        if _participant(family, confirmed_participants):
            evidence.append("verified_participant")
            person_context = True
        if family in explicit_families:
            evidence.append("local_explicit_name")
            person_context = True
        shrine_role = bool(re.search(r"(?:神社|神殿|境内|神楽)[^。！？!?、]{0,8}の$", before))
        literal_shrine = (_SHRINE.search(clause) and
                          (shrine_role or not person_context or (spelling == "巫女" and suffix == "さん")))
        if family == "Matsuri" and kind != "full" and _EVENT_PREFIX.search(before):
            label, evidence = COMMON_NOUN, ["literal_compound"]
        elif family == "Matsuri" and kind != "full" and not suffix and _EVENT_PREDICATE.match(after):
            label, evidence = COMMON_NOUN, ["literal_predicate"]
        elif family == "Miko" and kind == "ambiguous" and literal_shrine and suffix != "ちゃん":
            label, evidence = COMMON_NOUN, ["literal_shrine_context"]
        elif kind == "full":
            label, evidence = PERSON, ["explicit_full_name"]
        elif kind == "nickname":
            label, evidence = PERSON, ["spoken_nickname"]
        elif suffix and spelling not in {"巫女", "祭"}:
            label, evidence = PERSON, ["spoken_honorific"]
        else:
            syntax = bool(_PERSON_SYNTAX.match(after))
            isolated = japanese.strip(" 。、！？!?「」") == match[0]
            if syntax:
                evidence.append("person_syntax")
            if person_context and (syntax or isolated):
                label = PERSON
            elif spelling in {"巫女", "祭", "祭り"} and not person_context and not syntax:
                label = COMMON_NOUN
                evidence.append("literal_kanji_without_person_evidence")
            else:
                label = UNKNOWN
                evidence.append("insufficient_evidence")
        result.append(NameMention(match.start(), match.end(), match[0],
                                  target + _HONORIFICS.get(suffix, "") if label == PERSON else None,
                                  label, tuple(evidence), family))
    return result


def name_glossary(mentions) -> list[tuple[str, str]]:
    """Glossary pairs safe for PLaMo's global source-string glossary.

    One spelling can denote both a person and a literal noun in one subtitle.
    A global glossary cannot express that distinction, so omit that spelling
    and leave occurrence-level guidance/retry to the translation stage.
    """
    rows = [item.as_dict() if isinstance(item, NameMention) else item for item in mentions]
    blocked = {row["source"] for row in rows if row.get("classification") != PERSON}
    result = []
    for row in rows:
        pair = (row["source"], row.get("target"))
        if (not any(pair[0] in spelling for spelling in blocked) and pair[1]
                and row.get("classification") == PERSON and pair not in result):
            result.append(pair)
    return result
