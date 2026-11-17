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
    "獅白ぼたん": ("Shishiro Botan", "Botan", "full"),
    "獅白ボタン": ("Shishiro Botan", "Botan", "full"),
    "ししろぼたん": ("Shishiro Botan", "Botan", "full"),
    "ぼたん": ("Botan", "Botan", "ambiguous"),
    "ボタン": ("Botan", "Botan", "ambiguous"),
    "ししろん": ("Shishiron", "Botan", "nickname"),
    "シシロン": ("Shishiron", "Botan", "nickname"),
    "みこち": ("Mikochi", "Miko", "nickname"),
    "ミコチ": ("Mikochi", "Miko", "nickname"),
    "ラプ": ("Lap", "Lap", "nickname"),
    "らぷ": ("Lap", "Lap", "nickname"),
    "ラプラス": ("Laplus", "Lap", "nickname"),
    "らぷらす": ("Laplus", "Lap", "nickname"),
    "わため": ("Watame", "Watame", "nickname"),
    "ワタメ": ("Watame", "Watame", "nickname"),
    "わためぇ": ("Watamee", "Watame", "nickname"),
    # Existing reviewed glossary forms also need occurrence-level guidance;
    # a short spoken form never licenses expansion to a full stage name.
    "すいちゃん": ("Sui-chan", "Suisei", "nickname"),
    "ぺこら": ("Pekora", "Pekora", "nickname"),
    "ペコラ": ("Pekora", "Pekora", "nickname"),
    "ぺこーら": ("Pekoora", "Pekora", "nickname"),
    "ぺこちゃん": ("Peko-chan", "Pekora", "nickname"),
    "フブちゃん": ("Fubu-chan", "Fubuki", "nickname"),
    "ころさん": ("Koro-san", "Korone", "nickname"),
    "おかゆん": ("Okayun", "Okayu", "nickname"),
    "トワ様": ("Towa-sama", "Towa", "nickname"),
    "かなたん": ("Kanatan", "Kanata", "nickname"),
    "ねねち": ("Nenechi", "Nene", "nickname"),
    "ポルポル": ("Polpol", "Polka", "nickname"),
    "こよちゃん": ("Koyo-chan", "Koyori", "nickname"),
    "ルイ姉": ("Lui-nee", "Lui", "nickname"),
    "あくたん": ("Akutan", "Aqua", "nickname"),
    "フワワ": ("Fuwawa", "Fuwawa", "nickname"),
    "イオフィ": ("Iofi", "Iofi", "nickname"),
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
    r"(?:が|は|も)[^。！？!?、]{0,24}(?:来た|来る|いる|いた|行く|遊[ぶん]|話[すし]|配信|を押[すし])")
_EVENT_PREFIX = re.compile(r"(?:お|御|夏|秋|春|冬|文化|学園|ひな|雛|血|雪|収穫|体育|前夜)$")
_EVENT_PREDICATE = re.compile(
    r"(?:に|へ)(?:行[くっかき]|い[くっかき]|来[たるて]|きた|向か|参加|出かけ|到着)|"
    r"(?:を|が|は)(?:開催|開[くかき]|中止|延期)|(?:の)(?:屋台|開催|会場|日程|準備)")
_BUTTON_PREFIX = re.compile(r"(?:電源|停止|再生|終了|決定|確認|操作|録画|リセット|スタート|キャンセル)$")
_BUTTON_PREDICATE = re.compile(
    r"(?:を|が|は|も|に)?(?:押[すしせさそ]|お[すしせさそ]|押下|クリック|連打|長押し)|"
    r"(?:が|は|も)(?:光|点滅|壊れ|取れ|外れ)|の(?:位置|色|設定|配置)")
_SHRINE = re.compile(r"神社|神主|神職|お祓い|御祓|神楽|境内|神殿|装束|職業")
_END = re.compile(r"^(?:$|[\s。、！？!?「」『』：:,，]|(?:が|は|を|に|へ|も|の|と|で)|"
                  r"から|まで|より|だけ|なら|来[たてるな]|いた|いる|どこ|すご|かわいい|って|だ|です|おはよう)")
_GROUP_SUFFIX = re.compile(r"^(?:たち|達|ら)(?=" + _END.pattern[1:] + r")")
_VOCATIVE = re.compile(
    r"^[、,，]\s*(?:(?:こっち|こちら|ここ|そっち|そこ|あっち)(?:に|へ)?\s*)?"
    r"(?:来て|きて|おいで|待って|待て|見て|聞いて|聞け|止まって|止まれ)"
    r"(?=$|[\s。！？!?、,，]|ね|よ|くれ|ください)")


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
    before, after = text[:match.start()], _GROUP_SUFFIX.sub("", text[match.end():], count=1)
    # Literal festival compounds are retained for diagnostics and excluded
    # from the glossary individually, rather than hiding a whole subtitle.
    if family == "Matsuri" and _EVENT_PREFIX.search(before):
        return True
    if family == "Botan" and _BUTTON_PREFIX.search(before):
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


def _explicit_context(previous):
    if not previous:
        return set()
    return {mention.family for mention in resolve_names(previous[-600:])
            if mention.classification == PERSON}


def _literal_evidence(japanese, match, spelling, suffix, family, kind, person_context):
    before = japanese[:match.start()]
    after = _GROUP_SUFFIX.sub("", japanese[match.end():], count=1)
    if family == "Matsuri" and kind != "full":
        if _EVENT_PREFIX.search(before):
            return "literal_compound"
        if not suffix and _EVENT_PREDICATE.match(after):
            return "literal_predicate"
    if (family == "Botan" and kind == "ambiguous" and not suffix
            and (_BUTTON_PREFIX.search(before) or _BUTTON_PREDICATE.match(after))):
        return "literal_button_context"
    if family != "Miko" or kind != "ambiguous" or suffix == "ちゃん":
        return None
    clause = re.split(r"[。！？!?、]", before)[-1] + match[0] + re.split(r"[。！？!?、]", after)[0]
    shrine_role = bool(re.search(r"(?:神社|神殿|境内|神楽)[^。！？!?、]{0,8}の$", before))
    if (_SHRINE.search(clause)
            and (shrine_role or not person_context or (spelling == "巫女" and suffix == "さん"))):
        return "literal_shrine_context"
    return None


def _participant(family, participants):
    aliases = {"Matsuri": r"夏色まつり|Natsuiro Matsuri|^Matsuri$|^まつり$",
               "Miko": r"さくらみこ|Sakura Miko|^Miko$|^みこ$",
               "Botan": r"獅白ぼたん|獅白ボタン|Shishiro Botan|^Botan$|^ぼたん$|^ボタン$|^Shishiron$"}
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
    if not matches:
        return []
    previous_families = _explicit_context(previous_ja)
    explicit_families = {
        family for match, spelling, suffix, _, family, kind in matches
        if (kind in {"full", "nickname"} or (suffix and spelling not in {"祭", "巫女"}))
        and not _literal_evidence(japanese, match, spelling, suffix, family, kind, True)
    }
    result = []
    for match, spelling, suffix, target, family, kind in matches:
        after = _GROUP_SUFFIX.sub("", japanese[match.end():], count=1)
        evidence = []
        person_context = False
        if family in previous_families:
            evidence.append("recent_confirmed_name")
            person_context = True
        if _participant(family, confirmed_participants):
            evidence.append("verified_participant")
            person_context = True
        if family in explicit_families:
            evidence.append("local_explicit_name")
            person_context = True
        literal = _literal_evidence(japanese, match, spelling, suffix, family, kind, person_context)
        if literal:
            label, evidence = COMMON_NOUN, [literal]
        elif kind == "full":
            label, evidence = PERSON, ["explicit_full_name"]
        elif kind == "nickname":
            label, evidence = PERSON, ["spoken_nickname"]
        elif suffix and spelling not in {"巫女", "祭"}:
            label, evidence = PERSON, ["spoken_honorific"]
        else:
            syntax = bool(_PERSON_SYNTAX.match(after) or _VOCATIVE.match(after))
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


def reviewed_spoken_name_forms() -> frozenset[str]:
    """Reviewed Latin spellings, without asserting that a token is a person.

    ASR may retain a narrow candidate for further acoustic/context checks.
    Membership alone never proves Japanese speech or a name mention.
    """
    forms = set()
    for target, _, _ in _FORMS.values():
        forms.add(target)
        if not re.search(r"-(?:chan|san|sama|senpai|kun|nee)$", target):
            forms.update(target + suffix for suffix in _HONORIFICS.values())
    return frozenset(forms)


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
