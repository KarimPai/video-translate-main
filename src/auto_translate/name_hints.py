"""Short, context-gated Hololive name hints and safe English spelling repairs.

These only affect known names in Hololive streams. Do not use them for
general Japanese or reconstruct words absent from the audio.
"""
import re

_HOLO = re.compile(r"ホロ|holo", re.I)
_HOLOGTA = re.compile(r"ホロ(?:ライブ)?GTA|ホロスサントス|holo.?gta", re.I)
_HAACHAMA_JA = ("はあちゃま", "はぁちゃま", "はーちゃま", "はやちゃま", "ハーチャマ")
_LAPLUS_JA = ("ラプラス", "らぷらす")


_MATSURI = re.compile(r"(夏色)?(まつり|マツリ|祭り)(ちゃん|さん|様|さま)?")
_MATSURI_EVENT = re.compile(r"お祭|おまつり|夏祭|秋祭|祭典|文化祭|学園祭|神社|屋台|縁日|花火大会|開催|フェス")
_MATSURI_PERSON = re.compile(r"(?:が|は|も)?(?:来た|来ない|いる|いた|喋|しゃべ|言っ|歌っ|配信して|ログイン)|と(?:話|遊|通話)|の(?:配信|声|視点|チャンネル)")
_HONORIFIC = {"ちゃん": "-chan", "さん": "-san", "様": "-sama", "さま": "-sama"}


def matsuri_matches(ja, title, previous):
    # Retain the public helper while centralizing the actual evidence rules.
    from .proper_names import name_glossary, resolve_names
    mentions = resolve_names(ja, video_title=title, previous_ja=previous)
    return name_glossary([mention for mention in mentions if mention.family == "Matsuri"])


# Only substitute an existing English mention. Family membership is also
# used to avoid collapsing two different spoken forms in the same cue.
_FAMILIES = {
    "Lap": ("Lap Laplus Lap-sama Lap-chan Lap-san", r"(?:La\s*\+|La plus|Laplus|Laplace)(?:[ ·-]+Darkness+)?|Lap"),
    "Matsuri": ("Matsuri Matsuri-chan Matsuri-san Matsuri-sama", r"(?:Natsuiro )?Matsuri"),
    "Miko": ("Miko Mikochi Miko-chan Miko-san Miko-sama", r"(?:Sakura )?Miko(?:chi)?"),
    "Watame": ("Watame Watamee", r"(?:Tsunomaki )?Watamee?"),
    "Marine": ("Senchou", r"(?:Houshou )?Marine|Senchou"),
    "Suisei": ("Sui-chan", r"(?:Hoshimachi )?Suisei|Sui"),
    "Pekora": ("Peko-chan Pekoora", r"(?:Usada )?Pekora|Pekoora|Peko"),
    "Fubuki": ("Fubu-chan", r"(?:Shirakami )?Fubuki|Fubu"),
    "Korone": ("Koro-san", r"(?:Inugami )?Korone|Koro"),
    "Okayu": ("Okayun", r"(?:Nekomata )?Okayun?"),
    "Botan": ("Shishiron", r"(?:Shishiro )?Botan|Shishiron"),
    "Towa": ("Towa-sama", r"(?:Tokoyami )?Towa"),
    "Kanata": ("Kanatan", r"(?:Amane )?Kanatan?"),
    "Nene": ("Nenechi", r"(?:Momosuzu )?Nene(?:chi)?"),
    "Polka": ("Polpol", r"(?:Omaru )?Polka|Polpol"),
    "Koyori": ("Koyo-chan", r"(?:Hakui )?Koyori|Koyo"),
    "Lui": ("Lui-nee", r"(?:Takane )?Lui"),
    "Haachama": ("Haachama", r"Akai Haato|Haachama"),
    "Aqua": ("Akutan", r"(?:Minato )?Aqua|Akutan"),
    "Fuwawa": ("Fuwawa", r"Fuwawa(?: Abyssgard)?"),
    "Iofi": ("Iofi", r"(?:Airani )?Iofifteen|Iofi"),
    "Moona": ("Moona", r"Moona(?: Hoshinova)?"),
    "Kaela": ("Kaela", r"Kaela(?: Kovalskia)?"),
    "Mococo": ("Mococo", r"Mococo(?: Abyssgard)?"),
}


SPOKEN_NAME_FORMS = {form for forms, _ in _FAMILIES.values() for form in forms.split()}


def preserve_spoken_names(ja: str, english: str, names: list[tuple[str, str]], *, mentions=None) -> str:
    """Change one unambiguous output span per evidenced name family."""
    targets = {target for _, target in names}
    for family, (forms, aliases) in _FAMILIES.items():
        spoken = targets.intersection(forms.split())
        if len(spoken) != 1 or any(
            family.lower() in other.lower() for other in targets - spoken
        ):
            continue
        target = spoken.pop()
        source = next(source for source, en in names if en == target)
        if family == "Matsuri":
            # Avoid replacing a real event elsewhere in a mixed name/event cue.
            rest = ja.replace(source, "", 1)
            if not (_MATSURI.search(rest) or _MATSURI_EVENT.search(rest) or "祭" in rest):
                # Replacing a plural noun can require rewriting agreement
                # or possession. Leave those sentences to the selective
                # model retry rather than producing "Matsuri aren't ...".
                plural_grammar = re.search(
                    r"\b(?:are|aren't|were|weren't|have|haven't|do|don't)\b|\bfestivals['’]",
                    english.replace("’", "'"), re.I,
                )
                aliases += r"|(?:the |a )?festival" + ("" if plural_grammar else "s?")
        if family == "Lap":
            # A common English word needs an identifiable name slot. Avoid
            # touching any rap vocabulary in mixed music/name dialogue.
            if not re.search(r"ラップ|らっぷ|歌|音楽|曲|韻|包", ja):
                aliases += r"|rap(?=(?:[ -](?:sama|chan|san))?\s+(?:is here|came|arrived)\b)"
                if ja.strip(" 。！？!?") == source or "どこ" in ja:
                    aliases += r"|rap(?=(?:[ -](?:sama|san|chan))?[.!?, '’]*$)"
                if ja.strip(" 。！？!?") == source and re.fullmatch(r"rapping[.!?]?", english, re.I):
                    aliases += r"|rapping"
        if family == "Miko":
            # A confirmed person may itself be written 巫女 by ASR. Examine
            # other occurrences rather than rejecting every kanji source or
            # every sentence that also mentions a shrine as a location.
            rest = ja.replace(source, "", 1)
            rows = [(item.as_dict() if hasattr(item, "as_dict") else item)
                    for item in (mentions or [])]
            ambiguous_other = any(row.get("family") == "Miko"
                                  and row.get("classification") != "PERSON" for row in rows)
            if not ambiguous_other and not re.search(r"巫女|みこ|ミコ", rest):
                aliases += r"|(?:the |a )?shrine maiden"
        if family == "Marine" and "マリン" not in ja:
            aliases += r"|(?:the )?captain"
        pattern = re.compile(r"(?<![\w-])(?:" + aliases + r")(?:[ -](?:chan|san|sama|nee))?(?![\w-])", re.I)
        hits = list(pattern.finditer(english))
        if len(hits) == 1:
            hit = hits[0]
            english = english[:hit.start()] + target + english[hit.end():]
    return english


def canonicalize_english_names(japanese: str, english: str, video_title: str = "") -> str:
    """Correct only strongly evidenced, known English name misspellings."""
    if not _HOLO.search(video_title) or not japanese or not english:
        return english
    if any(name in japanese for name in _HAACHAMA_JA):
        english = re.sub(
            r"\b(?:Ha+[- ]?chama|Hasa[- ]?chama|Haya[- ]?chama)\b",
            "Haachama", english, flags=re.I,
        )
    if any(name in japanese for name in _LAPLUS_JA):
        # Prefer the short spoken name in English; do not expand it to a
        # full stage name, and do not introduce names absent from the JP.
        english = re.sub(
            r"(?<![A-Za-z])(?:La\s*\+|La\s+plus|Laplus|Laplace)"
            r"(?:[\s·・-]+Darkness+)?(?![A-Za-z])",
            "Laplus", english, flags=re.I,
        )
    return english
