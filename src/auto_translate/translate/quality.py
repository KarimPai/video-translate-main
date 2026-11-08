"""Conservative offline Japanese-to-English subtitle filters and repairs."""
import re
import unicodedata

SKIP_TOKEN = "[SKIP]"

_KANJI = re.compile(r"[\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_NON_LETTERS = re.compile(r"[\s、。，．・!?！？…〜～.,:;:「」『』（）()\[\]【】ー]+")
_KANA_LOOP = re.compile(r"([ぁ-ん])\1{3,}")
_KANA_PHRASE_LOOP = re.compile(r"([ぁ-ん]{1,3})\1{2,}")

# Narrow corrections for clear Japanese source phrases. No arbitrary rewriting.
_LEVEL_REJECTION = re.compile(
    r"\s*(?:レベル)?([0-9０-９]{1,3})は(?:のん[、\s]?){2,}(?:よ|ね)?[。!?！?、\s]*"
)
_THIRTY_MAN_LINE = re.compile(r"\s*30万(?:だろ|だよ|です|だ|でしょ|かな)?[。!?！？\s]*")
_THIRTY_THOUSAND_EN = re.compile(
    r"\bthirty[ -]+thousand\b|(?<![0-9])30,?000(?![0-9])", re.I,
)
_LOST_SIGHT_LINE = re.compile(r"\s*見失(?:いました|った|ってしまった)[。!?！？\s]*")
_WRONG_LOST_SIGHT_EN = re.compile(
    r"\b(?:lost my way|lost our way|got lost|am lost|we're lost)\b", re.I,
)
_ONE_MINUTE_EXTENSION = re.compile(r"(?:1|１|一)分に延ばせ")
_KEEP_EXTRA_MINUTE_EN = re.compile(
    r"\b(?:keep (?:it|this|that) going|extend (?:it|this|that))"
    r"\s+for (?:one more|another|an additional) minute\b", re.I,
)


def is_skip_token(text: str) -> bool:
    """Accept only the complete skip sentinel, never a substring."""
    return text.strip().upper() == SKIP_TOKEN


def is_obvious_nonverbal_japanese(text: str) -> bool:
    """Remove obvious syllable loops, not speech, names or real repetitions."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c
        for c in normalized
    )
    compact = _NON_LETTERS.sub("", normalized)
    if (
        len(compact) < 5
        or not _KANA.search(compact)
        or _KANJI.search(compact)
        or any(c.isascii() and c.isalnum() for c in compact)
    ):
        return False
    single_loop = _KANA_LOOP.search(compact)
    if single_loop:
        return single_loop.group(1) not in {"こ", "そ"}
    matched = _KANA_PHRASE_LOOP.search(compact)
    if matched is None or len(matched.group()) < 0.65 * len(compact):
        return False
    meaningful = {
        "ここ", "そこ", "そう", "はい", "うん", "いい", "いや",
        "いけ", "こい", "まて", "やめ", "だめ", "すご", "よし",
    }
    return len(matched.group(1)) <= 2 and matched.group(1) not in meaningful


def repair_high_confidence_meaning(japanese: str, english: str) -> str:
    """Apply only unambiguous meaning fixes already tested on real streams."""
    source = unicodedata.normalize("NFKC", japanese)

    if _THIRTY_MAN_LINE.fullmatch(source):
        english = _THIRTY_THOUSAND_EN.sub("300,000", english)
    if (_LOST_SIGHT_LINE.fullmatch(source)
            and _WRONG_LOST_SIGHT_EN.search(english)):
        english = "I lost sight of them."
    if _ONE_MINUTE_EXTENSION.search(source):
        english = _KEEP_EXTRA_MINUTE_EN.sub(
            "extend it to one minute", english
        )
    if re.search(r"読み込み.{0,8}軽くな(?:っ|った|る|い)", source):
        english = re.sub(
            r"\b(?:the\s+)?loading\s+has\s+slowed\s+down\b",
            "The load is lighter now", english, count=1, flags=re.I,
        )
    if (re.fullmatch(
        r"もつ煮(?:を)?(?:食う|食べる)しか(?:ねえ|ねぇ|ない)[。!！\s]*",
        source,
    ) and re.search(r"\b(?:can't|cannot)\s+eat\b", english, re.I)):
        return "Guess I'll have to eat motsuni stew."
    if match := _LEVEL_REJECTION.fullmatch(source):
        return f"Level {int(match.group(1))}? No, no, no."
    return english
