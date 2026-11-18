"""Conservative shared artifact recognition; never guess missing spoken words."""
import re
import unicodedata
from dataclasses import dataclass

_BAD_TAGS = r"smtbarking|HClackank|giggy|SYak|SGigh|CBDurring"
_EVENTS = (r"laughter|laughs|laughing|giggles|giggling|scream|screams|screaming|"
           r"sobbing|crying|gasping|sighing|shouting|笑い声|笑い|爆笑|悲鳴|泣き声|叫び声")
_TAG = re.compile(rf"\*(?:{_BAD_TAGS}|{_EVENTS})\*|"
                  rf"\[(?:{_EVENTS})\]|\((?:{_EVENTS})\)|【(?:{_EVENTS})】", re.I)
_LAUGH_ONLY = re.compile(r"(?i)^(?:(?:ha){3,}|(?:he){3,}|あ?ははは[はぁ]*|ふふふ[ふっ]*|[wｗ]{3,})[\s.!?。！？…]*$")


@dataclass(frozen=True)
class CleanText:
    text: str
    removed: tuple[str, ...] = ()


def strip_artifacts(text: str, *, english: bool = False) -> CleanText:
    removed = []
    def drop(match):
        removed.append(match[0])
        # A removed tag between English clauses can mark a sentence break.
        left, right = text[:match.start()].rstrip(), text[match.end():].lstrip()
        if english and left and right and left[-1].isalnum() and right[0].isupper():
            return ". "
        return " "
    value = _TAG.sub(drop, text)
    # This observed corruption repeats the *same* Thai letter around Japanese.
    # Ordinary Thai names/words and other scripts are deliberately untouched.
    if value.count("ฟ") >= 3 and ("\ufffd" in value or value.count("ฟ") >= 6):
        value = re.sub(r"(?<![\u0e00-\u0e7f])ฟ+(?![\u0e00-\u0e7f])",
                       lambda m: removed.append(m[0]) or " ", value)
    # A replacement character inside a word has lost semantic information.
    # Suppress that candidate instead of joining pieces into an invented word.
    if re.search(r"[^\W\d_]\ufffd+[^\W\d_]", value):
        return CleanText("", tuple(removed + ["unrecoverable_replacement_in_word"]))
    clean = []
    for char in value:
        if char == "\ufffd" or unicodedata.category(char) in ("Cs", "Cc", "Cf"):
            if char in "\n\r\t":
                clean.append(" ")
            else:
                removed.append(f"U+{ord(char):04X}")
                clean.append(" ")
        else:
            clean.append(char)
    value = " ".join("".join(clean).split()).strip()
    if _LAUGH_ONLY.fullmatch(value):
        removed.append(value)
        value = ""
    if not any(char.isalnum() for char in value):
        value = ""
    return CleanText(value, tuple(removed))


def clean_asr_text(text: str) -> CleanText:
    """Keep genuine dialogue, including code-switching and uncertain profanity."""
    return strip_artifacts(text)
