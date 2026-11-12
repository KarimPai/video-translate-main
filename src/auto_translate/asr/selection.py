"""Conservative Japanese comparison and exact two-candidate selection."""
import re
import unicodedata
from difflib import SequenceMatcher


def normalize(text):
    value = unicodedata.normalize("NFC", text)
    # Fold width only. Preserve kanji, particles, prolonged sounds and decimal points.
    value = re.sub(r"[！-～｡-ﾟ]+", lambda m: unicodedata.normalize("NFKC", m[0]), value)
    value = "".join(chr(ord(c) - 96) if "ァ" <= c <= "ヶ" else c for c in value)
    return re.sub(r"(?<![0-9])[.,]|[.,](?![0-9])|[\s、。！？!?「」『』…]", "", value)


def compare(a, b, terms):
    a, b = normalize(a), normalize(b)
    similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
    flags = []
    for label, pattern in (("negation", r"ない|なく|なかっ|ません|せん|ぬ|ず"),
                           ("number", r"[0-9一二三四五六七八九十百千万億〇零]+")):
        if re.findall(pattern, a) != re.findall(pattern, b):
            flags.append(label)
    if any((normalize(t) in a) != (normalize(t) in b) for t, _ in terms):
        flags.append("glossary")
    if a != b and (a in b or b in a or abs(len(a)-len(b)) >= 4):
        flags.append("missing phrase")
    return similarity, flags


def selected(content, haas, secondary):
    if content not in ("A", "B", "U"):
        raise ValueError("selector must return exactly A, B or U")
    label = {"A": "haas", "B": "nemotron", "U": "uncertain"}[content]
    return {"selection": label, "selected_japanese": secondary if content == "B" else haas,
            "uncertain": content == "U", "reason": "uncertain; using Haas" if content == "U"
            else "selected original hypothesis"}
