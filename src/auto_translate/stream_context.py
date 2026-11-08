"""Conservative ARK terminology, name and caption fixes for live Japanese speech."""
import re
from difflib import SequenceMatcher

ARK_GLOSSARY = {
    "ギガノトサウルス": "Giganotosaurus", "ギガノト": "Giganotosaurus",
    "テイム": "tame", "アルゲンタヴィス": "Argentavis",
    "アルゲン": "Argentavis", "クライオポッド": "Cryopod",
    "ティラノサウルス": "Tyrannosaurus", "ティラノ": "Tyrannosaurus",
    "サドル": "saddle",
}
_ARK_TITLE = re.compile(r"(?i)(?<![a-z])ark(?![a-z])|アークサバイバル|アーク：|アーク:|恐竜サバイバル")
_ARK_SIGNALS = ("ギガノト", "ギガのと", "ギガノと", "テイム", "テーム",
                "恐竜", "アルゲンタヴィス", "アルゲン", "ティラノ", "クライオポッド", "サドル",
                # Weak signals only: paired with genuine prior ARK evidence.
                "気がの都", "ギガナと", "ギガノド")
_PEKO_TITLE = re.compile(r"ぺこら|ペコラ|兎田|pekora", re.I)
_PEKO_FORMS = ("ぺこら", "ペコラ", "ぺこちゃん", "ペコちゃん", "兎田ぺこら")
_PEKO_CAPTIONS = ("ぺこちゃん", "ペコちゃん", "ぺこらちゃん", "ペコラちゃん")
_NEGATION = re.compile(r"ない|無い|ません|じゃ|では|でき|出来|否定|違う|だめ|駄目|良|悪")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_NUMBERS = re.compile(r"[0-9０-９]")
_UNFINISHED = re.compile(r"(?:けど|けれど|けれども|だけど|ですが|だが|から|ので|のに|"
                         r"っていうか|っていう|という|とは|ながら|たら|れば|のは|し|とか)[、,\s]*$")


def is_ark_context(title: str = "", current: str = "", recent: tuple[str, ...] | list[str] = ()) -> bool:
    """Accept an ARK title or at least two on-topic terms in recent Japanese."""
    if _ARK_TITLE.search(title):
        return True
    window = " ".join([*list(recent)[-8:], current])
    found = {term for term in _ARK_SIGNALS if term in window}
    strong = {"ギガノト", "ギガのと", "ギガノと", "テイム", "テーム",
              "アルゲンタヴィス", "アルゲン", "クライオポッド"}
    return len(found) >= 2 and bool(found & strong)


def ark_glossary_hits(text: str) -> list[tuple[str, str]]:
    remaining, hits = text, []
    for source, target in sorted(ARK_GLOSSARY.items(), key=lambda item: -len(item[0])):
        if source in remaining:
            hits.append((source, target))
            remaining = remaining.replace(source, "\0")
    return hits


def repair_ark_spelling(text: str, title: str = "", recent: tuple[str, ...] | list[str] = ()) -> str:
    # A repeated word from an adjacent clear sentence is strong evidence.
    # In this stream 更地 (cleared ground) was heard as おさらち in the
    # next two questions. Do not guess this term without recent confirmation.
    if any("更地" in phrase for phrase in list(recent)[-5:]):
        text = re.sub(r"おさらち(?=って|どういう|[?!？！。]|$)", "更地", text)
    if not is_ark_context(title, text, recent):
        return text
    # Preserve genuine "ギガの土地", don't rewrite broad homophones.
    text = re.sub(r"ギガのと(?=$|[、。！？!? \t]|[がをにはで]|討伐|チェック|えぐ)", "ギガノト", text)
    text = re.sub(r"ギガノと(?=$|[、。！？!? \t]|[がをにはで]|討伐|チェック|えぐ)", "ギガノト", text)
    # Small, attested ARK-only Haas confusions. Never globally rewrite
    # homophones such as ギガの土地 ("Giga's land").
    text = re.sub(r"気がの都(?=$|[がをにはで]|討伐|襲来)", "ギガノト", text)
    text = re.sub(r"ギガナと(?=$|[がをにはで]|討伐|どっち)", "ギガノト", text)
    text = re.sub(r"ギガノド(?=$|[がをにはで]|討伐|だけ)", "ギガノト", text)
    return re.sub(r"テーム(?=した|する|して|済み|でき)", "テイム", text)


def repair_peko_nickname(text: str, *, title: str = "", caption: str = "",
                         recent: tuple[str, ...] | list[str] = ()) -> str:
    """Require independent name evidence; don't convert generic honorifics."""
    caption_support = any(alias in caption for alias in _PEKO_CAPTIONS)
    recent_support = any(any(name in phrase for name in _PEKO_FORMS) for phrase in list(recent)[-8:])
    if caption_support or recent_support or _PEKO_TITLE.search(title):
        for misheard in ("テコちゃん", "てこちゃん"):
            text = text.replace(misheard, "ぺこちゃん")
    # 猫ちゃん is legitimate "kitten"; only accept nearly identical caption text.
    if caption_support and "猫ちゃん" in text:
        proposed = text.replace("猫ちゃん", "ぺこちゃん")
        if SequenceMatcher(None, proposed, caption).ratio() >= 0.90:
            text = proposed
    return text


def repair_verified_caption(asr: str, caption: str) -> str:
    """Repair one tiny spelling difference; caller checks timing and coverage."""
    if len(asr) < 12 or len(caption) < 12:
        return asr
    matcher = SequenceMatcher(None, asr, caption, autojunk=False)
    changes = [entry for entry in matcher.get_opcodes() if entry[0] != "equal"]
    if len(changes) != 1:
        return asr
    tag, i, j, x, y = changes[0]
    old, new = asr[i:j], caption[x:y]
    if tag != "replace" or not old or not new or max(len(old), len(new)) > 2:
        return asr
    # Two different kana in テコちゃん -> ぺこちゃん score 0.9167,
    # just below the generic 0.92 threshold. Permit ONLY this well-aligned
    # known nickname pair at 0.90; do not weaken unrelated caption repairs.
    peko_nickname = (
        old in {"テコ", "てこ"}
        and new in {"ぺこ", "ペコ"}
        and asr[j:j + 3] == "ちゃん"
        and caption[y:y + 3] == "ちゃん"
    )
    if matcher.ratio() < (0.90 if peko_nickname else 0.92):
        return asr
    if _NUMBERS.search(old + new) or not _JAPANESE.search(old + new):
        return asr
    # Very short negation/polarity changes must never be trusted to YouTube.
    surround = asr[max(0, i - 2):j + 3] + caption[max(0, x - 2):y + 3]
    if _NEGATION.search(surround):
        return asr
    return asr[:i] + new + asr[j:]


def is_incomplete_fragment(text: str) -> bool:
    text = text.strip().rstrip("…")
    return not text.endswith(("。", "！", "!", "？", "?")) and bool(_UNFINISHED.search(text))
