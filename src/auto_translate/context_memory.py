"""Bounded recent-context hints and conservative repetition repair for JP subtitles.

History comes from the current sequential video segment only. It is cleared by
the translation pipeline on a seek. No extra ASR/model request is made.
"""

import re
import unicodedata
from collections import Counter
from typing import Iterable

# ASR commonly varies the spelling of short Katakana game terms. Do not
# fuzzy-rewrite longer sentences, Kanji, Hiragana speech or named people.
_KATAKANA_TERM = re.compile(r"[ァ-ヶー]{4,12}\Z")
_SPLIT = re.compile(r"([、。，,.!！?？\s]+)")
_TRIM = " \r\n\t、。，,.!！?？「」『』（）()"
_EN_TERM = re.compile(r"[A-Za-z][A-Za-z '\-]{1,27}\Z")
_IGNORE_EN = {
    "yes", "no", "okay", "ok", "hello", "sorry", "thanks", "thank you",
    "wait", "what", "there", "right", "yeah", "oh", "huh", "hey",
}

# A small, evidence-based scene tag is kept separately from the 5-line Qwen
# dialogue window. Never summarize previous English model output or guess
# names, horses, locations, game outcomes or actions.
_SCENE_LABELS = {
    "horse_racing": "Japanese horse racing and betting",
    "ark": "ARK survival gameplay",
    "gta": "GTA driving/crime gameplay",
}
# Evidence strength: 2 = independently diagnostic, 1 = weak on its own.
# A single generic mention of 車, 新聞, レース, etc. must not invent a scene.
_SCENE_CUES = {
    "horse_racing": {
        "競馬": 2, "馬券": 2, "オッズ": 2, "パドック": 2,
        "推奨馬": 2, "三連複": 2, "三連単": 2, "単勝": 2,
        "複勝": 2,
        # Cashing out occurs in both betting and retail; one mention
        # cannot establish that the scene is horse racing.
        "払戻し": 1, "払い戻し": 1,
        "ダート": 1, "ワイド": 1, "レース": 1,
    },
    "ark": {
        "ギガノト": 2, "テイム": 2, "クライオポッド": 2,
        "アルゲンタヴィス": 2, "恐竜": 2,
        "アークサバイバル": 2, "ARK": 2, "サドル": 1,
    },
    "gta": {
        "グラセフ": 2, "グランドセフト": 2, "GTA": 2,
        "カージャック": 2, "カーチェイス": 2,
        "パトカー": 1, "警察": 1, "逃走": 1,
    },
}
_SCENE_DETAILS = {
    "horse_racing": (
        ("馬券", "betting tickets"), ("新聞", "racing newspapers"),
        ("オッズ", "betting odds"), ("パドック", "paddock"),
        ("払戻し", "payouts"), ("払い戻し", "payouts"),
        ("ワイド", "wide bets"), ("ダート", "dirt races"),
        ("推奨馬", "recommended horses"),
    ),
    "ark": (
        ("ギガノト", "Giganotosaurus"), ("テイム", "taming"),
        ("恐竜", "dinosaurs"), ("サドル", "saddles"),
        ("クライオポッド", "Cryopods"),
    ),
    "gta": (
        ("カージャック", "carjacking"), ("カーチェイス", "car chases"),
        ("パトカー", "police cars"), ("警察", "police"),
    ),
}


class SceneMemory:
    """Small scene context inferred only from recognized current Japanese.

    Scene hints are optional background for the current Qwen request, never
    evidence that a specific word/name was spoken in that request. One clear
    topic activates immediately; switching between established scenes needs
    two separate cues unless several independent signals occur together.
    Unconfirmed switches suppress the old hint instead of leaking stale
    context into a new topic.
    """

    def __init__(self, max_idle_lines: int = 32) -> None:
        self.max_idle_lines = max(8, min(96, int(max_idle_lines)))
        self.reset()

    def reset(self) -> None:
        self.topic: str | None = None
        self.lines = 0
        self.last_supported = 0
        self._details: dict[str, int] = {}
        self._pending_topic: str | None = None
        self._pending_count = 0
        self._pending_at = 0

    @staticmethod
    def _scores(text: str) -> dict[str, int]:
        normalized = unicodedata.normalize("NFKC", text)
        upper = normalized.upper()
        return {
            topic: sum(
                weight for word, weight in cues.items()
                if (
                    bool(re.search(
                        r"(?<![A-Z])" + re.escape(word.upper()) + r"(?![A-Z])",
                        upper,
                    ))
                    if word.isascii()
                    else word in normalized
                )
            )
            for topic, cues in _SCENE_CUES.items()
        }

    def _activate(self, topic: str) -> None:
        self.topic = topic
        self.last_supported = self.lines
        self._details.clear()
        self._pending_topic = None
        self._pending_count = 0
        self._pending_at = 0

    def observe(self, japanese: str) -> None:
        self.lines += 1
        source = unicodedata.normalize("NFKC", japanese[:512])
        scores = self._scores(source)
        ranked = sorted(scores, key=lambda name: scores[name], reverse=True)
        strongest = ranked[0] if ranked and scores[ranked[0]] > 0 else None
        score = scores[strongest] if strongest else 0
        # Equal evidence for multiple scenes is ambiguous ASR, not a
        # reason to choose an arbitrary category by dictionary order.
        ambiguous = (
            len(ranked) > 1 and score > 0 and scores[ranked[1]] == score
        )
        if ambiguous:
            strongest = None
            score = 0
            scores = {topic: 0 for topic in scores}

        if self.topic and self.lines - self.last_supported > self.max_idle_lines:
            self._activate_topic_expired()
        if self._pending_topic and self.lines - self._pending_at > 6:
            self._pending_topic = None
            self._pending_count = 0

        if self.topic is None:
            if strongest and score >= 2:
                self._activate(strongest)
        elif scores[self.topic] > 0 and (
            scores[self.topic] >= score or strongest is None
        ):
            self.last_supported = self.lines
            self._pending_topic = None
            self._pending_count = 0
        elif strongest and strongest != self.topic and score >= 2:
            if score >= 4:
                self._activate(strongest)
            elif self._pending_topic == strongest:
                self._pending_count += 1
                self._pending_at = self.lines
                if self._pending_count >= 2:
                    self._activate(strongest)
            else:
                self._pending_topic = strongest
                self._pending_count = 1
                self._pending_at = self.lines

        if not self.topic or ambiguous:
            return
        for word, label in _SCENE_DETAILS[self.topic]:
            if word in source:
                self._details[label] = self.lines
        self._details = {
            name: at for name, at in self._details.items()
            if self.lines - at <= self.max_idle_lines
        }

    def _activate_topic_expired(self) -> None:
        # Expire stale scene without resetting the monotonic line counter.
        self.topic = None
        self._details.clear()
        self._pending_topic = None
        self._pending_count = 0
        self._pending_at = 0

    def current_topic(self) -> str | None:
        if self.topic is None or self.lines - self.last_supported > self.max_idle_lines:
            return None
        # An emerging different scene is better left unclassified than
        # feeding the previous topic into an unrelated sentence.
        if self._pending_topic and self.lines - self._pending_at <= 6:
            return None
        return self.topic

    def hint(self) -> str:
        topic = self.current_topic()
        if topic is None:
            return ""
        recent = sorted(
            (
                (at, label) for label, at in self._details.items()
                if self.lines - at <= self.max_idle_lines
            ),
            reverse=True,
        )
        descriptions = [label for _, label in recent[:3]]
        suffix = "; earlier recognized terms: " + ", ".join(descriptions) if descriptions else ""
        return _SCENE_LABELS[topic] + suffix


def _word(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip(_TRIM))


def _one_edit(a: str, b: str) -> bool:
    """One substitution, deletion or insertion; bounded tiny strings only."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(left != right for left, right in zip(a, b)) == 1
    shorter, longer = sorted((a, b), key=len)
    i = j = errors = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            errors += 1
            if errors > 1:
                return False
            j += 1
    return True


def _recent_words(sources: Iterable[str]) -> list[str]:
    words: list[str] = []
    for ja in sources:
        source = _word(ja)
        if _KATAKANA_TERM.fullmatch(source):
            words.append(source)
        # Also recognize short, punctuated repetitions in a prior subtitle.
        elif len(source) <= 36:
            bits = [_word(part) for part in _SPLIT.split(source)]
            counts = Counter(x for x in bits if _KATAKANA_TERM.fullmatch(x))
            words.extend(word for word, count in counts.items() if count >= 2)
    return words



def _repair_compact_triple(japanese: str) -> str:
    """Repair e.g. シャッターシャッタシャッター without punctuation."""
    def replace(match: re.Match[str]) -> str:
        word = match.group()
        for size in range(4, min(12, (len(word) - 3) // 2) + 1):
            repeated = word[:size]
            if (
                not _KATAKANA_TERM.fullmatch(repeated)
                or not word.endswith(repeated)
            ):
                continue
            middle = word[size:-size]
            if len(middle) >= 3 and _one_edit(middle, repeated):
                return repeated * 3
        return word

    return re.sub(r"[ァ-ヶー]{11,36}", replace, japanese)



def repair_near_repeated_word(
    japanese: str,
    history: list[tuple[str, str]],
) -> str:
    """Correct only one-edit Katakana repeats supported by nearby Japanese.

    Examples:
        シャッター -> シャッタ (next cue) : restore シャッター
        シャッター、シャタ、シャッター : keep uncertain two-edit middle
        シャッター、シャッタ、シャッター : restore middle

    No correction when a prior term is absent, similarity is ambiguous,
    current Japanese is a longer sentence, or a different word is equally
    plausible. This intentionally cannot repair arbitrary ASR hallucinations.
    """
    if not japanese:
        return japanese

    japanese = _repair_compact_triple(japanese)
    parts = _SPLIT.split(japanese)
    tokens = [(i, _word(piece)) for i, piece in enumerate(parts)
              if _KATAKANA_TERM.fullmatch(_word(piece))]
    # For in-line repetitions, two exact repeats within this cue are strong.
    in_line_counts = Counter(word for _, word in tokens)
    strong = {word for word, count in in_line_counts.items() if count >= 2}
    recent = _recent_words(ja for ja, _ in history[-2:])
    standalone = bool(_KATAKANA_TERM.fullmatch(_word(japanese)))
    # Only allow one preceding occurrence when the CURRENT cue is a complete
    # short Katakana word; otherwise require two exact occurrences in-line.
    eligible = set(recent) if standalone else strong
    if not eligible:
        return japanese

    for i, token in tokens:
        matches = [reference for reference in eligible
                   if reference != token and _one_edit(token, reference)]
        if len(matches) != 1:
            continue
        raw = parts[i]
        # Keep outside punctuation and whitespace exactly as recognized.
        begin = len(raw) - len(raw.lstrip(_TRIM))
        end = len(raw.rstrip(_TRIM))
        if end < begin:
            continue
        parts[i] = raw[:begin] + matches[0] + raw[end:]
    return "".join(parts)



def reuse_recent_short_term(
    japanese: str,
    history: list[tuple[str, str]],
) -> str | None:
    """Reuse an identical isolated Katakana term's recent one-word English.

    This prevents e.g. シャッター -> Shutter / shutter / shudder on adjacent
    cues. Never reuse entire dialogue, names inferred from other Japanese,
    or multiword model guesses. A bad first translation may still persist
    briefly, so this memory is deliberately limited to the last two cues.
    """
    source = _word(japanese)
    if not _KATAKANA_TERM.fullmatch(source):
        return None

    for previous_ja, previous_en in reversed(history[-2:]):
        if _word(previous_ja) != source:
            continue
        target = previous_en.strip(_TRIM).strip("'\"“”")
        if (
            not _EN_TERM.fullmatch(target)
            or len(target.split()) != 1
            or target.casefold() in _IGNORE_EN
        ):
            return None
        punctuation = ""
        if japanese.rstrip().endswith(("？", "?")):
            punctuation = "?"
        elif japanese.rstrip().endswith(("！", "!")):
            punctuation = "!"
        return target + punctuation
    return None


def build_recent_context(
    history: list[tuple[str, str]],
    glossary,
    *,
    context_lines: int = 8,
    memory_lines: int = 24,
) -> str:
    """Short, non-authoritative memory beyond the normal the translator chat history."""
    recent = history[-max(0, memory_lines):] if memory_lines else []
    if not recent:
        return ""

    lines: list[str] = []
    found: list[str] = []
    seen: set[str] = set()
    # The static glossary is reliable even if past the translator output was not.
    for ja, _ in reversed(recent):
        for source, target in glossary.find(ja):
            if source in seen:
                continue
            seen.add(source)
            found.append(f"{source} = {target}")
            if len(found) >= 6:
                break
        if len(found) >= 6:
            break
    if found:
        lines.append("Recently mentioned glossary names/terms: " + "; ".join(found))

    # Only pass previous short isolated words as POSSIBLE translation hints:
    # past model output may itself be incorrect and must not become a rule.
    hints: list[str] = []
    seen.clear()
    for ja, en in reversed(recent[-10:]):
        source = _word(ja)
        target = en.strip(_TRIM).strip("'\"“”")
        if (
            not _KATAKANA_TERM.fullmatch(source)
            or source in seen
            or not _EN_TERM.fullmatch(target)
            or len(target.split()) > 2
            or target.casefold() in _IGNORE_EN
        ):
            continue
        seen.add(source)
        hints.append(f"{source} ~ {target}")
        if len(hints) == 4:
            break
    if hints:
        lines.append("Recent tentative term translations (use only if the current Japanese supports them): " + "; ".join(hints))

    # Older JP provides scene continuity, without reinforcing mistranslated
    # English pronouns. Keep it short to minimize tokens and latency.
    older = recent[:-context_lines] if context_lines > 0 else recent
    older_ja = [ja.strip()[:64] for ja, _ in older[-5:] if ja.strip()]
    if older_ja:
        lines.append("Earlier Japanese for scene continuity (do not translate): " + " / ".join(older_ja)[:195])
    return "\n".join(lines)
