"""Conservative guard for clearly broken ASR hypotheses.

Do not reject legitimate short Japanese reactions. Reject explicit nonspeech,
decoder loops, phantom outros and physically implausible output.
"""
import re
import unicodedata

from ..text_filter import rejection_reason

_JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z]{2,}")
_PHANTOM = re.compile(r"^(?:thank you for watching|subtitles? by .+|please subscribe|ご視聴ありがとうございました)(?:[.!。！ ]*)$", re.IGNORECASE)


def transcript_problem(text: str, duration_s: float) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    reason = rejection_reason(text)
    if reason:
        return reason
    normalized = unicodedata.normalize("NFKC", text).strip()
    if _PHANTOM.fullmatch(normalized):
        return "phantom_outro"
    if len(normalized) > 600 and duration_s < 8:
        return "implausible_length"
    # Japanese speech peaks near 10 morae/s; allow generous slack for VAD padding.
    if len(_JA.findall(normalized)) > 24 + duration_s * 14:
        return "implausible_speech_rate"
    words = _WORD.findall(normalized)
    if not _JA.search(normalized) and len(words) >= 6 and duration_s < 1:
        return "implausible_speech_rate"
    return None


def whisper_suspect_reason(text: str) -> str | None:
    """Flag phonetic decoder loops for Whisper-specific independent verification.

    Unlike the shared rejection guard, this is a *suspicion* detector: true
    speech can contain repeated words, so callers must confirm with a fresh
    unprompted decode before accepting a replacement. Only Whisper uses it.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    normalized = unicodedata.normalize("NFKC", text)
    # Non-Japanese repeated ideographs (e.g. 喋喋喋喋喋) are almost always a
    # decoder degeneration, and are not caught by kana-only loop detection.
    if re.search(r"([\u3400-\u9fff])\1{3,}", normalized):
        return "whisper_character_loop"

    kana = "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c
        for c in normalized
        if ("ぁ" <= c <= "ゖ") or ("ァ" <= c <= "ヶ") or c == "ー"
    )
    # Remove all punctuation/spacing and detect repeated syllable groups
    # even when the ASR inserts commas between copies. Avoid marking normal
    # short emphasis such as いやいや / います、います as hallucination.
    if len(kana) >= 10:
        for width in range(2, 10):
            for match in re.finditer(rf"(.{{{width}}})\1{{3,}}", kana):
                unit = match.group(1)
                # Normal spoken Japanese may emphasize familiar particles
                # or reactions repeatedly; don't flag those solely as loops.
                if unit in {"いや", "はい", "うん", "そう", "もう", "だめ", "やだ"}:
                    continue
                # Fast repeated reactions in chaotic gameplay are genuine
                # speech candidates, not proof that the entire utterance
                # hallucinated. HG05/HG21 were dropped even though the repeat
                # occupied only ~25% of the Japanese kana in each transcript.
                # Treat a repetition as suspicious only when it dominates
                # the kana stream, or is itself exceptionally long.
                repeated = len(match.group())
                if ((repeated >= 14 or (width <= 3 and repeated >= 10))
                        and (repeated * 2 >= len(kana) or repeated >= 40)):
                    return "whisper_phonetic_loop"
        if re.search(r"([ぁ-ゖ])\1{4,}", kana):
            return "whisper_phonetic_loop"

    # A long all-katakana hypothesis with no Japanese grammar is
    # *uncertain*, not necessarily wrong: VTuber names and game titles can
    # be Katakana. Ask for a second decoder agreement, never blindly ban it.
    if len(normalized.strip(" 、。！？!?")) >= 12 and re.fullmatch(
        r"[ァ-ヴー・！？!?,、。〜～\s]+", normalized
    ) and len(re.findall(r"[ァ-ヴー]", normalized)) >= 12:
        return "whisper_long_katakana"

    return None
