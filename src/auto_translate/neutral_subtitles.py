"""Conservative neutrality for Whisper's direct Japanese-audio-to-English AST.

Unlike Haas+PLaMo, this pipeline has no Japanese transcript by default.
Only a verified Japanese re-decode of the CURRENT audio can authorize a
limited English rewrite. A missing/uncertain verifier result keeps the text.
"""
from __future__ import annotations
import re

_ROLE = (
    r"(?:customers?|clients?|guests?|visitors?|patients?|employees?|"
    r"shops?|stores?|business(?:es)?|teams?|cars?|vehicles?|homes?|"
    r"houses?|jobs?|articles?|reports?|streams?|friends?|"
    r"characters?|orders?|purchases?|sales?|deliver(?:y|ies))"
)
_QUAL = r"(?:(?:very|first|second|third|next|last|new|only|regular|loyal|best|favorite|favourite)\s+){0,2}"
_OWNED_ROLE = re.compile(
    rf"\b(?P<owner>my|our|your|their|his|her)\s+(?P<role>{_QUAL}{_ROLE})\b", re.I
)
_SIMPLE_GENDERED = re.compile(
    r"^(?:he|she)\s+(?P<act>left|arrived|came back|went away|"
    r"is here|was here|is gone|went home)(?P<end>[.!?]?)$", re.I
)
_SIMPLE_FIRST = re.compile(
    r"^(?:I (?:don't|do not) know|I'm not sure|I found it|"
    r"I(?:'ve| have)? finished(?: it)?|I'm done|"
    r"I(?:'ve| have)? got it)[.!?]*$", re.I
)
# A Japanese [owner]の is stronger evidence than an English possessive.
# Avoid general [word]の matching, since Japanese uses の for other reasons.
_JA_OWNER = re.compile(
    r"(?:私|わたし|ワタシ|俺|おれ|オレ|僕|ぼく|ボク|あたし|"
    r"うち|自分|私たち|俺ら|僕ら|あなた|あんた|お前|おまえ|"
    r"君|彼|彼女|彼ら|あの人|その人|みんな|皆|"
    r"[一-龯ぁ-んァ-ヶー]{1,12}(?:さん|ちゃん|くん|君))の"
)
_JA_GENDER = re.compile(
    r"彼(?!女)|彼女|男性|女性|男の人|女の人|男の子|女の子|"
    r"父|母|パパ|ママ|兄|姉|弟|妹|息子|娘|夫|妻"
)
_JA_FIRST = re.compile(
    r"私|わたし|ワタシ|俺|おれ|オレ|僕|ぼく|ボク|"
    r"あたし|うち|自分|私たち|俺ら|僕ら"
)
_JA_IMPLIED_FIRST = re.compile(
    r"思(?:う|います|って|った)|考えて(?:いる|ます)|"
    r"つもり|行ってきます|いただきます|"
    r"(?:(?:し|やり|行き|食べ|見|帰り|戻り|遊び|買い|使い|会い|寝)たい)"
)


def needs_reference_evidence(english: str) -> bool:
    """Avoid additional inference on normal subtitles."""
    phrase = english.strip()
    return bool(_OWNED_ROLE.search(phrase)
                or _SIMPLE_GENDERED.fullmatch(phrase)
                or _SIMPLE_FIRST.fullmatch(phrase))


def neutralize_with_japanese(english: str, japanese: str) -> str:
    """Apply only supported noun/short-clause rewrites, never invent owners."""
    if not english or not japanese or not japanese.strip():
        return english
    jp = japanese.strip()
    result = english
    if not _JA_OWNER.search(jp):
        def drop_owner(match: re.Match[str]) -> str:
            role = match.group("role")
            pre = result[:match.start()]
            post = result[match.end():]
            standalone = not pre.strip() and not post.strip(" \t\n\r.!?…。、:;")
            if standalone:
                return role[0].upper() + role[1:] if match.group(0)[0].isupper() else role
            return ("The " if match.group(0)[0].isupper() else "the ") + role
        result = _OWNED_ROLE.sub(drop_owner, result)

    if not _JA_GENDER.search(jp):
        gendered = _SIMPLE_GENDERED.fullmatch(result.strip())
        if gendered:
            result = "Someone " + gendered.group("act") + gendered.group("end")

    if not (_JA_FIRST.search(jp) or _JA_IMPLIED_FIRST.search(jp)):
        short = result.strip()
        if re.fullmatch(r"I (?:don't|do not) know[.!?]*|I'm not sure[.!?]*", short, re.I):
            if re.search(r"分からな|わからな|知らな", jp):
                return "Not sure."
        if re.fullmatch(r"I found it[.!?]*", short, re.I) and "見つけた" in jp:
            return "Found it."
        if (re.fullmatch(r"I(?:'ve| have)? finished(?: it)?[.!?]*|I'm done[.!?]*", short, re.I)
                and re.search(r"終わった|おわった|完了|終了", jp)):
            return "Finished."
        if (re.fullmatch(r"I(?:'ve| have)? got it[.!?]*", short, re.I)
                and re.search(r"分かった|わかった|了解", jp)):
            return "Got it."
    return result
