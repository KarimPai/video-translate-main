"""Conservative, no-inference cleanup of unsupported subjects in short subtitles.

Japanese often omits subjects. Do not force generic English rewriting: only
normalize familiar gameplay noun phrases and complete collision clauses. These
edits run locally after translation and never make a second model request.
"""

import re

# Only explicit first-/second-person wording licenses the corresponding English
# pronoun. Japanese pronouns may appear in kana or kanji in Haas transcripts.
_FIRST_PERSON = re.compile(
    r"私|わたし|ワタシ|わたくし|ワタクシ|僕|ぼく|ボク|俺|おれ|オレ|"
    r"あたし|アタシ|あたい|オイラ|わし|ワシ|うちら|僕ら|俺ら|私たち"
)
_SECOND_PERSON = re.compile(
    r"あなた|アナタ|貴方|あんた|アンタ|お前|おまえ|オマエ|"
    r"君|きみ|キミ|おたく|オタク|そちら|そっち"
)
_THIRD_PERSON = re.compile(r"彼女?|彼ら|あの人|その人|あいつ|そいつ|あの子|その子")

# Restrict possessive removal to objects and performance concepts where
# "the ..." is idiomatic. Do not rewrite generic "my ...", "your ...",
# "my bad", or people's names.
_ACTIVITY_POSSESSIVE = re.compile(
    r"\b(?P<owner>my|your|our|his|her|their)\s+"
    r"(?P<activity>driving|steering|parking|aim|aiming|shooting|"
    r"gameplay|playing|timing|handling|controls|movement|"
    r"accuracy|performance|speed|skills?|reactions?|car|vehicle)\b",
    re.IGNORECASE,
)

# 当たる can mean many things, so it is only a collision cue when driving
# is mentioned nearby. ぶつかる and 衝突 clearly indicate collision.
_COLLISION_IN_JAPANESE = re.compile(
    r"ぶつか|ぶつけ|衝突|接触|事故|"
    r"運転.{0,35}当た[っる]|当た[っる].{0,35}運転"
)

# Only rewrite self-contained clauses with a predictable object. Unknown
# modifiers and objects are left intact rather than risking broken grammar.
_UNSUPPORTED_COLLISION_CLAUSE = re.compile(
    r"\b(?P<agent>I|you|we|they|he|she)\s+"
    r"(?:(?:just|accidentally|somehow)\s+)?"
    r"(?:(?:ended\s+up|managed\s+to)\s+)?"
    r"(?:hit|hitting|crashed|bumped|collided|ran)"
    r"(?:\s+(?:into|with|against))?"
    r"(?:\s+(?:it|that|them|something|someone|the\s+car|a\s+car|"
    r"another\s+car|the\s+wall|a\s+wall))?"
    r"(?=$|[.!?,;:—–])",
    re.IGNORECASE,
)


# PLaMO may express unsupported attribution as a predicate ("I'm bad at
# driving") instead of a possessive ("my driving is bad").
_DRIVING_SKILL_CLAUSE = re.compile(
    r"\b(?P<agent>I'm|I am|you're|you are)\s+"
    r"(?:bad|terrible|awful|not\s+(?:very\s+|that\s+)?good|not\s+great)"
    r"\s+at\s+driving\b",
    re.IGNORECASE,
)
_DRIVING_ACTION_CLAUSE = re.compile(
    r"\b(?P<agent>I|you)\s+(?:can't|cannot|don't|do not)"
    r"\s+drive\s+(?:well|properly)\b",
    re.IGNORECASE,
)


def _explicit_owner(japanese: str, owner: str) -> bool:
    """Is this English possessor overtly supported by the Japanese words?"""
    owner = owner.casefold()
    if owner in ("my", "our"):
        return bool(_FIRST_PERSON.search(japanese))
    if owner == "your":
        return bool(_SECOND_PERSON.search(japanese))
    if owner in ("his", "her", "their"):
        return bool(_THIRD_PERSON.search(japanese))
    return True


def _explicit_agent(japanese: str, agent: str) -> bool:
    agent = agent.casefold()
    if agent in ("i", "we"):
        return bool(_FIRST_PERSON.search(japanese))
    if agent == "you":
        return bool(_SECOND_PERSON.search(japanese))
    return bool(_THIRD_PERSON.search(japanese))


def neutralize_unsupported_subjects(japanese: str, english: str) -> str:
    """Avoid asserting invented ownership or blame in familiar game dialogue.

    An omitted Japanese subject is not evidence of who drove or collided. Keep
    all other translation text (including implied addressees such as "Are you
    okay?") as PLaMO generated it. This is a conservative fallback, not a
    general grammatical validator.
    """
    if not japanese or not english:
        return english

    def replace_owner(match: re.Match[str]) -> str:
        if _explicit_owner(japanese, match.group("owner")):
            return match.group(0)
        replacement = "The" if match.group("owner")[0].isupper() else "the"
        return f"{replacement} {match.group('activity')}"

    english = _ACTIVITY_POSSESSIVE.sub(replace_owner, english)

    if "運転" in japanese:
        def replace_skill(match: re.Match[str]) -> str:
            agent = match.group("agent")
            if _explicit_agent(japanese, "i" if agent.casefold().startswith("i") else "you"):
                return match.group(0)
            prefix = english[:match.start()]
            at_start = not prefix.strip() or bool(re.search(r"[.!?]\s*$", prefix))
            return "The driving isn't very good" if at_start else "the driving isn't very good"

        english = _DRIVING_SKILL_CLAUSE.sub(replace_skill, english)
        if re.search(r"下手|上手くない|うまくない|苦手|へた|ヘタ", japanese):
            english = _DRIVING_ACTION_CLAUSE.sub(replace_skill, english)

    if not _COLLISION_IN_JAPANESE.search(japanese):
        return english

    def replace_collision(match: re.Match[str]) -> str:
        if _explicit_agent(japanese, match.group("agent")):
            return match.group(0)
        prefix = english[:match.start()]
        sentence_start = not prefix.strip() or bool(re.search(r"[.!?]\s*$", prefix))
        return "There was a collision" if sentence_start else "there was a collision"

    return _UNSUPPORTED_COLLISION_CLAUSE.sub(replace_collision, english)


# A personal pronoun in English should not be inferred from a video title,
# the streamer's identity, previous English, or an unmarked Japanese subject.
# Japanese can also strongly imply first person through 思う / たい / つもり.
_FIRST_PERSON_IMPLIED = re.compile(
    r"思(?:う|います|って|った)|考えて(?:いる|ます)|"
    r"(?:(?:し|やり|行き|食べ|見|帰り|戻り|遊び|買い|使い|会い|寝)たい)"
    r"|つもり|行ってきます|いただきます|いただこう|"
    r"欲しい|ほしい|欲しー|ほしー|"
    r"ごめん(?:なさい)?|すみません|すいません|"
    r"(?:あげる|あげたい|あげよう|渡してあげる|差し上げる)|"
    r"(?:しよう|やろう|行こう|行きましょう|買おう|買おっ|取ろう)"
)
_FIRST_EN = re.compile(
    r"\b(?:I|I'm|I've|I'd|I'll|me|my|mine|myself|we|we're|we've|"
    r"we'll|we'd|us|our|ours|ourselves)\b", re.I
)
_MALE_JA = re.compile(r"彼(?!女)|男性|男の人|男の子|父|お父|パパ|兄|弟|息子|夫")
_FEMALE_JA = re.compile(r"彼女|女性|女の人|女の子|母|お母|ママ|姉|妹|娘|妻")
_MALE_EN = re.compile(r"\b(?:he|him|his|himself|he's|he'd|he'll)\b", re.I)
_FEMALE_EN = re.compile(r"\b(?:she|her|hers|herself|she's|she'd|she'll)\b", re.I)
# Preserve "you" where idiomatic, especially requests/commands: it does not
# assert gender and English often needs an addressee despite omitted Japanese.
_FIRST_SUBJECT_HEDGE = re.compile(r"^I (?:think|guess|suppose)(?: that)?\s+", re.I)
_NOT_SURE = re.compile(r"^(?:I (?:don't|do not) know|I'm not sure)[.!?]*$", re.I)
_SHORT_DONE = re.compile(r"^(?:I(?:'ve| have)? finished|I'm done|I finished(?: it)?)[.!?]*$", re.I)
_SHORT_FOUND = re.compile(r"^I found it[.!?]*$", re.I)
_SHORT_GOT_IT = re.compile(r"^I(?:'ve| have)? got it[.!?]*$", re.I)


# Only neutralize ownership on ordinary role/object noun phrases. We cannot
# safely erase every English possessive (e.g. "their own", "their way") without
# rewriting the rest of the sentence or changing the meaning.
_ROLE_POSSESSIVE = re.compile(
    r"\b(?P<owner>my|our|your|their|his|her)\s+"
    r"(?P<role>(?:(?:very|first|second|third|next|last|new|only|"
    r"regular|loyal|best|favorite|favourite)\s+){0,2}"
    r"(?:customers?|clients?|guests?|visitors?|patients?|"
    r"employees?|shops?|stores?|business(?:es)?|teams?|cars?|"
    r"vehicles?|homes?|houses?|jobs?|articles?|reports?|streams?|"
    r"friends?|characters?|orders?|purchases?|sales?|deliver(?:y|ies)))\b",
    re.IGNORECASE,
)
# Distinguish actual ownership from Japanese "最初のお客さん": the latter
# has an attributive の but names no owner. An English "your" must never be
# inferred merely from speaking to a customer/shopkeeper.
_STATED_OWNER = re.compile(
    r"(?:私|わたし|ワタシ|俺|おれ|オレ|僕|ぼく|ボク|"
    r"あたし|アタシ|うち|自分|私たち|俺ら|僕ら|"
    r"あなた|アナタ|あんた|アンタ|お前|おまえ|オマエ|"
    r"君|きみ|キミ|彼|彼女|彼ら|あの人|その人|みんな|皆|"
    r"[一-龯ぁ-んァ-ヶー]{1,12}(?:さん|ちゃん|くん|君))の"
)


def _unsupported_role_ownership(japanese: str, english: str) -> bool:
    return bool(
        _ROLE_POSSESSIVE.search(english)
        and not _STATED_OWNER.search(japanese)
    )


def neutralize_unwarranted_possessives(japanese: str, english: str) -> str:
    """Prefer "first customer" over guessed "their/your first customer".

    For an isolated subtitle remove the possessor entirely; within a longer
    sentence supply a grammatical definite article instead. Preserve any
    explicitly sourced owner. Never synthesize "your" from "their".
    """
    if not japanese or not english or not _unsupported_role_ownership(japanese, english):
        return english

    def replace_owner(match: re.Match[str]) -> str:
        role = match.group("role")
        prefix = english[:match.start()]
        suffix = english[match.end():]
        standalone = not prefix.strip() and not suffix.strip(" \t\r\n.!?…、。，:;")
        if standalone:
            return role[:1].upper() + role[1:] if match.group(0)[:1].isupper() else role
        article = "The" if match.group(0)[:1].isupper() else "the"
        return f"{article} {role}"

    return _ROLE_POSSESSIVE.sub(replace_owner, english)


def unsupported_person_reference(japanese: str, english: str) -> str | None:
    """Flag unsupported person/gender/ownership; not a semantic accuracy score."""
    if not japanese or not english:
        return None
    if _unsupported_role_ownership(japanese, english):
        return "unsupported_possessive_reference"
    if _MALE_EN.search(english) and not _MALE_JA.search(japanese):
        return "unsupported_male_reference"
    if _FEMALE_EN.search(english) and not _FEMALE_JA.search(japanese):
        return "unsupported_female_reference"
    if (_FIRST_EN.search(english)
            and not _FIRST_PERSON.search(japanese)
            and not _FIRST_PERSON_IMPLIED.search(japanese)):
        return "unsupported_first_person_reference"
    return None


# Common forms are adjusted *after* the PLaMo retry, not before. "That person"
# is grammatically singular regardless of the verb ("He runs" -> "That person
# runs"); naive he -> they would give "They runs". The model should normally
# produce a more natural non-personal rewrite; these are conservative fallbacks.
_GENDERED_SUBJECT = re.compile(r"\b(?:he|she)\b", re.I)
_GENDERED_OBJECT = re.compile(r"\b(?:him|her)\b", re.I)
_GENDERED_POSSESSIVE = re.compile(r"\b(?:his|hers)\b", re.I)
_GENDERED_REFLEXIVE = re.compile(r"\b(?:himself|herself)\b", re.I)
_HER_POSSESSIVE = re.compile(
    r"\bher(?=\s+(?:car|vehicle|name|friend|family|house|home|"
    r"phone|money|job|work|team|voice|role|place|turn|life|"
    r"stream|character|hands?|eyes?|head|things?|stuff|parents?|"
    r"brother|sister|mother|father|hat|clothes|gun|weapon|bag)\b)", re.I
)


def _case_like(word: str, replacement: str) -> str:
    if word.isupper():
        return replacement.upper()
    if word[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def neutralize_unwarranted_gender(japanese: str, english: str) -> str:
    """Limited gender-neutral fallback after model retry; preserve source cues.

    Avoid editing contractions (he's/he'd/she's/she'd) because their verb
    tense is ambiguous: a forced change may invert the meaning. They must be
    handled by the translator retry or retained as an unresolved warning.
    """
    if not english:
        return english
    result = english
    # If source explicitly states a male/female character, retain its gender
    # reference for the model to translate; do not rewrite it to "that person"
    # simply because the *other* gendered pronoun was mistakenly emitted.
    if _MALE_JA.search(japanese) or _FEMALE_JA.search(japanese):
        return english
    if not _MALE_JA.search(japanese):
        result = _GENDERED_REFLEXIVE.sub(
            lambda m: _case_like(m.group(), "themselves") if m.group().lower() == "himself"
            else m.group(), result)
        result = re.sub(r"\bhim\b", lambda m: _case_like(m.group(), "them"), result, flags=re.I)
        def rewrite_his(match: re.Match[str]) -> str:
            if match.group().lower() != "his":
                return match.group()
            remaining = result[match.end():].strip()
            # "It's his." -> "It's theirs.", but "his car" -> "their car".
            target = "theirs" if not remaining or remaining[0] in ".,!?;:" else "their"
            return _case_like(match.group(), target)

        result = _GENDERED_POSSESSIVE.sub(rewrite_his, result)
        result = re.sub(r"\bhe\b(?!['’])", lambda m: _case_like(m.group(), "that person"), result, flags=re.I)
    if not _FEMALE_JA.search(japanese):
        result = re.sub(r"\bherself\b", lambda m: _case_like(m.group(), "themselves"), result, flags=re.I)
        result = re.sub(r"\bhers\b", lambda m: _case_like(m.group(), "theirs"), result, flags=re.I)
        result = _HER_POSSESSIVE.sub(lambda m: _case_like(m.group(), "their"), result)
        result = re.sub(r"\bher\b", lambda m: _case_like(m.group(), "them"), result, flags=re.I)
        result = re.sub(r"\bshe\b(?!['’])", lambda m: _case_like(m.group(), "that person"), result, flags=re.I)
    return result


def neutralize_unwarranted_first_person(japanese: str, english: str) -> str:
    """Only rewrite idioms safely impersonal in this brief-subtitle setting."""
    if not english or _FIRST_PERSON.search(japanese) or _FIRST_PERSON_IMPLIED.search(japanese):
        return english
    if _NOT_SURE.fullmatch(english.strip()) and re.search(r"分からな|わからな|知らな", japanese):
        return "Not sure."
    if len(japanese) <= 20:
        if re.search(r"終わった|おわった|完了|終了", japanese) and _SHORT_DONE.fullmatch(english.strip()):
            return "Finished."
        if "見つけた" in japanese and _SHORT_FOUND.fullmatch(english.strip()):
            return "Found it."
        if re.search(r"分かった|わかった|了解", japanese) and _SHORT_GOT_IT.fullmatch(english.strip()):
            return "Got it."
    # Unsupported 'I think' from a simple utterance like すごいね can be
    # removed safely unless Japanese actually states 思う / 考える.
    if _FIRST_SUBJECT_HEDGE.match(english) and len(japanese) <= 24:
        tail = _FIRST_SUBJECT_HEDGE.sub("", english, count=1)
        return tail[:1].upper() + tail[1:] if tail else english
    return english
