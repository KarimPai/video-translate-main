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


# the translator may express unsupported attribution as a predicate ("I'm bad at
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
    okay?") as the translator generated it. This is a conservative fallback, not a
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
