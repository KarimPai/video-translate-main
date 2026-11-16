"""Conservative Japanese comparison and exact two-candidate selection."""
import re
import math
import unicodedata
from difflib import SequenceMatcher

from ..text_filter import KATA_TO_HIRA


def normalize(text):
    value = unicodedata.normalize("NFC", text)
    # Fold width only. Preserve kanji, particles, prolonged sounds and decimal points.
    value = re.sub(r"[！-～｡-ﾟ]+", lambda m: unicodedata.normalize("NFKC", m[0]), value)
    value = value.translate(KATA_TO_HIRA)
    return re.sub(r"(?<![0-9])[.,]|[.,](?![0-9])|[\s、。！？!?「」『』…]", "", value)


def compare(a, b, terms, scene_pairs=()):
    a, b = normalize(a), normalize(b)
    similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
    flags = []
    for label, pattern in (("negation", r"ない|なく|なかっ|ません|せん|ぬ|ず"),
                           ("number", r"[0-9一二三四五六七八九十百千万億〇零]+")):
        if re.findall(pattern, a) != re.findall(pattern, b):
            flags.append(label)
    differences = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    particles = set("はがをにへともので")
    if any(tag == "replace" and a[i:j] in particles and b[k:l] in particles
           for tag, i, j, k, l in differences):
        flags.append("particle")
    if any((x in a and y in b) or (y in a and x in b) for x, y, _ in scene_pairs):
        flags.append("confusable term")
    if any((normalize(t) in a) != (normalize(t) in b) for t, _ in terms):
        flags.append("glossary")
    if a != b and (a in b or b in a or abs(len(a)-len(b)) >= 4):
        flags.append("missing phrase")
    return similarity, flags


def selected(content, haas, secondary):
    if content not in ("A", "B", "U"):
        raise ValueError("selector must return exactly A, B or U")
    label = {"A": "haas", "B": "whisper_turbo", "U": "uncertain"}[content]
    return {"selection": label, "selected_japanese": secondary if content == "B" else haas,
            "uncertain": content == "U", "reason": "uncertain; using Haas" if content == "U"
            else "selected original hypothesis"}


# Scene homophones are configurable via [asr_rules] in glossary.toml.
_CLAUSE_END = re.compile(r"(?<=[。！？!?、])")


def clause_differences(haas, whisper):
    """Show bounded, ordered differences to the selector without proposing new speech."""
    left = [s for s in _CLAUSE_END.split(haas) if s]
    right = [s for s in _CLAUSE_END.split(whisper) if s]
    matcher = SequenceMatcher(None, list(map(normalize, left)), list(map(normalize, right)),
                              autojunk=False)
    return [{"haas": "".join(left[i:j])[:96], "whisper": "".join(right[k:l])[:96]}
            for op, i, j, k, l in matcher.get_opcodes() if op != "equal"][:4]


def scene_repair(chosen, other, title, history, scene_pairs=()):
    if chosen == other or not re.search(r"GTA|ＧＴＡ|ホロスサントス", title, re.I):
        return chosen, []
    clauses = _CLAUSE_END.split(chosen)
    peers = _CLAUSE_END.split(other)
    if len(clauses) != len(peers):
        return chosen, []
    evidence = []
    for i, (clause, peer) in enumerate(zip(clauses, peers)):
        for heard, term, anchors in scene_pairs:
            if clause.count(heard) != 1 or term in clause:
                continue
            candidate = clause.replace(heard, term, 1)
            if normalize(candidate) != normalize(peer):
                continue
            # Require aligned surrounding speech, plus independently supplied
            # topic evidence outside the disputed spelling.
            if len(normalize(clause.replace(heard, "", 1))) < 2:
                continue
            context = " ".join([title, *history, chosen])
            anchor = next((word for word in anchors if word in context), None)
            if anchor:
                clauses[i] = candidate
                evidence.append(dict(clause=i, original=heard, replacement=term,
                                     peer_clause=peer, context_term=anchor))
                break
    return "".join(clauses), evidence


def disagreement_details(haas, whisper):
    """Separate harmless width/kana/punctuation differences from lexical edits."""
    left, right = normalize(haas), normalize(whisper)
    if left == right:
        return {"kind": "agreement" if haas == whisper else "orthographic", "edits": []}
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return {"kind": "meaningful", "edits": [
        {"operation": tag, "haas": left[i:j][:48], "whisper": right[k:l][:48]}
        for tag, i, j, k, l in matcher.get_opcodes() if tag != "equal"
    ][:8]}


def targeted_retry_window(haas, whisper, confidence, duration_s):
    """Locate a short disputed Whisper segment; never guess word timestamps.

    Require equal ordered clause boundaries and a confidence-bearing timestamp
    for that exact clause. No whole-utterance retry or interpolation from text.
    Return only original candidate clauses so a retry cannot introduce words.
    """
    if not isinstance(confidence, dict) or duration_s <= 1:
        return None
    left = [part.strip() for part in _CLAUSE_END.split(haas) if part.strip()]
    right = [part.strip() for part in _CLAUSE_END.split(whisper) if part.strip()]
    if len(left) != len(right):
        return None
    differences = [(a, b) for a, b in zip(left, right) if normalize(a) != normalize(b)]
    # One localized disagreement only; otherwise a crop cannot select the
    # complete utterance and LFM must retain its normal abstention contract.
    if len(differences) != 1:
        return None
    a, b = differences[0]
    segments = confidence.get("segments", [])
    if not isinstance(segments, list):
        return None
    for segment in segments[:24]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            continue
        if normalize(segment["text"]) != normalize(b):
            continue
        start, end = segment.get("start"), segment.get("end")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in (start, end)):
            continue
        if not 0 <= start < end <= duration_s + .05:
            continue
        low = any(type(segment.get(key)) in (float, int) and math.isfinite(segment[key])
                  and predicate(segment[key]) for key, predicate in (
                      ("avg_logprob", lambda x: x <= -.65),
                      ("no_speech_prob", lambda x: x >= .35),
                      ("compression_ratio", lambda x: x >= 2.2)))
        begin, finish = max(0., start - .15), min(duration_s, end + .15)
        if low and .3 <= finish - begin <= min(4., duration_s * .8):
            return {"start_s": begin, "end_s": finish, "haas_clause": a, "whisper_clause": b}
    return None


def retry_is_supported(confidence):
    """A crop may corroborate an existing candidate only with usable signals."""
    if not isinstance(confidence, dict):
        return False
    logprob, silence = confidence.get("avg_logprob"), confidence.get("no_speech_prob")
    return (type(logprob) in (int, float) and math.isfinite(logprob) and logprob > -.6
            and type(silence) in (int, float) and math.isfinite(silence) and silence < .35)
