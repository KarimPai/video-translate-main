"""Conservative post-ASR filtering for non-linguistic Japanese output."""

import re
import unicodedata

import numpy as np

# Common Japanese laughter spellings and chat-style laughter.
_LAUGHTER = re.compile(
    r"(?:あ?は){2,}|あは+|わは+|ふふ+|へへ+|ほほ+|くく+|"
    r"ｗ{2,}|w{3,}|笑+|草+",
    re.IGNORECASE,
)
_NON_TEXT = re.compile(r"[\s、。，．・！？!?…〜～.,:;()（）「」『』【】\[\]{}<>]+")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]+")


def _compact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _NON_TEXT.sub("", normalized)



# Remove caption/ASR sound-effect labels, not actual spoken words.
_SOUND_WORDS = (
    r"applause|clapping|claps?|cheering|crowd cheering|crowd noise|footsteps|gunfire|"
    r"music|background music|bgm|instrumental|singing|"
    r"engine(?: noises?)?|engine revving|revving|motor noise|"
    r"sound effects?|sfx|laughter|laughing|laughs|"
    r"エンジン音|車のエンジン音|足音|銃声|手拍子|拍手|拍手喝采|歓声|声援|音楽|背景音楽|効果音|笑い声|笑い|爆笑"
)
_SOUND_BRACKET = re.compile(
    rf"[\[\(（【［]\s*(?:{_SOUND_WORDS})\s*[\]\)）】］]",
    re.IGNORECASE,
)
_SOUND_ONLY = re.compile(rf"^(?:{_SOUND_WORDS})$", re.IGNORECASE)
_MUSIC_MARKS = re.compile(r"[♪♫♬♩]+")
_SPACES = re.compile(r"\s+")


def strip_nonverbal_labels(text: str) -> str:
    """Remove explicit sound-event labels while preserving adjacent speech."""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned, removed = _SOUND_BRACKET.subn(" ", normalized)
    if not removed:
        cleaned = text
    cleaned = _MUSIC_MARKS.sub(" ", cleaned)
    cleaned = _SPACES.sub(" ", cleaned).strip()
    if not _compact(cleaned) or _SOUND_ONLY.fullmatch(
        unicodedata.normalize("NFKC", cleaned).strip(".,!?。！？ ")
    ):
        return ""
    return cleaned

# These stock decoder completions are also legitimate dialogue. Suppress them
# only when an aligned caption independently identifies a sound-only event.
def is_sound_caption_artifact(text: str) -> bool:
    compact = _compact(text)
    if re.fullmatch(r"(?:パチ|ぱち){2,}|(?:パン|ぱん){2,}|(?:ドン|どん){2,}", compact):
        return True
    return compact in {
        "ご視聴ありがとうございました", "ご視聴ありがとうございます",
        "チャンネル登録お願いします", "チャンネル登録よろしくお願いします",
    }



def is_short_acknowledgement(text: str) -> bool:
    """Target only very short replies; never suppress surrounding dialogue."""
    compact = _compact(text)
    return compact in {"うん", "ウン", "ん", "んん", "うーん"}


def is_stationary_mechanical_audio(audio: np.ndarray) -> bool:
    """Conservative high-confidence stationary engine/tone check.

    This is not general speech recognition or a reliable engine classifier.
    Flag only sustained, nearly unchanged low-frequency harmonic audio.
    Ordinary changing speech, mixed voices and broadband game sound pass.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size < 8000 or not np.all(np.isfinite(samples)):
        return False
    # Maximum 8s of PCM; bounded CPU work and no additional model.
    samples = samples[:128000]
    window_size, hop = 1024, 512
    count = 1 + (samples.size - window_size) // hop
    if count < 12:
        return False
    # Use at most ~160 windows even for an 8s VAD utterance.
    indices = np.linspace(0, count - 1, min(count, 160), dtype=int) * hop
    blocks = samples[indices[:, None] + np.arange(window_size)]
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    peak = float(np.max(rms))
    if peak < 1e-3:
        return False
    active = rms >= peak * 0.35
    if int(np.count_nonzero(active)) < 10 or float(np.mean(active)) < 0.65:
        return False
    active_blocks = blocks[active]
    active_rms = rms[active]
    envelope_variation = float(np.std(active_rms) / np.mean(active_rms))
    if envelope_variation > 0.08:
        return False
    windowed = active_blocks * np.hanning(window_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(window_size, d=1 / 16000)
    band = spectrum[:, (frequencies >= 60) & (frequencies <= 4000)]
    low = spectrum[:, (frequencies >= 60) & (frequencies < 350)]
    total_energy = float(np.sum(band))
    if total_energy < 1e-10 or float(np.sum(low) / total_energy) < 0.88:
        return False
    magnitude = band / np.maximum(
        np.linalg.norm(band, axis=1, keepdims=True), 1e-12
    )
    similarity = np.sum(magnitude[1:] * magnitude[:-1], axis=1)
    return bool(np.median(similarity) > 0.999)


def caption_confirms_short_acknowledgement(text: str, caption: object | None) -> bool:
    """Only aligned captions containing the *same* ack can verify it."""
    if caption is None or not bool(getattr(caption, "reliable", False)):
        return False
    if (float(getattr(caption, "coverage", 0.0)) < 0.8
            or float(getattr(caption, "overlap_s", 0.0)) < 0.45):
        return False
    return _compact(str(getattr(caption, "text", ""))) == _compact(text)



def is_nonspeech_probe(audio: np.ndarray) -> bool:
    """Only for VAD-negative probes. Recognize silence, sparse clicks or a pure tone."""
    frame_size = 320  # 20 ms of mono PCM at 16 kHz
    count = audio.size // frame_size
    if count < 5:
        return False
    frames = audio[:count * frame_size].reshape(count, frame_size)
    peak = float(np.max(np.abs(audio)))
    if peak < 1 / 32768:
        return True
    active = np.abs(frames) > peak * 0.05
    # Every active frame contains at most 1 ms above threshold.
    if np.max(active.sum(axis=1)) <= 16:
        return True
    # Multi-harmonic stationary motor noise may evade the pure-tone check.
    if is_stationary_mechanical_audio(audio):
        return True
    # Check only sustained, nearly single-frequency signals (e.g. game tones).
    # A sinusoid obeys x[n+2] + x[n] = 2*cos(w)*x[n+1].
    middle = audio[1:-1]
    neighbors = audio[2:] + audio[:-2]
    power = float(np.dot(middle, middle))
    if power == 0:
        return False
    coefficient = np.dot(neighbors, middle) / power
    residual = neighbors - coefficient * middle
    return float(np.dot(residual, residual)) < 1e-7 * power


def rejection_reason(text: str) -> str | None:
    """Return why an ASR result should be dropped, or None if it looks usable.

    Deliberately conservative: short reactions and speech containing laughter
    are retained. Only laughter-only output and obvious repeated-character
    loops are rejected.
    """
    compact = _compact(text)
    if not compact:
        return "empty_after_normalization"

    label = unicodedata.normalize("NFKC", text).strip().casefold()
    if re.fullmatch(r"(?:\[|\(|【)\s*(?:笑い声|笑い|爆笑|laughter|laughs)\s*(?:\]|\)|】)", label):
        return "laughter_only"

    # Japanese ASR may spell the same laugh in hiragana or katakana.
    laugh_kana = "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in compact)
    # Excited laughter is often rendered as ギャーハハ / ぎゃはは.
    if re.fullmatch(r"(?:ぎゃ+[ーぁあ]*|きゃ+[ーぁあ]*)?(?:は|あは|わは){2,}[ーぁあ]*", laugh_kana):
        return "laughter_only"
    without_laughter = _LAUGHTER.sub("", laugh_kana)
    if compact and not without_laughter:
        return "laughter_only"

    # A single kana repeated many times is usually a vocalization or decoder
    # loop, not useful lexical content. Keep short reactions such as 「ええ」.
    if len(compact) >= 6 and len(set(compact)) == 1:
        return "repeated_character"

    # Detect obvious 1-3 character kana loops (e.g. あいあいあいあい).
    # Require at least four repetitions to avoid rejecting normal emphasis.
    if len(compact) >= 8 and _KANA.fullmatch(compact):
        for width in (1, 2, 3):
            unit = compact[:width]
            if len(compact) % width == 0 and compact == unit * (len(compact) // width):
                if unit in ("はい", "うん"):
                    return None
                return "repeated_kana_loop"

    # Filter long outputs containing no Japanese text only when they are
    # mostly punctuation/symbol noise. Legitimate English names are retained.
    visible = [ch for ch in compact if ch.isalnum()]
    if len(visible) >= 8 and not _JAPANESE.search(compact):
        if sum(ch.isalpha() for ch in visible) / len(visible) < 0.5:
            return "symbol_noise"

    return None


def is_usable_transcript(text: str) -> bool:
    return rejection_reason(text) is None


# Aggressive suppression is applied in run_asr *after* caption reconciliation.
# Restrict it to whole utterances so speech such as "うわー、これすごい" survives.
_REACTION_SEPARATORS = re.compile(
    r"[\s、。，．・!?！？…〜～,.;:：~「」『』（）()\[\]【】]+"
)
_REACTION_RUN = re.compile(
    r"(?:"
    r"あ+[ぁー]*っ?|は+[ぁー]*っ?|え+[ぇー]*っ?|"
    r"お+[ぉー]*っ?|う+[ぅー]*っ?|うわ[あぁー]*っ?|"
    r"わ[あぁー]+|ふ[うぅー]+|へ[えぇー]+|ほ[おぉー]+|"
    r"う[うぅー]*ん+|ん+[ー]?|"
    r"よいしょ|よっこいしょ|え[ー]*っと|ええっと|あの[ー]*|はー+い|"
    r"すっ?ご[いぃっー]+|すっ?げ[えぇー]+|すすげ[えぇー]+|"
    r"やば[いぁっー]+|はい+|わお+|くっ|ぐっ|うぐ|"
    r"ah+|oh+|ooh+|uh+|um+|hm+|hmm+|huh+|"
    r"whoa+|woah+|wow+|hey+|yeah+|yep+|yup+|ugh+|sigh+"
    r")+"
)


def is_isolated_reaction(text: str) -> bool:
    """Return True only when the entire utterance is filler/interjections.

    Keep meaningful acknowledgements ("うん", "はい") and praise
    ("すごい") as complete dialogue. Drop only obvious isolated fillers,
    gasps, and other nonlinguistic reactions. Speech containing other words
    is preserved in full, including when an interjection starts a sentence.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in normalized
    )
    tokens = [part for part in _REACTION_SEPARATORS.split(normalized) if part]
    # Preserve meaningful short responses, even when they resemble filler.
    if any(re.fullmatch(r"(?:うん+|はい+|はー?い+|すっ?ご[いぃっー]+|すっ?げ[えぇー]+|やば[いぁっー]+|わお+|wow+|yeah+|yep+)", part) for part in tokens):
        return False
    # A solitary small tsu is an ASR clipping artifact, not a word. It was
    # previously sent to Qwen and surfaced as "[translation failed]".
    if tokens and all(set(part) == {"っ"} for part in tokens):
        return True
    return bool(tokens) and all(_REACTION_RUN.fullmatch(part) for part in tokens)


def is_whole_utterance_blocked(text: str, blocklist: list[str]) -> bool:
    """Match a custom ASR blocklist item exactly, ignoring punctuation.

    Do not reject an otherwise meaningful sentence merely because it contains
    a blocked filler word.
    """
    normalized = _compact(text)
    return bool(normalized) and any(
        normalized == _compact(phrase) for phrase in blocklist if phrase.strip()
    )
