"""Optional verified audio metadata. No classification is inferred from ASR/VAD."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AudioEvent:
    start_s: float
    end_s: float
    kind: str
    generation: int = 0
    verified: bool = False
    evidence: str = ""  # audio_classifier or human_audio_review; never an ASR tag
    speaker_id: str | None = None
    speaker_verified: bool = False


class EventHistory:
    def __init__(self, ttl_s=4.0, max_events=16):
        self.events = deque(maxlen=max_events)
        self.ttl_s = ttl_s
        self.generation = None

    def reset(self, generation):
        self.events.clear()
        self.generation = generation

    def add(self, event):
        if (not event.verified or event.evidence not in {"audio_classifier", "human_audio_review"}
                or event.kind not in {"laughter", "scream", "crying", "gasp", "shouting"}
                or not all(math.isfinite(x) for x in (event.start_s, event.end_s))
                or event.start_s < 0 or event.end_s < event.start_s):
            return False
        if event.generation != self.generation:
            self.reset(event.generation)
        self.events.append(event)
        return True

    def context(self, start_s, end_s, generation, speaker_id=None):
        if generation != self.generation:
            self.reset(generation)
        self.events = deque((e for e in self.events if start_s - e.end_s <= self.ttl_s),
                            maxlen=self.events.maxlen)
        kinds = []
        for event in self.events:
            if event.start_s > end_s:
                continue
            same_speaker = speaker_id is not None and event.speaker_verified and event.speaker_id == speaker_id
            # Without reliable attribution only overlapping events are relevant;
            # never carry an unknown speaker's emotion into the next utterance.
            if not same_speaker and (event.speaker_id is not None or event.end_s < start_s):
                continue
            if event.kind not in kinds:
                kinds.append(event.kind)
        if not kinds:
            return ""
        return "Verified nearby audio (not dialogue; do not translate or assign a speaker): " + ", ".join(kinds)
