# Proper-name recognition and translation implementation

Implemented on the existing `feature/haas-whisper-turbo-dual-asr-20261114`
branch, audited from remote commit `fd3aae8c9b174f5cbaceed2666cd5f873d301c80`.
The FireRedVAD → Haas + Whisper Turbo → LFM → name resolution → PLaMo
architecture is retained. No additional permanently loaded model was added.

## Root causes

1. Matsuri correction combined sentence-wide event exclusions with local
   matches. A literal festival could suppress a person elsewhere, while a
   Hololive title or person-like grammar could incorrectly establish identity.
2. Miko depended on a caller-supplied context boolean and sentence-wide shrine
   exclusions. Kanji variants and mixed person/occupation occurrences were
   inconsistent across glossary matching, validation and correction.
3. English correction assumed one target and one output mention per family.
   It could not align multiple forms, repeated names or mixed literal meanings.
   A full name could also satisfy a missing separate short-name occurrence.
4. PLaMo's glossary is global lexical data, without occurrence offsets. A
   source spelling shared by a name and a literal noun cannot safely receive
   one global target. Bad historical English could reinforce the same error.
5. Haas did not send custom vocabulary; Whisper exposed only a small subset
   of its supported options. Titles must not substitute for acoustic evidence.
6. An unrelated legacy path rewrote Watanabe-like surnames into Watame based
   only on a title. This now requires matching, reliable aligned captions and
   repairs individual supported spans, preserving actual surnames.

Official source revisions, supported API shapes and deployment compatibility
checks are documented in [the upstream audit](proper-name-upstream-audit.md).

## Final decisions and model options

`proper_names.resolve_names` reads existing Japanese occurrences and preserves
their character spans and alternative ASR spellings. Each receives `PERSON`,
`COMMON_NOUN`, or `UNKNOWN`, with evidence categories rather than probability
scores. Explicit spoken names/nicknames and honorifics keep their supplied
English form. Bare Matsuri/Miko need compatible local grammar and confirmed
Japanese person context or verified participants. Metadata and chat never
establish a spoken name. Literal compounds and occurrence-local shrine roles
remain literal. A missing name is never reconstructed from metadata.

Only unresolved occurrences invoke a separate constrained JSON classification
on the existing LFM instance, at temperature 0. Its answer cannot rewrite
Japanese or override strong literal evidence. Invalid output, cancellation,
timeout and insufficient context abstain safely. Selection and classification
share the existing LFM budget, GPU admission and lifecycle management.

Haas supports per-request `keywords`, `keywords_alpha`, and mandatory enabled
`keywords_guard`, gated by the actual worker/model's advertised options. Old
models decode normally without boosting. A specific option rejection allows
one unboosted retry under the original deadline. Reviewed recent spoken forms
are bounded; titles and full-roster lists are not vocabulary sources. Production
boosting remains disabled pending representative audio measurements.

Whisper exposes `hotwords` or `initial_prompt`, `beam_size`, `patience`,
`length_penalty`, `temperature`, `best_of`, `repetition_penalty`,
`no_repeat_ngram_size`, `word_timestamps`, `log_prob_threshold`,
`no_speech_threshold`, `compression_ratio_threshold`, `suppress_tokens`, and
`max_new_tokens`. The actual installed `transcribe` signature filters optional
arguments. Japanese transcription, `condition_on_previous_text=False`, original
audio, hallucination checks and existing targeted crop retries are retained.
Default beam remains 6; default prompt mode remains `none`. Both configuration
examples document the controls without overwriting user settings.

PLaMo receives only current, supported lexical pairs using the official
`chat_template_kwargs.glossary` list of `{source,target}` objects. Conflicting
or substring-overlapping literal/name glossary entries are omitted. Individual
span decisions accompany the original source as context. Safe single English
name spans can be corrected; plural agreement and mixed nouns prohibit unsafe
literal replacement. One selective retry romanizes only confirmed PERSON spans
in a temporary translation input, retaining the final Japanese unchanged.
Persistent confirmed-name/literal errors or unsupported names suppress that
English cue with explicit diagnostics, while later cues continue normally.
Possible omitted literal nouns receive a bounded review retry; this heuristic
does not claim to validate every English paraphrase.

Context clears on seek/reset; cached Japanese can reestablish confirmed recent
dialogue. Contaminated or ambiguous history does not become a translation
example. Lap, Lap-sama, Lap-chan, Laplus, Miko, Mikochi, Miko-chan, Matsuri,
Matsuri-chan, Natsuiro Matsuri and Watame retain their spoken lengths. Existing
nickname policy is preserved; profile-backed Moona, Kaela and Mococo short forms
were added with sources in the glossary.

Asynchronous bounded JSONL diagnostics link each cue's original ASR alternatives,
keyword/prompt modes, LFM selection, classifications/evidence/abstentions, final
Japanese, matched glossary, raw/corrected English, retries, failures and timing.
Audio arrays and full transcript history are excluded from comparison records.

## Verification

| Evidence | Baseline | Final local implementation |
|---|---:|---:|
| Full offline unittest suite | 404 tests, 19 skips, passed | 495 tests, 19 skips, passed |
| Synthetic expected names retained | 17/20 | 20/20 |
| Synthetic false names / negative slots | 5/16 | 0/16 |
| Synthetic literal noun slots preserved | 7/7 | 7/7 |
| Deterministic English fixture repairs | 23/24 | 24/24 |

The full local suite completed in 5.273 seconds. Ruff's repository CI selection
`E9,F63,F7,F82`, compilation and `git diff --check` passed. The Python wheel built
successfully; its archive was checked to include the resolver and exclude scratch,
attachments and logs. The 19 local skips cover Windows Job Objects/session
lifecycle, optional Qt, and one optional DeepFilter dependency.

The synthetic results use 32 authored cases and actual baseline functions loaded
from the immutable baseline commit. They measure deterministic text behavior,
not acoustic accuracy, statistical generalization, LFM judgment accuracy or
PLaMo model translation accuracy. Full per-case observations are in
[proper-name-text-results.json](proper-name-text-results.json); reproduction and
the recorded-audio experiment matrix are in
[proper-name-benchmarks.md](proper-name-benchmarks.md).

The Windows workflow runs the complete suite with Qt, real Windows ownership
checks, GUI/control smoke checks and PyInstaller packaging, then publishes
`VideoTranslate-EXE-Only` and `VideoTranslate-Windows-GUI`. The task's final
delivery links the run for the pushed source commit and its artifact results;
the Linux wheel is not evidence of a successful Windows EXE build.

## Remaining validation limits

No reusable audio corpus, speech.cpp executable/model, faster-whisper runtime,
LFM/PLaMo server, RTX 3070 or Windows 11 host was available locally. Consequently
Haas alpha 1.0/1.5/2.0, Whisper prompt modes and beam 6/8/10, real pronunciation
recall, false acoustic names, literal English accuracy, CPU/GPU or quantization
accuracy and peak VRAM have no measured results here. The supplied sequential
audio harness records real observations when those resources are available;
unsupported options and fallback runs are explicitly marked.

Heuristic grammar remains conservative and incomplete. Both ASRs can agree on
an incorrect word; no text resolver proves the original audio. Missing or
contradictory evidence remains UNKNOWN rather than forcing a name. A model that
repeatedly ignores confirmed span guidance can still cause an explicitly logged
English abstention. Maximum LFM evidence can exceed its finite context window;
server rejection falls back to UNKNOWN. Representative audio, real model
inference and end-to-end playback on the user's target hardware remain required
before asserting improved acoustic accuracy or full deployment acceptance.

## Modified files

- `.github/workflows/windows-gui.yml`
- `config.dual-asr.example.toml`
- `config.example.toml`
- `docs/proper-name-benchmarks.md`
- `docs/proper-name-implementation.md`
- `docs/proper-name-text-results.json`
- `docs/proper-name-upstream-audit.md`
- `glossary.toml`
- `scripts/benchmark_proper_names.py`
- `src/auto_translate/asr/__init__.py`
- `src/auto_translate/asr/diagnostics.py`
- `src/auto_translate/asr/dual.py`
- `src/auto_translate/asr/haas.py`
- `src/auto_translate/asr/reconciler.py`
- `src/auto_translate/asr/whisper_turbo.py`
- `src/auto_translate/config.py`
- `src/auto_translate/context_memory.py`
- `src/auto_translate/glossary.py`
- `src/auto_translate/name_hints.py`
- `src/auto_translate/pipeline.py`
- `src/auto_translate/proper_names.py`
- `src/auto_translate/translate/plamo.py`
- `src/auto_translate/translation_guard.py`
- `src/auto_translate/types.py`
- `src/auto_translate/watame.py`
- `tests/fixtures/proper_name_cases.json`
- `tests/test_asr_name_options.py`
- `tests/test_context_memory_extended.py`
- `tests/test_dual_pipeline_stability.py`
- `tests/test_hololive_name_hints.py`
- `tests/test_name_disambiguation.py`
- `tests/test_name_pipeline.py`
- `tests/test_name_translation.py`
- `tests/test_proper_names.py`
- `tests/test_watame_acoustic_evidence.py`
