# Upstream model API verification (2026-11-17)

These are source/API checks, not acoustic or translation quality benchmarks.

## speech.cpp / Haas

Official repository: `nyosegawa/speech.cpp`, inspected main tree
`8d18924cd238b87c0297df112f7ff40783a6414f` (commit dated 2026-11-15;
worker docs report version 0.11.0).

Sources:

- https://github.com/nyosegawa/speech.cpp/blob/8d18924cd238b87c0297df112f7ff40783a6414f/docs/models/fastconformer.md
- https://github.com/nyosegawa/speech.cpp/blob/8d18924cd238b87c0297df112f7ff40783a6414f/docs/worker.md
- https://github.com/nyosegawa/speech.cpp/blob/8d18924cd238b87c0297df112f7ff40783a6414f/docs/c-api.md
- https://github.com/nyosegawa/speech.cpp/blob/8d18924cd238b87c0297df112f7ff40783a6414f/docs/adr/haas-takes-haas-asrs-custom-vocabulary-through-the-request-options-keywords-keywords_alpha-and-keywords_guard.md

`ready.model.options` is a list of objects with `name`, `type`, `required`,
`steers`, and applicable range/default metadata. Per-request `transcribe` accepts
`keywords` as newline-separated `written form=reading,reading`, `keywords_alpha`
as a positive number (upstream default 2), and `keywords_guard` as a boolean
(default true). Alpha/guard without nonempty keywords are rejected. Empty or
whitespace-only written forms are rejected. The worker protocol uses a string,
unlike HTTP's keyword array.

Only Haas GGUF layout 4 with `fastconformer.vocabulary.kind=haas-asr` supports
these options. Haas files downloaded before speech.cpp 0.9.4 lack the vocabulary;
they load normally but must be downloaded again to support keywords. Capability
checks must use actual ready model options, not executable version alone.

`keywords` with `timestamps` is refused before inference. Keyword decoding uses
one encoder pass and two decodes, retaining only complete listed keyword changes
when guarded. This is a text guard, not proof a proper name was spoken.

The vocabulary formatter replaces recognized readings with the written form.
Consequently `夏色まつり=まつり` would expand an unspoken surname and must not be
used. Short spoken forms need separate vocabulary entries. A word such as
まつり/祭り or みこ/巫女 cannot be acoustically disambiguated by boosting alone.

Terminal errors are `{"type":"error","id":"...","error":{"code":
"unsupported"|"invalid_argument"|"out_of_range","option":"keywords",...}}`.
A safe compatibility fallback sends audio under a fresh ID without keyword
options, bounded by the original deadline. No second worker is necessary.

GPU encoder execution and CPU/GPU decoding are independently controlled;
quantized Haas files decode on CPU by default, F16/F32 on GPU. Different
arithmetic can change close word choices, so changing placement or quantization
requires actual audio comparisons.

## faster-whisper / Whisper Large-v3 Turbo

Repository project minimum is `faster-whisper>=1.2.1`; `uv.lock` pins 1.2.1.
Official v1.2.1 source
already supports every decoding option requested in the task:

https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py

Verified file blob: `51eb1c505bf1aa5d3a9fb5b7d2cd9811845b6558`. Latest master
also inspected at commit `6cb639e70786aea42b9233006ae60eaf07190bdb`, file blob
`d44fef739f3ac165a7146c5f1c1d9d7d4f18a3aa`.

`WhisperModel.transcribe` supports hotwords, initial_prompt, beam_size,
patience, length_penalty, temperature, best_of, repetition_penalty,
no_repeat_ngram_size, word_timestamps, log_prob_threshold,
no_speech_threshold, compression_ratio_threshold, suppress_tokens and
max_new_tokens. Inspect the installed method signature in the child worker;
source verification does not establish installed runtime compatibility.

At temperature 0 decoding uses beam_size/patience. At positive temperature it
forces beam_size=1 and uses best_of sampling hypotheses. A temperature tuple
can cause additional decodes after confidence/compression failures. hotwords
is a text hint in the previous-text prompt and is ignored when prefix is set.
It does not verify acoustic support. max_new_tokens plus prompt tokens must
fit model maximum length (448 in inspected implementation).

The BatchedInferencePipeline is not interchangeable with WhisperModel:
its documentation marks confidence thresholds, condition_on_previous_text,
and several other arguments unused. Preserve ordinary WhisperModel decoding
for confidence and hallucination behavior.

## PLaMo 3 Translate

Official model repository revision:
`pfnet/plamo-3-translate-2b@cf66d22fbe6202c56b8e5d283691aa1da53d05af`.

- https://huggingface.co/pfnet/plamo-3-translate-2b/blob/cf66d22fbe6202c56b8e5d283691aa1da53d05af/README.md
- https://huggingface.co/pfnet/plamo-3-translate-2b/blob/cf66d22fbe6202c56b8e5d283691aa1da53d05af/chat_template.jinja
- https://huggingface.co/pfnet/plamo-3-translate-2b/blob/cf66d22fbe6202c56b8e5d283691aa1da53d05af/tokenizer_config.json

Chat Completions accepts translation options through `chat_template_kwargs`.
glossary is a list of objects containing exactly source and target string
members, e.g. `[{"source":"まつりちゃん","target":"Matsuri-chan"}]`.
The template rejects a mapping/string or entries with extra keys. `context` is
plain text. History alternates user source / assistant translation pairs; only
the final user input is translated. Template places glossary immediately before
the current input, after history, so historical glossary contamination needs
separate filtering in the application.

The glossary schema has no occurrence offsets/span IDs. A lexical entry may
affect all occurrences of the same source term. Mixed person/literal occurrences
therefore require span-aware handling or abstention beyond ordinary glossary
injection. No source confirms that a glossary forces successful name retention;
validation remains necessary. History, context, glossary, source and translation
all count toward the context limit.

No models were loaded, no representative audio was tested, and no measured
accuracy or performance improvement is claimed in this research.

## Actual runtime limitations

The execution host is Linux, not the requested Windows 11/RTX 3070 host.
At inspection, `speech`, `llama-server` and `nvidia-smi` were absent, and
faster-whisper/CTranslate2 were not installed. The lightweight CI environment
provides NumPy/httpx and deterministic model doubles. Therefore installed
speech.cpp/model-file compatibility, actual CUDA/Vulkan execution, actual
LFM/PLaMo classifications, representative audio accuracy, peak VRAM, decoding
strength comparisons, and Windows executable behavior cannot be established
by these tests. Runtime capability detection is implemented for deployment
on the actual application host; upstream source verification alone is not an
installed-runtime validation.

## Existing failure paths identified

- `name_hints.matsuri_matches` mixed sentence-wide event checks with local
  matches. An event elsewhere could suppress a true person, while person-like
  grammar could promote an ambiguous term without a confirmed participant.
  Isolated kana forms also used Hololive titles as sufficient name context.
- `Glossary.find` gated Miko through one external context boolean and a
  sentence-wide shrine-word exclusion. It could not represent a confirmed
  person and literal shrine maiden separately in the same subtitle. Generic
  raw substring matching also needed protection for kanji spellings.
- Legacy name preservation used one target per family and at most one English
  hit. It could not safely align several names or repeated meanings in one
  cue; broad lexical replacement could not identify which festival was a name.
- `watame.correct_asr_name` converted every Watanabe-like occurrence into
  Watame based only on the video title, except when a caption used a phonetic
  Watanabe spelling. A real kanji surname or mixed cue could be overwritten.
  This was changed to require whole-line reliable aligned caption agreement,
  pair mentions in order, and repair only individual Watame-supported spans.
  A caption agreeing with a surname keeps that occurrence unchanged.
- `asr_correction.correct_known_asr_terms` still has reviewed title-gated
  Haachama/organization spelling aliases. It does not add Matsuri or Miko;
  its Lap repair requires a whole-line normalized matching peer. The pipeline
  passes only captions marked reliable. The legacy Haachama aliases are a
  spelling policy rather than newly measured acoustic evidence.
- `asr/selection.selected` admits only exact A/B/U labels and returns an
  original candidate. `scene_repair` requires an entire normalized matching
  peer clause and an independent scene anchor. The existing configured scene
  repairs are organization/common-word homophones, not person-name additions.

## Verified nickname evidence and scope

Official profile pages inspected on 2026-11-17:

- https://hololive.hololivepro.com/talents/sakuramiko/ explicitly answers
  `愛称は？` with `みこち、エリート巫女` (supports Mikochi).
- https://hololive.hololivepro.com/talents/airani-iofifteen/ includes the
  self-introduction `hololive AREA15のイオフィだよ！` (supports Iofi).
- https://hololive.hololivepro.com/talents/houshou-marine/ includes the
  self-introduction `宝鐘海賊団船長` (supports Senchou in person context).
- https://hololive.hololivepro.com/en/talents/moona-hoshinova/ includes
  `Moon Moon~ Moona da yo!`, supporting Moona without an unspoken surname.
- https://hololive.hololivepro.com/en/talents/kaela-kovalskia/ includes
  `Hello, everyone! I'm Kaela, the blacksmith of hololive holoh3ro!`.
- https://hololive.hololivepro.com/en/talents/mococo-abyssgard/ uses
  `The rambunctious Mococo` and `her older twin sister Fuwawa` in its
  biography, supporting both given-name forms without their surname.
- https://hololive.hololivepro.com/talents/natsuiro-matsuri/ and
  https://hololive.hololivepro.com/talents/la-darknesss/ verify canonical
  Matsuri and Laplus identities. Their profile text alone does not verify
  every colloquial nickname.

The English profiles for Koseki Bijou, Hakos Baelz and Mori Calliope did not
independently substantiate the colloquial aliases Biboo, Bae or Calli; those
aliases were not added on the strength of these profile checks.

The task itself specifies Lap/Lap-sama/Lap-chan, Laplus, Miko/Mikochi/Miko-chan,
Matsuri/Matsuri-chan, Natsuiro Matsuri and Watame as required preservation
forms. These user-specified renderings are output policy; they are not claims
that every form was independently verified by the above profiles. Ordinary
honorific morphology must not expand a nickname into a full stage name.
