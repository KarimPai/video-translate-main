
import json
import subprocess
import time
import urllib.request
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:11434/api/chat"

MODELS = [
    ("QWEN", "hololive-qwen-4b"),
    ("GEMMA", "gemma4:e4b"),
]

SAMPLES = [
    "すぐ忘れちゃうんで",
    "忘れちゃうなりの",
    "対応をしております",
    "もう忘れちゃうのはもう直んないんで",
    "忘れちゃうなりの対応をね",
    "していきたいとおもいます",
    "はい、ということで本日も宜しくお願いします",
    "えー、そうじゃん、昨日なんか",
    "てかめっちゃストレス値がバグってた治ったのかなぁ?",
    "え、でもワンちゃんあれだよね",
    "サーバーリセットが買ってるからワンチャン治ってる説はある",
]

# Exact system prompt from the current application.
SYSTEM_PROMPT = r"""You are a highly accurate Japanese-to-English subtitle translator specializing in Hololive, VTubers, livestreams, gaming and Japanese internet culture.

Your priority is faithful meaning, natural English and consistency.

TRANSLATION PRIORITIES:
1. Preserve the original Japanese meaning.
2. Preserve all meaningful information and speaker intent.
3. Use natural conversational English.
4. Keep subtitles concise without removing important details.
5. Maintain consistency with previous translations and the glossary.

JAPANESE UNDERSTANDING:
- Interpret casual Japanese, contractions, slang, incomplete sentences and conversational speech.
- Understand implied subjects, omitted pronouns and context-dependent expressions.
- Use conversation context to resolve ambiguous words and expressions.
- Treat the supplied Japanese transcription as the primary source.
- Correct a likely transcription error only when the surrounding context strongly supports the correction.
- Never replace unclear words with guesses merely to make a sentence sound natural.
- If the meaning is uncertain, preserve the closest reasonable interpretation.

HOLOLIVE AND INTERNET CULTURE:
- Nakiri Ayame (τÖ╛Θ¼╝πüéπéäπéü) sometimes refers to herself as πÇîΣ╜ÖπÇì(πéê / yo). Interpret this as her first-person pronoun, meaning "I", "me" or "my" according to context. Do not translate it literally as "surplus" or "remainder". Preserve the self-reference naturally in English when meaningful.
- Recognize Hololive talents, generations, groups, managers, nicknames and fan names.
- Preserve official English spellings of talent names.
- Recognize Hololive terminology, memes, inside jokes and recurring references.
- Understand VTuber expressions, gaming terminology, streamer slang and Japanese internet abbreviations.
- Do not translate proper names literally.
- Prefer established English terminology over literal dictionary translations.

SPEECH AND PERSONALITY:
- Preserve humor, sarcasm, excitement, frustration and emotional tone.
- Translate reactions and interjections naturally.
- Preserve meaningful repetitions and speech patterns.
- Do not unnecessarily remove short responses or conversational details.
- Do not add explanations or information that the speaker did not express.

CONTEXT AND GLOSSARY:
- Earlier JA/EN lines are context only.
- Use context to understand pronouns, subjects, references and ongoing jokes.
- Follow glossary translations when the Japanese term matches the glossary entry.
- Preserve consistent translations for recurring names and terminology.
- Do not force glossary entries when the context clearly indicates a different meaning.

INPUT:
- Translate ONLY the Japanese text following "Translate:".
- Do not translate or repeat previous context lines.

OUTPUT:
- Output ONLY the English translation.
- Use natural, conversational English suitable for subtitles.
- Keep the translation concise but preserve the complete meaning.
- No explanations, notes, alternatives, romaji or quotation marks.
- No speaker labels.
- Do not output Japanese text.
"""


def load_glossary():
    path = ROOT / "glossary.toml"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing glossary: {path}"
        )

    with path.open("rb") as f:
        data = tomllib.load(f)

    return data.get("terms", {})


def build_prompt(japanese, glossary):
    hits = []
    remaining = japanese

    for term, english in sorted(
        glossary.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        if term in remaining:
            hits.append((term, english))
            remaining = remaining.replace(term, "\0")

    parts = []

    if hits:
        parts.append(
            "Glossary:\n" +
            "\n".join(
                f"{term} = {english}"
                for term, english in hits
            )
        )

    parts.append(f"Translate:\nJA: {japanese}")

    return "\n\n".join(parts)


def gpu_memory():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        return result.replace(",", " /") + " MiB"

    except Exception:
        return "Unavailable"


def ollama_status():
    try:
        return subprocess.check_output(
            ["ollama", "ps"],
            text=True,
            stderr=subprocess.STDOUT
        ).strip()

    except Exception:
        return "Unable to read Ollama status"


def send_request(model, prompt, unload=False):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": 0 if unload else "10m",
        "options": {
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.85,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            "num_predict": 120
        }
    }

    request = urllib.request.Request(
        API,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        }
    )

    started = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=300
    ) as response:
        result = json.load(response)

    elapsed = time.perf_counter() - started

    message = result.get("message", {})
    translation = (message.get("content") or "").strip()

    thinking = (message.get("thinking") or "").strip()

    if not translation and thinking:
        translation = "[EMPTY TRANSLATION; THINKING WAS RETURNED]"

    return translation, elapsed, result


def run_model(label, model, prompts):
    print("\n" + "=" * 75)
    print(f"{label}: {model}")
    print("=" * 75)

    times = []
    results = []

    for index, prompt in enumerate(prompts):
        last_request = index == len(prompts) - 1

        try:
            translation, elapsed, response = send_request(
                model,
                prompt,
                unload=last_request
            )

            times.append(elapsed)
            results.append(translation)

            print(f"\n{index + 1:02d}. {SAMPLES[index]}")
            print(f"    EN: {translation}")
            print(f"    Time: {elapsed:.2f}s")

            if index == 0:
                print("\n    Ollama status:")
                print("    " + ollama_status().replace("\n", "\n    "))
                print(f"    GPU memory: {gpu_memory()}")

        except Exception as error:
            results.append("[ERROR]")
            print(
                f"\n{index + 1:02d}. ERROR: "
                f"{type(error).__name__}: {error}"
            )

    if times:
        print(f"\n{label} average including initial load: "
              f"{sum(times) / len(times):.2f}s")

        if len(times) > 1:
            print(f"{label} average excluding first request: "
                  f"{sum(times[1:]) / len(times[1:]):.2f}s")

    print(f"GPU memory after test: {gpu_memory()}")

    return results, times


def main():
    print("Japanese to English model benchmark")
    print("===================================")

    print(f"Samples: {len(SAMPLES)}")

    glossary = load_glossary()

    print(f"Glossary entries: {len(glossary)}")
    print("Context: disabled for this controlled first pass")
    print("Thinking: disabled")
    print("Temperature: 0.2")
    print("Top-k: 20")
    print("Top-p: 0.85")
    print("Repeat penalty: 1.05")
    print("Context window: 8192")
    print("Maximum output: 120 tokens")

    print(f"\nInitial GPU memory: {gpu_memory()}")

    prompts = [
        build_prompt(sample, glossary)
        for sample in SAMPLES
    ]

    all_results = {}

    for label, model in MODELS:
        results, times = run_model(
            label,
            model,
            prompts
        )

        all_results[label] = {
            "model": model,
            "translations": results,
            "times_seconds": times
        }

    output = {
        "settings": {
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.85,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            "num_predict": 120
        },
        "samples": SAMPLES,
        "results": all_results
    }

    output_path = ROOT / "benchmark_results.json"

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)
    print(f"Results saved to: {output_path}")
    print("\nCompare:")
    print("- Meaning accuracy")
    print("- Natural conversational English")
    print("- Correct interpretation of ワンチャン")
    print("- Hallucinated details")
    print("- Omitted information")
    print("- Translation speed")


if __name__ == "__main__":
    main()
