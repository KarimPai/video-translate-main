import base64
import json
import sys

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


def send(data):
    print(json.dumps(data, ensure_ascii=True), flush=True)


def main():
    model_id = sys.argv[1]

    try:
        model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )

        send({"ready": True})

    except Exception as exc:
        send({"ready": False, "error": str(exc)})
        return

    for line in sys.stdin:
        try:
            request = json.loads(line)

            audio = np.frombuffer(
                base64.b64decode(request["audio"]),
                dtype=np.float32,
            )

            results = model.transcribe(
                audio=(audio, 16000),
                language="Japanese",
            )

            text = results[0].text if results else ""

            send({"text": text})

        except Exception as exc:
            send({"error": str(exc)})


if __name__ == "__main__":
    main()