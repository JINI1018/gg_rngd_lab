# https://huggingface.co/furiosa-ai/Qwen3-32B-FP8
# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 
# https://huggingface.co/furiosa-ai/Qwen3-VL-4B-Instruct
# furiosa-llm serve furiosa-ai/Qwen3-VL-4B-Instruct
# watch -n 1 furiosa-smi status


"""Call the FuriosaAI Qwen3-VL image-analysis API and print its answer."""

import os
import sys

import requests

API_URL = os.getenv("FURIOSA_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL = "furiosa-ai/Qwen3-VL-4B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def main() -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": "너는 이미지 해설사야. 이 이미지에 무엇이 있는지 한국어로 깔끔하게 요약 설명해줘."},
                ],
            }
        ],
        "max_tokens": 256,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        print(f"API 호출 실패: {exc}", file=sys.stderr)
        if getattr(exc, "response", None) is not None:
            print(exc.response.text, file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"응답 파싱 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=== 이미지 분석 결과 ===")
    print(answer)


if __name__ == "__main__":
    main()
