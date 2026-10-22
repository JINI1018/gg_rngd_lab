# https://huggingface.co/furiosa-ai/Qwen3-32B-FP8
# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 
# https://huggingface.co/furiosa-ai/Qwen3-VL-4B-Instruct
# furiosa-llm serve furiosa-ai/Qwen3-VL-4B-Instruct
# watch -n 1 furiosa-smi status


import requests

# NPU 추론 서버 설정
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "furiosa-ai/Qwen3-8B-FP8 "
# MODEL = "furiosa-ai/Qwen3-VL-4B-Instruct"


def chat(prompt: str) -> str:
    """NPU 추론 서버에 요청하고 모델 응답을 반환한다."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 2048,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    try:
        answer = chat("NPU와 GPU의 차이를 설명해줘.")
        print("\n=== Qwen3-VL 응답 ===")
        print(answer)

    except requests.RequestException as e:
        print(f"API 요청 오류: {e}")