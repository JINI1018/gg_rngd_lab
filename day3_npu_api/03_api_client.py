"""Windows PC: 모델 목록을 확인하고 Qwen3 Chat Completion을 한 번 호출한다.

준비: .env, SSH -L 터널, requirements 설치. 실행: python 03_api_client.py
예상: 모델 ID와 질문·응답 출력. 연결/모델 오류는 README의 점검표를 따른다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    # 프로젝트 루트의 .env 로딩
    load_dotenv(Path(__file__).with_name(".env"))
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "furiosa-ai/Qwen3-8B-FP8")
    if not base_url.endswith("/v1") or not api_key:
        raise SystemExit(".env의 OPENAI_BASE_URL(/v1)과 OPENAI_API_KEY를 확인한다")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    models = client.models.list()
    available = [item.id for item in models.data]
    print("서버 모델 목록:", available)
    if model_id not in available:
        raise SystemExit(f"{model_id}이 모델 목록에 없다. 서버 실행 모델과 .env를 대조한다")

    question = "RNGD NPU의 역할을 한 문장으로 설명해줘."
    messages = [
        {"role": "system", "content": "짧고 정확한 한국어로 답하라."},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=96,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if not response.choices:
        raise SystemExit("응답 choice가 없다. 서버 로그를 확인한다")
    answer = response.choices[0].message.content or ""
    if not answer.strip():
        raise SystemExit("응답 본문이 비어 있다. 서버 로그와 thinking 설정을 확인한다")
    print("질문:", question)
    print("응답:", answer)


if __name__ == "__main__":
    main()
