"""Windows PC: 사용자 질문마다 Streaming 결과를 순차 출력하는 콘솔 챗봇.

준비: 03_api_client.py 성공. 실행: python 04_streaming_chat.py
예상: 질문 후 답변이 여러 조각으로 표시된다. /exit로 종료한다.
이 단계는 요청마다 이전 대화를 보내지 않는 Single-turn 방식이다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "furiosa-ai/Qwen3-8B-FP8")
    if not base_url.endswith("/v1") or not api_key:
        raise SystemExit(".env의 API 주소와 키 자리표시자를 확인한다")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    print("Single-turn Streaming 챗봇. 종료: /exit")

    while True:
        question = input("나> ").strip()
        if question.lower() == "/exit":
            break
        if not question:
            continue
        messages = [
            {"role": "system", "content": "짧고 정확한 한국어로 답하라."},
            {"role": "user", "content": question},
        ]
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            max_tokens=128,
            temperature=0,
            # OpenAI SDK 표준 파라미터가 아니므로 extra_body로 전달해야 서버까지 도달한다.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        parts = []
        print("봇> ", end="", flush=True)
        for chunk in stream:
            # 일부 chunk는 choices가 빈 keep-alive성 조각이라 건너뛴다.
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if piece:
                parts.append(piece)
                print(piece, end="", flush=True)
        print()
        if not "".join(parts).strip():
            print("응답 본문이 비어 있다. 서버 로그를 확인한다.")


if __name__ == "__main__":
    main()
