"""Windows PC: 클라이언트가 대화 이력을 관리하는 Multi-turn Streaming 챗봇.

준비: 04_streaming_chat.py 성공. 실행: python 05_multiturn_chat.py
예상: 앞선 질문을 반영한 후속 답변. /reset은 이력 초기화, /exit는 종료.
오류 또는 문맥 초과 시 README의 점검표와 서버 로그를 확인한다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 프로젝트 루트의 .env 로딩
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(override=True, dotenv_path=BASE_DIR / ".env")

def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "furiosa-ai/Qwen3-8B-FP8")
    if not base_url.endswith("/v1") or not api_key:
        raise SystemExit(".env의 API 주소와 키 자리표시자를 확인한다")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    system = {"role": "system", "content": "짧고 정확한 한국어로 답하라."}
    history = [system]
    print("Multi-turn Streaming 챗봇. 초기화: /reset, 종료: /exit")

    while True:
        question = input("나> ").strip()
        command = question.lower()
        if command == "/exit":
            break
        if command == "/reset":
            history = [system]
            print("대화 이력을 초기화했다.")
            continue
        if not question:
            continue
        current_user = {"role": "user", "content": question}
        request_messages = history + [current_user]
        stream = client.chat.completions.create(
            model=model_id,
            messages=request_messages,
            stream=True,
            max_tokens=128,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        parts = []
        print("봇> ", end="", flush=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if piece:
                parts.append(piece)
                print(piece, end="", flush=True)
        print()
        answer = "".join(parts).strip()
        if not answer:
            print("응답 본문이 비어 있다. 이 턴은 이력에 저장하지 않는다.")
            continue
        history.extend([current_user, {"role": "assistant", "content": answer}])
        print(f"대화 턴: {(len(history) - 1) // 2}")


if __name__ == "__main__":
    main()
