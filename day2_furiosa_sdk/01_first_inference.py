"""Qwen3-8B-FP8로 첫 채팅 추론을 수행한다.

[실습 내용]
00_preflight.py로 환경 점검이 끝난 뒤, furiosa-llm SDK의 LLM 클래스를 이용해
조별로 배정된 NPU 장치에 Qwen3-8B-FP8 모델을 실제로 로딩하고 첫 채팅 추론을
수행해본다. 단일 사용자 메시지로 chat()을 호출해 응답을 받고, 응답 개수·내용이
비어 있지 않은지 검증한 뒤 출력한다. LLM 인스턴스를 with 블록으로 사용해
추론이 끝나면 NPU 자원(메모리/장치 점유)이 자동으로 반환되도록 하는 것이
이 실습의 핵심 패턴이다.
"""

from day02_config import settings

def main() -> None:
    # .env / 환경변수에서 모델 ID와 조별 NPU 장치 번호를 읽어온다
    model, device = settings()
    
    # furiosa-llm은 무거운 임포트라 환경 점검 이후, 실제 사용 시점에 지연 임포트
    from furiosa_llm import LLM, SamplingParams

    # 채팅 형식 입력: role/content 쌍의 리스트
    message = [{"role": "user",
                "content": "NPU가 LLM 추론에서 맡는 역할을 한 문장으로 설명해줘."}]

    # temperature=0, top_p=1로 결정적(항상 같은) 출력을 유도하고 최대 64토큰까지 생성
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
    print(f"모델 로딩: {model} / 장치: {device}")

    # with 블록 진입 시 지정한 NPU 장치에 모델을 로딩하고, 블록을 벗어나면 자원을 해제
    with LLM(model, devices=device) as llm:
        results = llm.chat(
            message,
            sampling_params=params,
            # Qwen3의 추론(reasoning) 과정을 끄고 바로 최종 답변만 생성
            chat_template_kwargs={"enable_thinking": False},
        )

        # 메시지 1개를 보냈으므로 결과도 1개, 그 안에 생성된 출력이 있어야 정상
        if len(results) != 1 or not results[0].outputs:
            raise RuntimeError("응답 개수가 예상과 다르다")

        answer = results[0].outputs[0].text.strip()

        if not answer:
            raise RuntimeError("생성된 텍스트가 비어 있다. 강사에게 로그를 보여준다")
        print("질문:", message[0]["content"])
        print("응답:", answer)

    # with 블록을 벗어난 시점 = LLM.__exit__ 호출 = NPU 장치/메모리 반환 완료
    print("모델 종료: with 블록을 벗어나 NPU 자원을 반환했다")

if __name__ == "__main__":
    main()
