"""프롬프트와 SamplingParams를 한 번에 한 조건씩 바꾸어 비교한다.

[실습 내용]
동일한 모델(Qwen3-8B-FP8)을 한 번만 로딩한 상태에서, 
(1) 기준 조건, (2) 프롬프트만 바꾼 조건, (3) 샘플링 파라미터(temperature/top_p)만 
바꾼조건, 이렇게 3가지 케이스를 차례로 돌려 "무엇을 바꾸면 응답이 어떻게
달라지는지"를 한 번에 하나의 변수만 바꿔 비교(A/B 테스트)해보는 실습이다.
temperature=0인 조건은 결정적 출력을, temperature=0.8인 조건은 더 다양하고
창의적인 출력을 유도한다. 각 조건의 조건명/파라미터/프롬프트/응답을
모아 CSV로 저장해 조건별 결과를 표로 비교할 수 있게 한다.
"""

import argparse
import csv
from pathlib import Path

from day02_config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 비교 결과를 저장할 CSV 경로 (기본: output_/sampling_results.csv)
    parser.add_argument("--output", type=Path, default=Path("output_/sampling_results.csv"))
    args = parser.parse_args()
    model, device = settings()

    # 파일 맨 위가 아니라 main() 함수 안, 그것도 argparse로 인자를 다 파싱한 뒤에 import하는 이유
    # 쓸 일이 확정된 다음에야 무거운 걸 불러온다
    # furiosa-llm은 무거운 임포트라 인자 파싱 이후 지연 임포트
    from furiosa_llm import LLM, SamplingParams

    base_prompt = "RNGD NPU가 LLM 추론에 쓰이는 이유를 설명해줘."
    # (조건명, 프롬프트, temperature, top_p) 튜플 목록.
    # 기준 대비 "프롬프트 변경"은 프롬프트만, "샘플링 변경"은 temperature/top_p만 다르다
    cases = [
        ("기준", base_prompt, 0.0, 1.0),
        ("프롬프트 변경", "초보자를 위해 RNGD NPU의 역할을 두 문장으로 설명해줘.", 0.0, 1.0),
        ("샘플링 변경", base_prompt, 0.8, 0.9),
    ]
    rows: list[dict[str, str | float]] = []
    print(f"모델 로딩: {model} / 장치: {device}")
    # 모델은 한 번만 로딩하고, with 블록 안에서 3가지 조건을 순서대로 추론
    with LLM(model, devices=device) as llm:
        for label, prompt, temperature, top_p in cases:
            params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=72)
            result = llm.chat(
                [{"role": "user", "content": prompt}],
                sampling_params=params,
                # Qwen3의 추론(reasoning) 과정을 끄고 바로 최종 답변만 생성
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(result) != 1 or not result[0].outputs:
                raise RuntimeError(f"{label} 조건의 응답이 없다")
            answer = result[0].outputs[0].text.strip()
            if not answer:
                raise RuntimeError(f"{label} 조건의 생성 텍스트가 비어 있다")
            # 조건별 설정과 응답을 한 행으로 모아 나중에 CSV로 저장
            rows.append({
                "condition": label, "temperature": temperature,
                "top_p": top_p, "prompt": prompt, "answer": answer,
            })
            print(f"\n[{label}] temperature={temperature}, top_p={top_p}")
            print("응답:", answer)
            
    # 3개 조건의 결과를 CSV 한 파일에 모아 저장 (엑셀 호환을 위해 utf-8-sig 사용)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("condition", "temperature", "top_p", "prompt", "answer"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장 완료: {output} / {len(rows)}조건")


if __name__ == "__main__":
    main()