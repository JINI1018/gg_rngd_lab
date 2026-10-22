"""JSONL 질문을 여러 개씩 LLM.chat에 전달하고 결과를 저장한다.

[실습 내용]
01_first_inference.py가 질문 1개짜리 단발 추론이었다면, 이 실습은 JSONL 파일에
담긴 여러 질문(id + prompt)을 읽어 --batch-size 개씩 묶어 llm.chat()에 한 번에
전달하는 배치 추론을 다룬다. 입력 파일을 검증(JSON 문법, 필수 필드, id 중복)
하는 read_prompts(), NPU에 모델을 한 번만 로딩한 뒤 여러 배치를 순회하며
추론하는 main()의 루프, 그리고 결과를 JSONL로 저장하는 부분이 핵심이다.
배치 처리는 모델을 매번 새로 로딩하지 않고 재사용해 여러 질문을 효율적으로
처리하는 패턴을 보여준다.
"""

import argparse
import json
from pathlib import Path

from day02_config import settings

def read_prompts(path: Path) -> list[dict[str, str]]:
    """JSONL 파일을 한 줄씩 읽어 {id, prompt} 레코드 목록으로 검증·변환한다."""

    records: list[dict[str, str]] = []
    seen: set[str] = set()  # id 중복 검사를 위한 집합

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue  # 빈 줄은 건너뜀
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{line_no}행: JSON 문법 오류: {exc.msg}") from exc

        if not isinstance(row, dict):
            raise ValueError(f"{line_no}행: JSON 객체가 필요하다")

        prompt_id, prompt = row.get("id"), row.get("prompt")

        # id와 prompt가 비어 있지 않은 문자열인지 검증
        if not isinstance(prompt_id, str) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{line_no}행: id와 비어 있지 않은 prompt 문자열이 필요하다")

        if prompt_id in seen:
            raise ValueError(f"{line_no}행: 중복 id {prompt_id}")

        seen.add(prompt_id)
        records.append({"id": prompt_id, "prompt": prompt})

    if not records:
        raise ValueError("질문 파일이 비어 있다")

    return records


def main() -> None:
    # 커맨드라인 인자 파서. description에 모듈 docstring을 그대로 사용(--help 출력용)
    parser = argparse.ArgumentParser(description=__doc__)
    # 질문 JSONL 경로. 생략 시 이 스크립트와 같은 디렉터리의 prompts.jsonl 사용
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("prompts.jsonl"))
    # 결과를 저장할 JSONL 경로. 생략 시 실행 위치 기준 batch_results.jsonl에 저장
    parser.add_argument("--output", type=Path, default=Path("output_/batch_results.jsonl"))
    # 한 번의 llm.chat() 호출에 몇 개의 질문을 묶어 보낼지
    parser.add_argument("--batch-size", type=int, default=3)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 한다")

    if args.input.resolve() == args.output.resolve():
        parser.error("입력 파일과 출력 파일은 달라야 한다")

    records = read_prompts(args.input)
    model, device = settings()

    # furiosa-llm은 무거운 임포트라 입력 검증이 끝난 뒤 지연 임포트
    from furiosa_llm import LLM, SamplingParams

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=72)
    saved: list[dict[str, str]] = []

    print(f"모델 로딩: {model} / 장치: {device} / 질문: {len(records)}개")

    # 모델은 with 블록에서 단 한 번만 로딩하고, 그 안에서 배치 단위로 반복 추론
    with LLM(model, devices=device) as llm:
        for start in range(0, len(records), args.batch_size):
            # records를 --batch-size 크기로 잘라 한 번에 llm.chat에 전달
            chunk = records[start : start + args.batch_size]
            conversations = [[{"role": "user", "content": row["prompt"]}] for row in chunk]
            outputs = llm.chat(
                conversations,
                sampling_params=params,
                # Qwen3의 추론(reasoning) 과정을 끄고 바로 최종 답변만 생성
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(chunk):
                raise RuntimeError("질문과 응답 개수가 다르다")
            # 입력 순서와 출력 순서가 1:1로 대응한다는 전제로 zip하여 매핑
            for row, result in zip(chunk, outputs):
                if not result.outputs:
                    raise RuntimeError(f"{row['id']}의 출력이 없다")
                answer = result.outputs[0].text.strip()
                if not answer:
                    raise RuntimeError(f"{row['id']}의 생성 텍스트가 비어 있다")
                saved.append({"id": row["id"], "prompt": row["prompt"], "answer": answer})
                print(f"{row['id']}: {answer[:200]}")

    # 결과를 한 줄에 하나씩 JSON으로 직렬화하여 JSONL 형식으로 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in saved) + "\n",
        encoding="utf-8",
    )

    print(f"저장 완료: {args.output} / {len(saved)}건")

if __name__ == "__main__":
    main()
