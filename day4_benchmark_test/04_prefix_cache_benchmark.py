"""Prefix Caching OFF/ON 서버에서 동일한 공통 문맥 요청을 순차 측정한다.

Windows PC에서 실행: python 04_prefix_cache_benchmark.py --mode off
서버를 ON 설정으로 재시작한 뒤: python 04_prefix_cache_benchmark.py --mode on
결과: results/prefix_off.csv, prefix_on.csv. 모드는 실제 서버 시작 옵션을 확인한 뒤 지정한다.
"""

import argparse
import csv
import json

from day04_common import DATA, RESULTS, client_and_model, config, measure_stream, metric_delta, metrics_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    if args.repeats < 3 or args.max_tokens < 1:
        parser.error("--repeats는 3 이상, --max-tokens는 1 이상이어야 한다")
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    shared_system = dataset["prefix_system"]
    questions = dataset["prefix_questions"]
    if not shared_system or not questions:
        raise SystemExit("공통 문맥과 질문이 필요하다")
    client, model_id = client_and_model()
    base_url, _, _ = config()
    initial = metrics_snapshot(base_url)
    if initial["furiosa_llm_generation_tokens_total"] is None:
        raise SystemExit("생성 토큰 Counter가 없어 실제 Output TPS를 계산할 수 없다")
    if args.mode == "on" and initial["furiosa_llm_prefix_cache_hits"] is None:
        raise SystemExit("ON 서버의 Prefix Cache Hit Counter가 없어 캐시 재사용을 검증할 수 없다")

    # 다른 시작 문맥의 워밍업만 수행해 공통 접두부를 미리 캐시하지 않는다.
    measure_stream(client, model_id,
                   [{"role": "user", "content": "이 문장은 별도의 워밍업이며 공통 참고 문맥과 다르다. 짧게 답해줘."}],
                   args.max_tokens)
    rows = []
    for repeat in range(1, args.repeats + 1):
        for index, question in enumerate(questions):
            messages = [
                {"role": "system", "content": shared_system},
                {"role": "user", "content": question["prompt"]},
            ]
            before = metrics_snapshot(base_url)
            result = measure_stream(client, model_id, messages, args.max_tokens)
            after = metrics_snapshot(base_url)
            tokens = metric_delta(before, after, "furiosa_llm_generation_tokens_total")
            if tokens is None or tokens <= 0:
                raise RuntimeError("생성 토큰 Counter 변화량을 확인하지 못했다")
            hits = metric_delta(before, after, "furiosa_llm_prefix_cache_hits")
            queries = metric_delta(before, after, "furiosa_llm_prefix_cache_queries")
            row = {
                "mode": args.mode, "repeat": repeat, "prompt_id": question["id"],
                "phase": "first" if repeat == 1 and index == 0 else "later",
                "model_id": model_id, "max_tokens": args.max_tokens,
                "temperature": 0.0, "top_p": 1.0,
                "shared_system_chars": len(shared_system),
                **result,
                "output_tokens_counter_delta": round(tokens),
                "output_tokens_per_s": round(tokens / max(result["e2e_client_s"], 0.0001), 4),
                "prefix_hits_delta": hits, "prefix_queries_delta": queries,
                "kv_cache_usage_after": after["furiosa_llm_kv_cache_usage_percent"],
            }
            rows.append(row)
            print(f"{args.mode.upper()} {repeat}-{question['id']}: TTFT {row['ttft_client_s']:.3f}s, "
                  f"E2E {row['e2e_client_s']:.3f}s, Hit {hits}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"prefix_{args.mode}.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장 완료: {output} / {len(rows)}행")
    print("이 CSV의 모드는 명령행 표기다. 실제 서버 로그와 시작 옵션을 별도로 기록해야 한다.")


if __name__ == "__main__":
    main()
