"""고정 크기 버스트 1·2·4·8 요청을 각 3회 보내 동시성 성능을 측정한다.

Windows PC에서 DAY 3 SSH 터널과 서버가 켜진 상태로 실행한다.
실행: python 03_concurrency_benchmark.py
결과: results/concurrency_runs.csv, concurrency_requests.csv
오류·메트릭 누락·해석 방법은 README를 참조한다. 실제 장비 측정값만 저장한다.
"""

import argparse
import csv
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

from day04_common import DATA, RESULTS, client_and_model, config, measure_stream, metric_delta, metrics_snapshot


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile_95(values):
    """소표본 교육용 최근접 순위 P95. 표본 수가 적으므로 과해석하지 않는다."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.repeats < 3 or args.max_tokens < 1:
        parser.error("--repeats는 3 이상, --max-tokens는 1 이상이어야 한다")

    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    prompts = dataset["concurrency_prompts"]
    if len(prompts) < 8:
        raise SystemExit("동시성 테스트 질문이 8개 이상 필요하다")
    client, model_id = client_and_model()
    base_url, _, _ = config()
    initial_metrics = metrics_snapshot(base_url)
    if initial_metrics["furiosa_llm_generation_tokens_total"] is None:
        raise SystemExit("생성 토큰 Counter가 없어 실제 Output TPS를 계산할 수 없다")

    # 워밍업은 본 측정 전에 한 번만 수행하고 CSV에 기록하지 않는다.
    measure_stream(client, model_id, [{"role": "user", "content": "별도의 서버 워밍업 질문에 짧게 답해줘."}], args.max_tokens)
    print("워밍업 완료. 본 측정은 고정 크기 버스트 방식이며 지속 부하 테스트가 아니다.")
    run_rows = []
    request_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for concurrency in (1, 2, 4, 8):
        for repeat in range(1, args.repeats + 1):
            run_id = f"C{concurrency}-R{repeat}-{uuid.uuid4().hex[:8]}"
            # 식별자를 입력의 시작에 넣어 요청 간 공통 접두부를 최소화한다.
            tasks = [
                (row["id"], [{"role": "user", "content": f"{uuid.uuid4().hex} {row['prompt']}"}])
                for row in prompts[:concurrency]
            ]
            before = metrics_snapshot(base_url)
            barrier = Barrier(concurrency + 1)

            def worker(messages):
                barrier.wait(timeout=30)
                return measure_stream(client, model_id, messages, args.max_tokens)

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(worker, messages) for _, messages in tasks]
                run_started_utc = datetime.now(timezone.utc).isoformat()
                window_start = time.perf_counter()
                barrier.wait(timeout=30)
                measurements = [future.result() for future in futures]
                window_s = time.perf_counter() - window_start
            after = metrics_snapshot(base_url)
            output_tokens = metric_delta(before, after, "furiosa_llm_generation_tokens_total")
            if output_tokens is None or output_tokens <= 0:
                raise RuntimeError("생성 토큰 Counter 변화량이 없다. 다른 요청·메트릭·서버를 확인한다")
            e2e = [item["e2e_client_s"] for item in measurements]
            ttft = [item["ttft_client_s"] for item in measurements]
            hits = metric_delta(before, after, "furiosa_llm_prefix_cache_hits")
            queries = metric_delta(before, after, "furiosa_llm_prefix_cache_queries")
            run_rows.append({
                "run_id": run_id, "started_utc": run_started_utc,
                "model_id": model_id, "concurrency": concurrency, "repeat": repeat,
                "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0,
                "window_s": round(window_s, 4),
                "mean_e2e_s": round(sum(e2e) / len(e2e), 4),
                "p95_e2e_s": round(percentile_95(e2e), 4),
                "mean_ttft_s": round(sum(ttft) / len(ttft), 4),
                "output_tokens_total": round(output_tokens),
                "output_tokens_per_s": round(output_tokens / window_s, 4),
                "requests_per_s": round(concurrency / window_s, 4),
                "prefix_hits_delta": hits, "prefix_queries_delta": queries,
            })
            for (prompt_id, _), result in zip(tasks, measurements):
                request_rows.append({"run_id": run_id, "concurrency": concurrency,
                                     "repeat": repeat, "prompt_id": prompt_id, **result})
            print(f"동시 {concurrency} / 반복 {repeat}: 평균 E2E {run_rows[-1]['mean_e2e_s']:.3f}s, "
                  f"전체 Output TPS {run_rows[-1]['output_tokens_per_s']:.2f}")

    run_path = args.output_dir / "concurrency_runs.csv"
    request_path = args.output_dir / "concurrency_requests.csv"
    save_csv(run_path, run_rows, list(run_rows[0]))
    save_csv(request_path, request_rows, list(request_rows[0]))
    print(f"저장 완료: {run_path} ({len(run_rows)}행), {request_path} ({len(request_rows)}행)")
    print("Usage가 없는 요청의 토큰 수는 비워 두고 조별 서버 Counter 차이로 전체 TPS를 계산했다.")


if __name__ == "__main__":
    main()
