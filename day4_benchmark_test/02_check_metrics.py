"""Windows PC: SSH 터널의 GET /metrics에서 실제 노출 이름과 값을 확인한다.

실행: python 02_check_metrics.py. 정상: Counter/Gauge/Histogram 샘플이 출력된다.
없음(None)인 항목은 임의 값으로 채우지 말고 서버 버전·/metrics를 대조한다.
"""

from day04_common import METRICS, config, metrics_snapshot


def main():
    base_url, _, _ = config()
    print("GET", base_url[:-3] + "/metrics")
    snapshot = metrics_snapshot(base_url)
    for name in METRICS:
        print(f"{name}: {snapshot[name]}")
    if snapshot["furiosa_llm_generation_tokens_total"] is None:
        raise SystemExit("생성 토큰 Counter가 없다. 이 환경에서 Output TPS를 계산할 수 없다")
    print("메트릭 확인 완료. Counter는 구간 차이, Gauge는 관찰 시점의 값으로 해석한다.")


if __name__ == "__main__":
    main()
