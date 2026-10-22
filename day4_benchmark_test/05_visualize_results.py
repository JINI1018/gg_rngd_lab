"""실제 CSV로 동시성 처리량·지연시간과 Prefix ON/OFF TTFT 그래프 3개를 만든다.

Windows PC: python 05_visualize_results.py
결과: results/*.png와 summary.csv. CSV가 없으면 가상 수치를 만들지 않고 종료한다.
"""

import matplotlib.pyplot as plt
import pandas as pd

from day04_common import RESULTS


def require_columns(frame, columns, filename):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(f"{filename}에 열이 없다: {sorted(missing)}")


def main():
    paths = [RESULTS / name for name in ("concurrency_runs.csv", "prefix_off.csv", "prefix_on.csv")]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"실제 측정 CSV가 없다: {path}")
    runs, off, on = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    require_columns(runs, ("concurrency", "mean_e2e_s", "mean_ttft_s", "output_tokens_per_s", "repeat"), paths[0].name)
    for frame, path in ((off, paths[1]), (on, paths[2])):
        require_columns(frame, ("mode", "phase", "ttft_client_s", "prefix_hits_delta"), path.name)
    if runs.empty or off.empty or on.empty:
        raise SystemExit("CSV에 실제 측정 행이 없다")
    if set(runs["concurrency"]) != {1, 2, 4, 8}:
        raise SystemExit("동시 요청 1·2·4·8 결과가 모두 필요하다")
    if min(runs.groupby("concurrency")["repeat"].nunique()) < 3:
        raise SystemExit("각 동시성 조건의 반복 측정이 3회 미만이다")

    summary = runs.groupby("concurrency", as_index=False).agg(
        throughput_mean=("output_tokens_per_s", "mean"),
        throughput_std=("output_tokens_per_s", "std"),
        latency_mean=("mean_e2e_s", "mean"),
        latency_std=("mean_e2e_s", "std"),
        ttft_mean=("mean_ttft_s", "mean"),
    )
    summary.to_csv(RESULTS / "summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(summary["concurrency"], summary["throughput_mean"],
                yerr=summary["throughput_std"].fillna(0), marker="o", capsize=4)
    ax.set(xlabel="Concurrent requests (fixed burst)", ylabel="Output tokens / s (server)",
           title="RNGD serving throughput")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "01_throughput_by_concurrency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(summary["concurrency"], summary["latency_mean"],
                yerr=summary["latency_std"].fillna(0), marker="o", capsize=4, color="#b45c36")
    ax.set(xlabel="Concurrent requests (fixed burst)", ylabel="Mean client E2E latency (s)",
           title="RNGD request latency")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "02_latency_by_concurrency.png", dpi=180)
    plt.close(fig)

    prefix = pd.concat((off, on), ignore_index=True)
    means = prefix.groupby(["mode", "phase"])["ttft_client_s"].mean()
    labels = ["OFF first", "OFF later", "ON first", "ON later"]
    values = [means.get((mode, phase), float("nan")) for mode, phase in
              (("off", "first"), ("off", "later"), ("on", "first"), ("on", "later"))]
    if any(pd.isna(value) for value in values):
        raise SystemExit("Prefix CSV의 first/later, off/on 조건을 확인한다")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#9ca3af", "#6b7280", "#93b6d5", "#346c99"])
    ax.set(ylabel="Client TTFT to first content (s)", title="Prefix caching: OFF vs ON")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "03_prefix_ttft_comparison.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("그래프 3개와 summary.csv 저장 완료. 원인 해석에는 Hit Counter 변화와 워밍업 상태를 함께 사용한다.")


if __name__ == "__main__":
    main()
