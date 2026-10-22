"""DAY 4 벤치마크 결과(results/*.csv)를 Streamlit 대시보드로 보여준다.

실행: uv run streamlit run 06_dashboard.py
입력: results/concurrency_runs.csv, concurrency_requests.csv, prefix_off.csv, prefix_on.csv
CSV가 없으면 해당 섹션에 경고만 표시하고 가상 숫자를 만들지 않는다.
03_concurrency_benchmark.py / 04_prefix_cache_benchmark.py를 다시 실행해 CSV를 갱신한 뒤
브라우저를 새로고침하면(또는 사이드바의 새로고침 버튼) 최신 결과가 반영된다.
"""

import altair as alt
import pandas as pd
import streamlit as st

from day04_common import RESULTS

CONCURRENCY_RUNS = RESULTS / "concurrency_runs.csv"
CONCURRENCY_REQUESTS = RESULTS / "concurrency_requests.csv"
PREFIX_OFF = RESULTS / "prefix_off.csv"
PREFIX_ON = RESULTS / "prefix_on.csv"

st.set_page_config(page_title="RNGD DAY4 성능 대시보드", layout="wide")


def load_csv(path):
    if not path.is_file():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame if not frame.empty else None


def error_band(frame, x, mean_col, std_col):
    band = frame.copy()
    std = band[std_col].fillna(0)
    band["_lo"] = band[mean_col] - std
    band["_hi"] = band[mean_col] + std
    line = alt.Chart(band).mark_line(point=True).encode(
        x=alt.X(f"{x}:O", title="동시 요청 수"),
        y=alt.Y(f"{mean_col}:Q", title=None),
        tooltip=list(band.columns.drop(["_lo", "_hi"])),
    )
    rule = alt.Chart(band).mark_rule().encode(
        x=alt.X(f"{x}:O"), y="_lo:Q", y2="_hi:Q",
    )
    return (rule + line).properties(height=320)


st.title("RNGD · Qwen3-8B-FP8 서빙 성능 대시보드")
st.caption("results/ 폴더의 실제 측정 CSV만 사용한다. 숫자가 비어 있으면 벤치마크를 먼저 실행한다.")

LOG_FILES = {
    CONCURRENCY_RUNS.name: CONCURRENCY_RUNS,
    CONCURRENCY_REQUESTS.name: CONCURRENCY_REQUESTS,
    PREFIX_OFF.name: PREFIX_OFF,
    PREFIX_ON.name: PREFIX_ON,
}

with st.sidebar:
    if st.button("새로고침"):
        st.rerun()

    st.header("로그 파일 보기")
    available_logs = {name: path for name, path in LOG_FILES.items() if path.is_file()}
    if available_logs:
        selected_log_name = st.radio("CSV 파일 선택", list(available_logs.keys()))
        selected_log_path = available_logs[selected_log_name]
    else:
        selected_log_name, selected_log_path = None, None
        st.caption("아직 생성된 CSV가 없다.")
    st.caption("CSV를 갱신하려면: `uv run python 03_concurrency_benchmark.py`, "
               "`uv run python 04_prefix_cache_benchmark.py --mode off|on`")

runs = load_csv(CONCURRENCY_RUNS)
requests_df = load_csv(CONCURRENCY_REQUESTS)
off = load_csv(PREFIX_OFF)
on = load_csv(PREFIX_ON)

model_id = next(
    (frame["model_id"].iloc[0] for frame in (runs, off, on) if frame is not None and "model_id" in frame.columns),
    None,
)
latest_times = [
    frame["started_utc"].max() for frame in (runs, off, on)
    if frame is not None and "started_utc" in frame.columns
]

meta = st.columns(4)
meta[0].metric("모델", model_id or "—")
meta[1].metric("최근 측정 시각(UTC)", max(latest_times)[:19] if latest_times else "—")
meta[2].metric("동시성 실행 수", len(runs) if runs is not None else 0)
meta[3].metric("Prefix OFF / ON 행 수", f"{len(off) if off is not None else 0} / {len(on) if on is not None else 0}")

st.divider()

# ------------------------------------------------------------- 선택한 로그 파일
st.header("선택한 로그 파일")
if selected_log_path is None:
    st.info("사이드바에 표시할 CSV가 없다. `03`·`04` 스크립트를 먼저 실행한다.")
else:
    selected_frame = load_csv(selected_log_path)
    st.subheader(selected_log_name)
    if selected_frame is None:
        st.warning("파일이 비어 있다.")
    else:
        st.caption(f"{len(selected_frame)}행 · {selected_log_path}")
        st.dataframe(selected_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "CSV 다운로드",
            data=selected_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=selected_log_name,
            mime="text/csv",
        )

st.divider()

# ---------------------------------------------------------------- 동시성 실험
st.header("1. 동시성 실험 — 동시 요청 1 · 2 · 4 · 8")

if runs is None:
    st.warning(f"`{CONCURRENCY_RUNS.name}`가 없다. `uv run python 03_concurrency_benchmark.py`를 먼저 실행한다.")
else:
    missing = {1, 2, 4, 8} - set(runs["concurrency"])
    if missing:
        st.warning(f"동시성 조건 {sorted(missing)}의 결과가 아직 없다. 지금 보이는 요약은 일부 조건 기준이다.")

    summary = runs.groupby("concurrency", as_index=False).agg(
        throughput_mean=("output_tokens_per_s", "mean"),
        throughput_std=("output_tokens_per_s", "std"),
        latency_mean=("mean_e2e_s", "mean"),
        latency_std=("mean_e2e_s", "std"),
        ttft_mean=("mean_ttft_s", "mean"),
    ).sort_values("concurrency")

    cols = st.columns(len(summary))
    for col, row in zip(cols, summary.itertuples()):
        col.metric(
            f"동시 {int(row.concurrency)}",
            f"{row.throughput_mean:.1f} tok/s",
            help=f"평균 E2E {row.latency_mean:.3f}s · 평균 TTFT {row.ttft_mean:.3f}s",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("처리량 (Output tokens / s, 서버 Counter 기준)")
        st.altair_chart(error_band(summary, "concurrency", "throughput_mean", "throughput_std"),
                         use_container_width=True)
    with right:
        st.subheader("평균 E2E 지연시간 (s)")
        st.altair_chart(error_band(summary, "concurrency", "latency_mean", "latency_std"),
                         use_container_width=True)

    st.caption("오차 막대는 동시성 조건별 3회 반복 간 표준편차다. 표본이 적어 참고용으로만 본다.")

    with st.expander("요청별 상세 (concurrency_requests.csv)"):
        if requests_df is None:
            st.info(f"`{CONCURRENCY_REQUESTS.name}`가 없다.")
        else:
            scatter = alt.Chart(requests_df).mark_circle(size=70, opacity=0.6).encode(
                x=alt.X("concurrency:O", title="동시 요청 수"),
                y=alt.Y("e2e_client_s:Q", title="요청별 E2E (s)"),
                color=alt.Color("repeat:N", title="반복 회차"),
                tooltip=["run_id", "prompt_id", "concurrency", "repeat", "ttft_client_s", "e2e_client_s"],
            ).properties(height=320)
            st.altair_chart(scatter, use_container_width=True)
            st.dataframe(requests_df, use_container_width=True, hide_index=True)

    with st.expander("조건별 요약 (concurrency_runs.csv)"):
        st.dataframe(runs, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------ Prefix caching
st.header("2. Prefix Caching OFF vs ON")

if off is None or on is None:
    missing_name = PREFIX_OFF.name if off is None else PREFIX_ON.name
    st.warning(f"`{missing_name}`가 없다. `uv run python 04_prefix_cache_benchmark.py --mode off|on`을 먼저 실행한다.")
else:
    def phase_mean(frame, phase, column):
        subset = frame.loc[frame["phase"] == phase, column]
        return float(subset.mean()) if not subset.empty else float("nan")

    off_first, off_later = phase_mean(off, "first", "ttft_client_s"), phase_mean(off, "later", "ttft_client_s")
    on_first, on_later = phase_mean(on, "first", "ttft_client_s"), phase_mean(on, "later", "ttft_client_s")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("OFF · first TTFT", f"{off_first:.3f}s")
    k2.metric("OFF · later TTFT", f"{off_later:.3f}s", delta=f"{off_later - off_first:+.3f}s", delta_color="inverse")
    k3.metric("ON · first TTFT", f"{on_first:.3f}s")
    k4.metric("ON · later TTFT", f"{on_later:.3f}s", delta=f"{on_later - on_first:+.3f}s", delta_color="inverse")

    on_hits = on["prefix_hits_delta"].sum()
    on_queries = on["prefix_queries_delta"].sum()
    off_hits = off["prefix_hits_delta"].sum()
    if pd.notna(on_hits) and on_hits > 0:
        st.success(f"ON 서버에서 Prefix Cache Hit {on_hits:.0f} / Query {on_queries:.0f} — 캐시 재사용이 확인된다.")
    else:
        st.warning("ON 서버인데 Hit Counter 증가가 없다. System Prompt 토큰 일치 여부와 실제 서버 옵션을 다시 확인한다.")
    if pd.notna(off_hits) and off_hits > 0:
        st.info(f"OFF 서버에서도 Hit {off_hits:.0f}이 기록됐다. `--no-enable-prefix-caching`이 실제로 적용됐는지 서버 로그를 확인한다.")

    left, right = st.columns(2)
    with left:
        st.subheader("TTFT: first vs later, OFF vs ON")
        bar_data = pd.DataFrame({
            "mode": ["OFF", "OFF", "ON", "ON"],
            "phase": ["first", "later", "first", "later"],
            "ttft_client_s": [off_first, off_later, on_first, on_later],
        })
        bars = alt.Chart(bar_data).mark_bar().encode(
            x=alt.X("phase:N", title=None, sort=["first", "later"]),
            xOffset="mode:N",
            y=alt.Y("ttft_client_s:Q", title="Client TTFT (s)"),
            color=alt.Color("mode:N", title="모드", scale=alt.Scale(domain=["OFF", "ON"], range=["#9ca3af", "#346c99"])),
            tooltip=["mode", "phase", "ttft_client_s"],
        ).properties(height=320)
        st.altair_chart(bars, use_container_width=True)

    with right:
        st.subheader("요청 순서별 TTFT 추이")
        combined = pd.concat(
            [off.assign(mode_label="OFF"), on.assign(mode_label="ON")], ignore_index=True
        )
        combined["request_seq"] = combined.groupby("mode_label").cumcount() + 1
        trend = alt.Chart(combined).mark_line(point=True).encode(
            x=alt.X("request_seq:Q", title="요청 순서 (서버 재시작 후)"),
            y=alt.Y("ttft_client_s:Q", title="Client TTFT (s)"),
            color=alt.Color("mode_label:N", title="모드", scale=alt.Scale(domain=["OFF", "ON"], range=["#9ca3af", "#346c99"])),
            tooltip=["mode_label", "repeat", "phase", "ttft_client_s"],
        ).properties(height=320)
        st.altair_chart(trend, use_container_width=True)

    st.subheader("KV 캐시 사용률 (요청 직후, Gauge)")
    kv_chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X("request_seq:Q", title="요청 순서"),
        y=alt.Y("kv_cache_usage_after:Q", title="KV 사용률 (0~1)"),
        color=alt.Color("mode_label:N", title="모드", scale=alt.Scale(domain=["OFF", "ON"], range=["#9ca3af", "#346c99"])),
        tooltip=["mode_label", "repeat", "phase", "kv_cache_usage_after"],
    ).properties(height=260)
    st.altair_chart(kv_chart, use_container_width=True)

    with st.expander("원본 데이터 (prefix_off.csv / prefix_on.csv)"):
        tab_off, tab_on = st.tabs(["OFF", "ON"])
        tab_off.dataframe(off, use_container_width=True, hide_index=True)
        tab_on.dataframe(on, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "데이터 출처: day4_benchmark_test/results/*.csv (03·04 스크립트가 실제 서버에서 측정한 값). "
    "OFF→ON 순서로 측정했기 때문에 시간 경과 효과가 섞일 수 있어 결론을 과장하지 않는다."
)
