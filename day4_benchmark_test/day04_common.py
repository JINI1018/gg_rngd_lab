"""DAY 4 실습에서 공통으로 사용하는 설정·메트릭·Streaming 측정 함수."""

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI


PRACTICE = Path(__file__).resolve().parent
DATA = PRACTICE / "data" / "benchmark_prompts.json"
RESULTS = PRACTICE / "results"
METRICS = (
    "furiosa_llm_generation_tokens_total",
    "furiosa_llm_prefix_cache_hits",
    "furiosa_llm_prefix_cache_queries",
    "furiosa_llm_kv_cache_usage_percent",
    "furiosa_llm_time_to_first_token_seconds_sum",
    "furiosa_llm_time_to_first_token_seconds_count",
    "furiosa_llm_inter_token_latency_seconds_sum",
    "furiosa_llm_inter_token_latency_seconds_count",
    "furiosa_llm_e2e_request_latency_seconds_sum",
    "furiosa_llm_e2e_request_latency_seconds_count",
)


def config():
    load_dotenv(PRACTICE / ".env")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "furiosa-ai/Qwen3-8B-FP8")
    if not base_url.endswith("/v1") or not api_key:
        raise SystemExit(".env의 OPENAI_BASE_URL(/v1)과 OPENAI_API_KEY를 확인한다")
    return base_url, api_key, model_id


def client_and_model():
    base_url, api_key, model_id = config()
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)
    listed = [item.id for item in client.models.list().data]
    if model_id not in listed:
        raise SystemExit(f"서버 모델 목록 {listed}에 {model_id}이 없다")
    return client, model_id


def metrics_url(base_url):
    return base_url[:-3] + "/metrics"


def metrics_snapshot(base_url):
    """Prometheus 샘플을 이름별로 합산한다. 조별 서버 한 개만 사용해야 한다."""
    with urlopen(metrics_url(base_url), timeout=10) as response:
        payload = response.read().decode("utf-8")
    values = {name: [] for name in METRICS}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([+\-0-9.eE]+)", line)
        if not match:
            continue
        name, raw_value = match.groups()
        # Prometheus Counter는 실제 노출 이름에 _total이 붙는 구현도 있다.
        canonical = name[:-6] if name.endswith("_total") and name[:-6] in values else name
        if canonical in values:
            values[canonical].append(float(raw_value))
    snapshot = {}
    for name, samples in values.items():
        # 여러 엔진의 Counter/Histogram은 합산한다. KV 사용률 Gauge는 최대를 본다.
        snapshot[name] = (max(samples) if name.endswith("kv_cache_usage_percent") else sum(samples)) if samples else None
    return snapshot


def metric_delta(before, after, name):
    a, b = before.get(name), after.get(name)
    if a is None or b is None:
        return None
    difference = b - a
    if difference < 0:
        raise RuntimeError(f"{name} 카운터가 감소했다. 측정 중 서버 재시작 여부를 확인한다")
    return difference


def measure_stream(client, model_id, messages, max_tokens=64):
    """요청 시작부터 첫 비어 있지 않은 content 및 종료까지 클라이언트 시간 측정."""
    wall_start = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    first_event = None
    first_content = None
    chunks = []
    usage_tokens = None
    for chunk in stream:
        now = time.perf_counter()
        if first_event is None:
            first_event = now
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            usage_tokens = getattr(usage, "completion_tokens", usage_tokens)
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        if piece:
            if first_content is None:
                first_content = now
            chunks.append(piece)
    end = time.perf_counter()
    if first_content is None:
        raise RuntimeError("Streaming에서 실제 content가 없다. 서버 로그와 thinking 설정을 확인한다")
    return {
        "started_utc": wall_start,
        "first_event_s": round(first_event - start, 4),
        "ttft_client_s": round(first_content - start, 4),
        "e2e_client_s": round(end - start, 4),
        "usage_completion_tokens": usage_tokens,
        "answer_chars": len("".join(chunks)),
    }
