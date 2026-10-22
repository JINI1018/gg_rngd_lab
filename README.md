# RNGD LLM Serving Lab

FuriosaAI **RNGD NPU**에서 `Qwen3-8B-FP8` 모델을 올리고, Python SDK 추론 → OpenAI 호환 API 서빙 → 동시 요청 부하 테스트까지 단계별로 실습한 기록입니다.

- **하드웨어**: FuriosaAI RNGD NPU 1장 (메모리 47.5 GiB, 8 코어)
- **소프트웨어**: Furiosa SDK / `furiosa-llm` 2026.4.0, Python 3.12, [uv](https://docs.astral.sh/uv/)
- **모델**: `furiosa-ai/Qwen3-8B-FP8`, `furiosa-ai/Qwen3-VL-4B-Instruct`

## 구성

| 폴더 | 주제 | 주요 내용 |
|---|---|---|
| [`day1_npu_chat_test/`](day1_npu_chat_test/) | NPU 서빙 첫 실행 | `furiosa-llm serve`로 텍스트·비전 모델 서빙, 정밀도별(BF16/FP8) 메모리 예산 계산 |
| [`day2_furiosa_sdk/`](day2_furiosa_sdk/) | Python SDK 추론 | `furiosa_llm.LLM`으로 단건·배치 추론, `temperature`/`top_p`와 프롬프트를 한 변수씩 바꾼 A/B 비교 |
| [`day3_npu_api/`](day3_npu_api/) | OpenAI 호환 API | 서버 점검·기동 스크립트, `openai` SDK 클라이언트, Streaming 챗봇, 클라이언트 측 이력을 쓰는 Multi-turn 챗봇 |
| [`day4_benchmark_test/`](day4_benchmark_test/) | 서빙 성능 측정 | `/metrics` 기반 동시성·Prefix cache 벤치마크, matplotlib·Streamlit 시각화 |
| [`day4_fxb_ex/`](day4_fxb_ex/) | FXB(컴파일된 NPU 실행 번들) | `fxb` CLI로 캐시 조회·호환성 검사, Hugging Face의 FXB 목록 조회 |

## 동시 요청 부하 테스트 결과

강사님이 제공한 Furiosa용 vLLM 벤치마크 도구(`benchmark_serving.py`)로 측정했습니다. 도구는 이 저장소에 포함하지 않았습니다. 입력과 출력을 1024 토큰으로 고정하고, 동시 요청 수만 바꿔 가며 각 구간에서 동시 요청 수의 3배만큼 요청을 보냈습니다. 모든 요청이 실패 없이 완료됐습니다.

| 동시 요청 | TTFT (ms) | TPOT (ms) | 출력 처리량 (tok/s) | 요청/s | 평균 전력 (W) |
|---:|---:|---:|---:|---:|---:|
| 1 | 56.5 | 9.60 | 103.7 | 0.10 | 128.6 |
| 4 | 132.6 | 10.77 | 366.4 | 0.36 | 130.1 |
| 8 | 97.8 | 11.86 | 669.6 | 0.65 | 131.5 |
| 16 | 144.7 | 14.21 | 1,114.9 | 1.09 | 140.1 |
| 32 | 200.5 | 21.11 | 1,502.6 | 1.47 | 145.7 |
| 64 | 294.5 | 34.73 | 1,827.4 | 1.78 | 143.0 |
| 128 | 477.4 | 50.38 | 2,515.0 | 2.46 | 156.5 |

<sub>TTFT: 첫 토큰까지 걸린 시간 · TPOT: 출력 토큰 1개당 생성 시간 · 전력: 측정 구간 NPU 평균</sub>

**해석**

- **처리량과 지연의 트레이드오프**: 동시 요청을 1에서 128로 늘리면 출력 처리량은 약 24배 증가합니다. 같은 구간에서 요청 하나의 TPOT는 약 5배, TTFT는 약 8배 늘어납니다. 서비스 목표가 응답 속도(대화형)인지 처리량(배치)인지에 따라 운영할 동시성 구간이 달라집니다.
- **전력 효율**: 같은 구간에서 전력은 약 22%만 늘어납니다. 그래서 와트당 처리량은 동시성이 높을수록 크게 좋아집니다(약 0.8 → 16 tok/s/W).
- **측정의 한계**: 구간별 요청 수가 적어서(최소 3건), 낮은 동시성 구간의 TTFT는 편차가 큽니다. 동시 4와 8의 TTFT가 역전된 것도 이 때문으로 봅니다. 단발 측정이므로 절대 수치보다 추세를 보는 용도입니다.

## 실행 방법

```bash
uv sync                        # pyproject.toml / uv.lock 기준으로 가상환경 생성
```

각 폴더의 설정 템플릿을 복사해 장치 번호와 포트를 채운 뒤 실행합니다.

| 폴더 | 템플릿 | 주요 값 |
|---|---|---|
| `day2_furiosa_sdk` | `.env.sample` → `.env` | `TEAM_NPU_DEVICE` |
| `day3_npu_api`, `day4_benchmark_test` | `.env.example` → `.env` | `TEAM_NPU_DEVICE`, `TEAM_SERVER_PORT`, `OPENAI_BASE_URL` |

```bash
# 예: API 서버를 띄우고 클라이언트로 호출
cd day3_npu_api
source .env
./01_check_server.sh           # 장치·포트·CLI 옵션 점검
./02_run_llm_server.sh         # 터미널 A: 서버 실행
uv run 03_api_client.py        # 터미널 B: 모델 목록 확인 → Chat Completion 호출
```

## 저장소에 포함하지 않은 것

- 원시 벤치마크 결과(JSON/CSV)와 실행 로그. 위 표는 그 결과를 요약한 값입니다.
- 실습 서버 접속 정보가 들어 있는 문서와 스크린샷, 개인별 `.env`
- 가상환경(`.venv`)과 강사 제공 벤치마크 도구(vLLM 소스 포함)
