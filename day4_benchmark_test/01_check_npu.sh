#!/usr/bin/env bash
# 목적: DAY 3 서버와 조별 RNGD, SDK, /metrics를 읽기 전용으로 확인한다.
# 위치: Linux SSH 터미널. 준비: source .env. 실행: bash 01_check_npu.sh
# 정상: 배정 장치·서버 모델·메트릭 이름을 확인한다. 장애 조치는 README를 참조한다.
set -euo pipefail

device="${TEAM_NPU_DEVICE:-}"
port="${TEAM_SERVER_PORT:-}"
model="${MODEL_ID:-furiosa-ai/Qwen3-8B-FP8}"
if [[ ! "$device" =~ ^npu:[0-9]+$ ]]; then
  echo 'TEAM_NPU_DEVICE를 DAY 1 PCI-BDF 배정표의 npu:X로 설정한다.' >&2
  exit 1
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'TEAM_SERVER_PORT는 강사가 배정한 실제 포트여야 한다.' >&2
  exit 1
fi
command -v furiosa-smi >/dev/null || { echo 'furiosa-smi가 없다.' >&2; exit 1; }
command -v furiosa-llm >/dev/null || { echo 'furiosa-llm이 없다.' >&2; exit 1; }

echo "기대 모델: $model / 배정 장치: $device / 서버 포트: $port"
python3 -c 'import sys, importlib.metadata as m; print("Python:", sys.version.split()[0], "Furiosa-LLM:", m.version("furiosa-llm"))'
furiosa-smi info
furiosa-smi status
furiosa-smi ps
echo '모델 목록:'
curl -fsS "http://127.0.0.1:${port}/v1/models"
echo
echo '측정에 사용할 메트릭 샘플:'
curl -fsS "http://127.0.0.1:${port}/metrics" | grep -E '^(# TYPE )?furiosa_llm_(generation_tokens_total|time_to_first_token_seconds|prefix_cache_hits|prefix_cache_queries|kv_cache_usage_percent)' | head -n 16 || true
echo '장치 번호·PCI-BDF·서버 PID가 우리 조의 DAY 3 기록과 일치하는지 확인한다.'
