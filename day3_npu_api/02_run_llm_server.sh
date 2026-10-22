#!/usr/bin/env bash
# 목적: 각 조의 독립 RNGD 카드에서 Qwen3-8B-FP8 API 서버를 실행한다.
# 위치: Linux SSH 터미널 A. 준비: .env 설정, DAY 2 Python 추론 종료, 01 점검 통과.
# 실행: source .env && bash 02_run_llm_server.sh
# 정상: Uvicorn running 안내 후 다른 터미널에서 /v1/models 응답. 종료: 이 터미널에서 Ctrl+C.
set -euo pipefail

model_id="${MODEL_ID:-furiosa-ai/Qwen3-8B-FP8}"
device="${TEAM_NPU_DEVICE:-}"
port="${TEAM_SERVER_PORT:-}"
if [[ ! "$device" =~ ^npu:[0-9]+$ ]]; then
  echo 'TEAM_NPU_DEVICE가 없다. .env의 장치 번호를 수정하고 source .env를 실행한다.' >&2
  exit 1
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'TEAM_SERVER_PORT가 올바르지 않다.' >&2
  exit 1
fi

bash "$(dirname "$0")/01_check_server.sh"
echo "서버 시작: $model_id / $device / http://127.0.0.1:$port"
echo '준비 완료 로그가 뜰 때까지 이 SSH 터미널을 유지한다. 종료는 Ctrl+C다.'
exec furiosa-llm serve "$model_id" \
  --devices "$device" \
  --host 127.0.0.1 \
  --port "$port" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
