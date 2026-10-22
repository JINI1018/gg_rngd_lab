#!/usr/bin/env bash
# 목적: Linux RNGD 서버에서 SDK, 카드 배정, 포트와 Serving 옵션을 읽기 전용으로 점검한다.
# 준비: 조별 .env 작성 후 source .env. 실행: bash 01_check_server.sh
# 예상: 장치/패키지/옵션/포트 상태가 표시된다. 오류 시 README의 점검표를 따른다.
set -euo pipefail

model_id="${MODEL_ID:-furiosa-ai/Qwen3-8B-FP8}"
device="${TEAM_NPU_DEVICE:-}"
port="${TEAM_SERVER_PORT:-}"
if [[ ! "$device" =~ ^npu:[0-9]+$ ]]; then
  echo 'TEAM_NPU_DEVICE는 DAY 1 PCI-BDF로 대조한 npu:X여야 한다.' >&2
  exit 1
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'TEAM_SERVER_PORT는 강사가 배정한 1024~65535 포트여야 한다.' >&2
  exit 1
fi
command -v furiosa-smi >/dev/null || { echo 'furiosa-smi가 없다.' >&2; exit 1; }
command -v furiosa-llm >/dev/null || { echo 'furiosa-llm이 없다.' >&2; exit 1; }

echo "모델: $model_id / 조별 카드: $device / 서버 포트: $port"
python3 -c 'import sys, importlib.metadata as m; print("Python:", sys.version.split()[0], "/ furiosa-llm:", m.version("furiosa-llm"))'
furiosa-smi info
furiosa-smi status
furiosa-smi ps

help_text="$(furiosa-llm serve --help 2>&1)"
for option in --devices --host --port --reasoning-parser --default-chat-template-kwargs; do
  if ! grep -q -- "$option" <<< "$help_text"; then
    echo "현재 CLI에 $option 옵션이 없다. 강사에게 설치 버전을 보여준다." >&2
    exit 1
  fi
done
echo 'Serving 옵션 확인: 통과'

python3 - "$port" <<'PY'
import socket
import sys
port = int(sys.argv[1])
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"포트 {port} 사용 중: {exc}") from exc
print(f"포트 {port}: 사용 가능")
PY
echo 'DAY 1 배정표의 PCI-BDF와 위 장치 출력을 대조하고, 본인 카드가 비어 있는지 확인한다.'
