"""DAY 2 공통 설정. 조별 장치 번호는 환경변수로만 받는다."""

import os
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트의 .env 로딩
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(override=True, dotenv_path=BASE_DIR / ".env")

def settings() -> tuple[str, str]:
    model = os.getenv(
        "FURIOSA_MODEL_ID",
        "furiosa-ai/Qwen3-8B-FP8"
    )

    device = os.getenv(
        "TEAM_NPU_DEVICE",
        "npu:7"
    )

    return model, device
