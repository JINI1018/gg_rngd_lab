"""모델 로딩 전에 Linux, SDK, FXB 도구와 배정 장치를 확인한다.

[실습 내용]
FuriosaAI NPU로 모델을 로딩하기 전에 실행 환경이 준비되었는지 점검하는
"프리플라이트(preflight)" 스크립트다. (1) OS가 Linux인지, (2) furiosa-llm
SDK가 설치되어 있는지, (3) furiosa-smi/fxb CLI 도구가 PATH에 있는지(실행은
furiosa-smi만 하며, fxb는 존재 여부만 확인) 점검하고, (4) furiosa-smi로
NPU 장치 정보와 점유 상태를 출력해 조별로 배정된 장치가 비어 있는지 눈으로
대조하게 한다. 각 단계에서 문제가 발견되면 바로
SystemExit으로 중단시켜, 뒤 단계(실제 모델 로딩)에서 불명확한 에러를
마주치지 않도록 한다.
"""

import importlib.metadata
import platform
import shutil
import subprocess
import sys

from day02_config import settings


def main() -> None:
    # .env / 환경변수에서 모델 ID와 조별 NPU 장치 번호를 읽어온다
    model, device = settings()
    print(f"운영체제: {platform.system()} / Python: {sys.version.split()[0]}")
    print(f"모델: {model} / 조별 장치: {device}")

    # furiosa-smi/fxb는 Linux 전용 도구이므로 다른 OS에서는 바로 중단
    if platform.system() != "Linux":
        raise SystemExit("이 파일은 SSH로 접속한 Linux 서버에서 실행한다")
    try:
        # furiosa-llm 패키지 설치 여부를 버전 조회로 확인
        print("furiosa-llm:", importlib.metadata.version("furiosa-llm"))
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit("furiosa-llm 패키지가 없다. 강사에게 설치 상태를 문의한다") from exc

    # furiosa-smi(NPU 상태 조회)와 fxb(모델 빌드, 이후 실습에서 사용)가
    # PATH에 설치돼 있는지만 확인한다. 이 스크립트에서 fxb를 직접 실행하지는 않는다
    for command in ("furiosa-smi", "fxb"):
        path = shutil.which(command)
        if path is None:
            raise SystemExit(f"{command} 명령을 찾지 못했다. 강사에게 문의한다")
        print(f"{command}: {path}")

    # furiosa-smi info/status/ps를 차례로 실행해 장치 목록과 점유 현황을 출력
    for subcommand in ("info", "status", "ps"):
        print(f"\n$ furiosa-smi {subcommand}")
        subprocess.run(["furiosa-smi", subcommand], check=True)
    print("\n장치 이름과 PCI-BDF를 DAY 1 배정표와 대조하고, 점유 중이면 모델을 로딩하지 않는다.")


if __name__ == "__main__":
    main()
