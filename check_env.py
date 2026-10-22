import getpass
import sys
from pathlib import Path

print("=== RNGD 실습환경 확인 ===")
print(f"사용자: {getpass.getuser()}")
print(f"홈 디렉터리: {Path.home()}")
print(f"Python 버전: {sys.version.split()[0]}")
print(f"Python 실행 경로: {sys.executable}")
print(f"프로젝트 위치: {Path.cwd()}")
PY