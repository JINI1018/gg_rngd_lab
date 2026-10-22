# DAY 2 Furiosa SDK와 LLM 직접 추론 실습

## Furiosa LLM SDK 설치
```
uv add furiosa-llm
# 버전 의존성 있으므로, 버전 고정하긴
# uv remove requests
uv add requests==2.32.2
```

이 폴더의 Python 파일은 SSH로 접속한 Linux 서버에서 실행한다. 기준 문서는 Furiosa SDK 2026.4.0 공식 문서이며, 수업 서버의 설치 버전과 장비 출력은 강사가 수업 전에 확인한다. 실제 RNGD 서버에서 이 예제를 실행해 검증할 수 없었으므로, 이 폴더에는 가상의 모델 응답이나 성능 수치를 넣지 않았다.

## 강사 사전 준비

1. 각 조의 계정, 카드 Device/PCI-BDF, `npu:X` 표기, 작업 경로를 대조한다. `npu0`은 Python에서 `npu:0`으로 지정한다. 번호는 조별 배정값을 따른다.
2. 서버에 드라이버·펌웨어·Furiosa-LLM이 설치되고 `furiosa-smi`와 `fxb` 명령이 동작하는지 확인한다.
3. `furiosa-ai/Qwen3-8B-FP8`의 가중치와 FXB를 다운로드할 수 있거나 미리 캐시해 둔다. 모델 로딩에는 가중치와 호환 FXB가 모두 필요하다.
4. 학생 계정에서 카드가 보이고 다른 추론 또는 Serving 프로세스가 점유하지 않는지 `furiosa-smi status`, `furiosa-smi ps`로 확인한다.
5. 첫 로딩 시간, 네트워크 접근, 디스크 여유, 조별 출력 폴더의 쓰기 권한을 실제 서버에서 점검한다.

## 조별 설정

Windows PowerShell에서 `ssh 사용자@서버주소`로 접속한 뒤 Linux 서버에서 아래를 실행한다. 두 학생은 같은 카드를 사용하므로 추론 스크립트를 동시에 실행하지 않는다.

```bash
cp team_config.example.sh team_config.sh
# VS Code 또는 편집기로 team_config.sh의 npu:__을 배정 번호로 바꾼다.
source ./team_config.sh
uv run 00_preflight.py
```

`team_config.sh`에는 비밀번호나 개인키를 넣지 않는다. 배정 번호는 DAY 1 분석지의 Device와 강사 PCI-BDF 배정표를 함께 대조한다. `source` 명령은 새 SSH 세션마다 다시 실행한다.

## 순서대로 실행

```bash
uv run 01_first_inference.py
furiosa-smi ps
uv run 02_batch_inference.py --batch-size 1 --output A_batch1.jsonl
uv run 02_batch_inference.py --batch-size 3 --output B_batch3.jsonl
uv run 03_sampling_experiment.py --output sampling_results.csv
furiosa-smi status
furiosa-smi ps
```

한 스크립트가 종료되어 NPU 자원을 반환한 뒤 다음 스크립트를 실행한다. `with LLM(...)` 블록을 벗어날 때 엔진이 종료된다. 실행이 오래 걸리면 해당 터미널에서 `Ctrl+C`를 누른 뒤 강사와 `furiosa-smi ps`를 확인한다. 다른 조의 프로세스를 종료하지 않는다.

## 결과 확인

- 첫 추론: 질문과 비어 있지 않은 응답이 화면에 나온다. 응답 문구는 모델 실행마다 달라질 수 있다.
- 배치: `prompts.jsonl`의 Q1–Q4에 대응하는 결과 4건이 입력 순서대로 JSONL 파일에 저장된다.
- 샘플링: 기준, 프롬프트 변경, 샘플링 변경의 세 조건이 CSV에 저장된다. 결과가 반드시 서로 달라야 성공인 것은 아니다.
- 종료: `furiosa-smi ps`에서 **배정 카드**를 점유하는 본인 스크립트가 남지 않아야 한다.

## 오류가 나면

| 증상 | 확인할 것 |
|---|---|
| `TEAM_NPU_DEVICE` 오류 | `team_config.sh` 편집 후 `source ./team_config.sh`를 다시 실행했는지 확인 |
| `ModuleNotFoundError: furiosa_llm` | 강사에게 설치 Python 환경과 패키지 버전 확인 요청 |
| 모델 또는 FXB를 찾지 못함 | 강사가 모델 접근·캐시·SDK/FXB 호환을 확인; 학생은 임의 빌드나 설치를 하지 않음 |
| device busy 또는 메모리 부족 | 조별 카드 지정, `furiosa-smi ps/status`, 동시 실행 여부 확인 |
| 다운로드 실패 | 네트워크·인증·캐시·디스크 여유를 강사가 확인 |
| 빈 응답 또는 응답 수 불일치 | 전체 로그와 설정값을 보존하고 강사에게 전달 |

## 파일 구성

- `day02_config.py`: 조별 장치 환경변수 검사
- `00_preflight.py`: 환경과 장치의 읽기 전용 확인
- `01_first_inference.py`: 첫 단일 채팅 추론
- `02_batch_inference.py`: JSONL 질문을 묶어 추론하고 결과 저장
- `03_sampling_experiment.py`: 프롬프트·샘플링 조건 비교
- `prompts.jsonl`: 배치 실습 입력
- `DAY02_결과기록_조번호.md`, `DAY02_체크리스트.md`: 제출 양식

## 공식 근거

- [Furiosa-LLM Quick Start](https://developer.furiosa.ai/latest/en/get_started/furiosa_llm.html)
- [LLM class](https://developer.furiosa.ai/latest/en/furiosa_llm/reference/llm.html)
- [SamplingParams](https://developer.furiosa.ai/latest/en/furiosa_llm/reference/sampling_params.html)
- [Qwen3 dense 모델](https://developer.furiosa.ai/latest/en/furiosa_llm/models/qwen3.html)
- [FXB](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)
