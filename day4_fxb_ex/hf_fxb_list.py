# uv add -U huggingface_hub

# HF 모델 조회
from huggingface_hub import HfApi

api = HfApi()

models = api.list_models(author="furiosa-ai")

print("=== 허깅페이스에 등록된 furiosa NPU 실행 모델 ======================")
for model in models:
    print(model.id)


# 실제 FXB 파일이 있는 모델만 조회
print("=== 허깅페이스에 등록된 furiosa NPU 실행을 위한 fxb 코드 ======================")
for model in api.list_models(author="furiosa-ai"):
    try:
        files = api.list_repo_files(model.id)
        fxbs = [f for f in files if f.endswith(".fxb")]

        if fxbs:
            print(f"\n{model.id}")
            for f in fxbs:
                print(f"  └─ {f}")
    except Exception as e:
        print(f"조회 실패: {model.id} ({e})")