# Qwen3-8B: 약 82억 파라미터
params = 8.2e9

# 정밀도별 메모리 계산
precisions = {
    "BF16": 2,  # 16bit = 2Byte
    "FP8": 1,   # 8bit = 1Byte
}

print("모델: Qwen3-8B")
print("-" * 45)

for dtype, bytes_per_param in precisions.items():
    memory_bytes = params * bytes_per_param

    gb = memory_bytes / 1e9
    gib = memory_bytes / (1024 ** 3)

    print(f"{dtype:4s} | {gb:6.2f} GB | {gib:6.2f} GiB")
