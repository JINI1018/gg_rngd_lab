"""모델 가중치 크기의 이론적 하한을 계산한다. NPU 실측값이 아니다."""

import argparse


def estimate(parameters_billions: float, bits: int) -> tuple[float, float]:
    if parameters_billions <= 0:
        raise ValueError("파라미터 수는 0보다 커야 한다")
    if bits not in (4, 8, 16, 32):
        raise ValueError("bits는 4, 8, 16, 32 중 하나여야 한다")
    total_bytes = parameters_billions * 1_000_000_000 * bits / 8
    return total_bytes / 1_000_000_000, total_bytes / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters-billions", type=float, required=True)
    parser.add_argument("--bits", type=int, required=True, choices=(4, 8, 16, 32))
    args = parser.parse_args()
    gb, gib = estimate(args.parameters_billions, args.bits)
    print(f"파라미터: {args.parameters_billions:g}B, 저장 정밀도: {args.bits}비트")
    print(f"이론적 가중치 크기: {gb:.2f} GB / {gib:.2f} GiB")
    print("주의: KV 캐시·버퍼·런타임 예약 공간은 포함하지 않음")


if __name__ == "__main__":
    main()