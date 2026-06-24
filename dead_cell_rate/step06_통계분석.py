#!/usr/bin/env python3
"""
클래스별 세포사멸율 통계 분석
- 표준편차, 상대표준편차(CV) 계산
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    csv_path = r"C:\Users\admin\Desktop\이미지별_세포사멸율_7200.csv"

    print("=" * 80)
    print("📊 클래스별 세포사멸율 통계 분석")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    print(f"총 이미지 수: {len(df)}")

    print("\n" + "=" * 80)
    print(
        f"{'Class':<12} {'N':>8} {'Mean':>12} {'Std':>12} {'CV(%)':>12} {'Min':>12} {'Max':>12}"
    )
    print("-" * 80)

    results = []
    for cls in ["Control", "GBA", "LRRK2", "SNCA"]:
        rates = df[df["class"] == cls]["intensity_rate"]
        if len(rates) == 0:
            continue

        mean = rates.mean()
        std = rates.std()
        cv = (std / mean * 100) if mean > 0 else 0  # Coefficient of Variation (%)

        results.append(
            {
                "class": cls,
                "n": len(rates),
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": rates.min(),
                "max": rates.max(),
            }
        )

        print(
            f"{cls:<12} {len(rates):>8} {mean:>12.6f} {std:>12.6f} {cv:>12.2f} {rates.min():>12.6f} {rates.max():>12.6f}"
        )

    print("-" * 80)

    # 누가 가장 큰 CV를 가지는지 확인
    print("\n📈 상대표준편차(CV) 순위:")
    results_sorted = sorted(results, key=lambda x: x["cv"], reverse=True)
    for i, r in enumerate(results_sorted, 1):
        print(f"   {i}. {r['class']}: CV = {r['cv']:.2f}%")

    # Pooled 방식으로 재계산해서 비교
    print("\n" + "=" * 80)
    print("📊 Pooled 방식 재계산 (Step03와 비교용)")
    print("-" * 80)

    for cls in ["Control", "GBA", "LRRK2", "SNCA"]:
        cls_df = df[df["class"] == cls]
        if len(cls_df) == 0:
            continue

        total_apop = cls_df["apoptotic_intensity"].sum()
        total_intensity = cls_df["total_intensity"].sum()
        pooled_rate = total_apop / total_intensity if total_intensity > 0 else 0

        per_image_mean = cls_df["intensity_rate"].mean()

        print(
            f"{cls:<12} | Pooled: {pooled_rate:.6f} | Per-Image Mean: {per_image_mean:.6f} | Ratio: {pooled_rate/per_image_mean if per_image_mean > 0 else 0:.2f}x"
        )

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()

# multi otsu 이용해서 (3 class) 38448.78338 이정도 뽑고 이미지마다 세포사멸율 정량화. threshold 구할때도 핵 비율 0.25함.
# Control      | Pooled: 0.412548 | Per-Image Mean: 0.167791 | Ratio: 2.46x
# GBA          | Pooled: 0.589028 | Per-Image Mean: 0.202781 | Ratio: 2.90x
# LRRK2        | Pooled: 0.392823 | Per-Image Mean: 0.084661 | Ratio: 4.64x
# SNCA         | Pooled: 0.680205 | Per-Image Mean: 0.192590 | Ratio: 3.53x
# 일단 이런 결과 까지가 최대. 이걸로 일단 켄달 상관계수 해보자.
