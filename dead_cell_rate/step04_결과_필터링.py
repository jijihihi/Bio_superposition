#!/usr/bin/env python3
"""
세포사멸율 결과 필터링 스크립트

step03에서 생성된 CSV 결과에서:
- Control보다 GBA, SNCA, LRRK2 모두 높은 조합만 추출
- 모든 알고리즘 × Rolling Ball Radius × 평가방식(rate, intensity_rate, intensity_per_pixel) 조합 평가

Usage:
    python step04_결과_필터링.py --input_csv 세포사멸율_pooled_결과.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter apoptosis rate results where disease classes > Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_pooled_결과.csv",
        help="Input CSV file from step03",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_필터링_결과.csv",
        help="Output CSV file with filtered results",
    )
    parser.add_argument(
        "--qc_filter",
        type=str,
        choices=["all", "qc_only", "both"],
        default="both",
        help="QC filter to analyze",
    )

    return parser.parse_args()


def analyze_results(df, qc_filter_value):
    """
    특정 qc_filter에 대해 Control보다 질병군이 모두 높은 조합 찾기
    """
    df_filtered = df[df["qc_filter"] == qc_filter_value]

    # 알고리즘 목록 추출
    rate_cols = [
        c for c in df.columns if c.startswith("rate_") and not c.startswith("rate_mean")
    ]
    intensity_rate_cols = [c for c in df.columns if c.startswith("intensity_rate_")]
    intensity_per_pixel_cols = [
        c for c in df.columns if c.startswith("intensity_per_pixel_")
    ]

    algorithms = set()
    for col in rate_cols:
        algo = col.replace("rate_", "")
        algorithms.add(algo)

    radii = sorted(df_filtered["rolling_ball_radius"].unique())

    valid_combinations = []

    for radius in radii:
        df_radius = df_filtered[df_filtered["rolling_ball_radius"] == radius]

        for algo in algorithms:
            for metric_type in ["rate", "intensity_rate", "intensity_per_pixel"]:
                col_name = f"{metric_type}_{algo}"

                if col_name not in df_radius.columns:
                    continue

                # 각 클래스의 값 추출
                control_row = df_radius[df_radius["class"] == "Control"]
                gba_row = df_radius[df_radius["class"] == "GBA"]
                lrrk2_row = df_radius[df_radius["class"] == "LRRK2"]
                snca_row = df_radius[df_radius["class"] == "SNCA"]

                if (
                    len(control_row) == 0
                    or len(gba_row) == 0
                    or len(lrrk2_row) == 0
                    or len(snca_row) == 0
                ):
                    continue

                control_val = control_row[col_name].values[0]
                gba_val = gba_row[col_name].values[0]
                lrrk2_val = lrrk2_row[col_name].values[0]
                snca_val = snca_row[col_name].values[0]

                # NaN 체크
                if any(pd.isna([control_val, gba_val, lrrk2_val, snca_val])):
                    continue

                # 조건: GBA, LRRK2, SNCA 모두 Control보다 큰 경우
                if (
                    gba_val > control_val
                    and lrrk2_val > control_val
                    and snca_val > control_val
                ):
                    # threshold 값 추출
                    thresh_col = f"global_thresh_{algo}"
                    thresh_val = (
                        control_row[thresh_col].values[0]
                        if thresh_col in df_radius.columns
                        else np.nan
                    )

                    valid_combinations.append(
                        {
                            "qc_filter": qc_filter_value,
                            "rolling_ball_radius": radius,
                            "algorithm": algo,
                            "metric_type": metric_type,
                            "global_threshold": thresh_val,
                            "Control": control_val,
                            "GBA": gba_val,
                            "LRRK2": lrrk2_val,
                            "SNCA": snca_val,
                            "GBA_vs_Control": (
                                gba_val / control_val if control_val > 0 else np.inf
                            ),
                            "LRRK2_vs_Control": (
                                lrrk2_val / control_val if control_val > 0 else np.inf
                            ),
                            "SNCA_vs_Control": (
                                snca_val / control_val if control_val > 0 else np.inf
                            ),
                            "min_ratio": (
                                min(
                                    gba_val / control_val,
                                    lrrk2_val / control_val,
                                    snca_val / control_val,
                                )
                                if control_val > 0
                                else np.inf
                            ),
                        }
                    )

    return valid_combinations


def main():
    args = get_args()

    print("=" * 70)
    print("🔍 세포사멸율 결과 필터링")
    print("=" * 70)
    print(f"   입력 파일: {args.input_csv}")
    print(f"   출력 파일: {args.output_csv}")
    print("=" * 70)

    # CSV 로드
    try:
        df = pd.read_csv(args.input_csv)
        print(f"\n✅ CSV 로드 완료: {len(df)}행")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    # 분석
    all_valid = []

    qc_filters = ["all", "qc_only"] if args.qc_filter == "both" else [args.qc_filter]

    for qc_filter in qc_filters:
        print(f"\n[{qc_filter}] 분석 중...")
        valid = analyze_results(df, qc_filter)
        all_valid.extend(valid)
        print(f"   유효 조합: {len(valid)}개")

    if len(all_valid) == 0:
        print("\n❌ 유효한 조합이 없습니다 (모든 질병군 > Control 조건 불충족)")
        return

    # 결과 DataFrame
    df_valid = pd.DataFrame(all_valid)

    # 정렬: min_ratio 내림차순 (가장 차이가 큰 것부터)
    df_valid = df_valid.sort_values("min_ratio", ascending=False)

    # 저장
    df_valid.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n✅ 결과 저장: {args.output_csv}")

    # 요약 출력
    print("\n" + "=" * 70)
    print("📊 유효한 조합 요약 (Control < GBA, LRRK2, SNCA)")
    print("=" * 70)

    print(f"\n총 유효 조합: {len(df_valid)}개")

    # QC 필터별
    for qc in df_valid["qc_filter"].unique():
        df_qc = df_valid[df_valid["qc_filter"] == qc]
        print(f"\n[{qc}] {len(df_qc)}개 조합")

    # Metric 타입별
    print("\n📈 Metric 타입별:")
    for metric in ["rate", "intensity_rate", "intensity_per_pixel"]:
        count = len(df_valid[df_valid["metric_type"] == metric])
        print(f"   {metric}: {count}개")

    # 상위 10개 출력
    print("\n" + "=" * 70)
    print("🏆 상위 10개 조합 (min_ratio 기준)")
    print("=" * 70)
    print(
        f"{'QC':<10} {'Radius':<8} {'Algorithm':<12} {'Metric':<20} {'MinRatio':<10} {'Control':<12} {'GBA':<12} {'LRRK2':<12} {'SNCA':<12}"
    )
    print("-" * 110)

    for _, row in df_valid.head(10).iterrows():
        print(
            f"{row['qc_filter']:<10} {row['rolling_ball_radius']:<8} {row['algorithm']:<12} {row['metric_type']:<20} {row['min_ratio']:.4f}    {row['Control']:.6f}   {row['GBA']:.6f}   {row['LRRK2']:.6f}   {row['SNCA']:.6f}"
        )

    print("\n" + "=" * 70)
    print("✅ 완료!")


if __name__ == "__main__":
    main()
