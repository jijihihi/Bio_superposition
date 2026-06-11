import os
import glob
import pandas as pd
import numpy as np
import re

def independent_dimension_diagnostic():
    # --- [사용자 설정 항목] 기존 코드의 경로와 동일하게 맞춰주세요 ---
    BASE_DIR = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
    LINEAR_PROBE_CSV = "sae_linear_probe_1e5_results.csv"
    SAVE_DIR = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/L0_FVU_linear_classification"
    WELCH_CSV = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/welch_bound/welch_bound_results.csv"

    print("=" * 60)
    print("📢 [독립 진단 시스템] SAE 차원 및 색상 누락 원인 분석")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1단계: 원본 Summary 및 Trainlog 파일 스캔
    # -------------------------------------------------------------------------
    pattern_summary = os.path.join(BASE_DIR, "*", "*", "all_experiments_summary.csv")
    summary_files = glob.glob(pattern_summary)
    
    logs_dims = set()
    logs_count = 0
    
    if summary_files:
        for sum_file in summary_files:
            try:
                df_sum = pd.read_csv(sum_file)
                if "d_sae" in df_sum.columns:
                    logs_dims.update(df_sum["d_sae"].dropna().unique())
                    logs_count += len(df_sum)
            except:
                pass
    else:
        # Fallback trainlog 스캔
        pattern_trainlog = os.path.join(BASE_DIR, "*", "*", "*_trainlog.csv")
        trainlog_files = glob.glob(pattern_trainlog)
        for csv_file in trainlog_files:
            match = re.search(r'SAE_dim(\d+)', csv_file)
            if match:
                logs_dims.add(int(match.group(1)))
                logs_count += 1

    print(f"[1] 원본 파일 스캔 결과:")
    print(f"    - 발견된 고유 차원 종류: {list(logs_dims)}")
    print(f"    - 읽어온 총 데이터 행 수: {logs_count}개")
    if len(logs_dims) <= 1:
        print("    ❌ [경고] 구글 드라이브 원본 파일 자체에 차원이 1종류뿐이거나 찾지 못했습니다.")

    # -------------------------------------------------------------------------
    # 2단계: Linear Probe CSV 파일 스캔
    # -------------------------------------------------------------------------
    linear_csv_path = os.path.join(SAVE_DIR, LINEAR_PROBE_CSV) if not os.path.isabs(LINEAR_PROBE_CSV) else LINEAR_PROBE_CSV
    acc_dims = set()
    acc_count = 0
    
    if os.path.exists(linear_csv_path):
        try:
            df_acc = pd.read_csv(linear_csv_path)
            acc_count = len(df_acc)
            if "Dimension" in df_acc.columns:
                acc_dims.update(df_acc["Dimension"].dropna().unique())
            
            # 정규식 파싱 시뮬레이션 (기존 코드 로직 적용)
            if "File" in df_acc.columns and (len(acc_dims) <= 1 or 600 in acc_dims):
                simulated_dims = set()
                for fpath in df_acc["File"].astype(str):
                    if "cnn_gap" in fpath or "proxy" in fpath.lower():
                        simulated_dims.add(600)
                    else:
                        m = re.search(r'SAE_dim(\d+)', fpath)
                        if m: simulated_dims.add(int(m.group(1)))
                if simulated_dims:
                    acc_dims.update(simulated_dims)
        except:
            pass
            
    print(f"\n[2] 정확도(Linear Probe) 파일 스캔 결과:")
    print(f"    - 파일 경로: {linear_csv_path}")
    print(f"    - 파일 내 발견된 고유 차원 종류: {list(acc_dims)}")
    print(f"    - 정확도 데이터 행 수: {acc_count}개")
    if len(acc_dims) <= 1:
        print("    ❌ [경고] 정확도 CSV에서 인식된 차원이 1종류(혹은 기본값 600)뿐입니다.")

    # -------------------------------------------------------------------------
    # 3단계: 가상 병합 및 최종 결론 도출
    # -------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("🚨 [최종 진단 및 해결책 요약]")
    print("-"*50)

    # 조건별 범인 확인
    if logs_count == 0:
        print("❌ 원인: 구글 드라이브 내 'all_experiments_summary.csv'나 'trainlog.csv'를 하나도 찾지 못했습니다.")
        print("   👉 해결책: `BASE_DIR` 경로가 올바른지, 코랩에 구글 드라이브 마운트가 정상 처리되었는지 확인하세요.")
        return

    if len(logs_dims) <= 1 and len(acc_dims) <= 1:
        print("❌ 원인: 수집된 모든 파일에 차원(Dimension)이 실제로 1종류밖에 없습니다.")
        print("   👉 해결책: 실험 데이터가 의도대로 여러 차원(예: 512, 1024 등)으로 추출되었는지 폴더 구조를 직접 확인해야 합니다.")
        return

    if len(logs_dims) > 1 and len(acc_dims) <= 1:
        print("❌ 원인: 원본 데이터에는 여러 차원이 있으나, 'Linear Probe 정확도 CSV'를 병합하는 과정에서 차원이 유실되었습니다.")
        print("   👉 원인 세부: 기존 코드의 `if 'SAE_dim' not in fpath:` 조건 때문에 파일명을 제대로 읽지 못해 모든 차원이 '600'으로 강제 초기화되었을 확률이 매우 높습니다.")
        print("   👉 해결책: `sae_linear_probe_1e5_results.csv` 파일 내부의 'File' 컬럼 경로에 'SAE_dim' 문자열이 정확히 포함되어 있는지 확인하세요.")
        return

    # 두 데이터의 데이터 타입 불일치 가능성 확인
    mix_check_logs = [type(x) for x in logs_dims]
    mix_check_acc = [type(x) for x in acc_dims]
    if str in mix_check_logs or str in mix_check_acc:
        print("❌ 원인: 데이터 타입 불일치 (String vs Integer)")
        print("   👉 특정 파일에서는 차원이 문자열('600')로, 다른 곳에서는 숫자(600)로 저장되어 판다스가 병합(`merge`)에 실패했습니다.")
        print("   👉 해결책: 데이터 병합 직전 양쪽 데이터프레임의 Dimension을 `.astype(int)`로 통일해야 합니다.")
        return

    print("✅ 데이터 및 병합 조건은 정상(차원이 2개 이상 독립 존재)입니다.")
    print("❌ 원인: Matplotlib 시각화 코드 오작동 (`plt.errorbar` 고유 이슈)")
    print("   👉 루프 내부에서 `color=palette[i]`를 맵핑할 때, 에러바 플롯 구조상 색상이 단일 색상으로 덮어써지고 있습니다.")
    print("   👉 해결책: 그래프 함수 내부의 컬러 설정을 `color=plt.cm.Set1(i)` 나 `plt.cm.viridis(i / len(dimensions))` 형태로 명시적 변경해야 합니다.")

# 실행
independent_dimension_diagnostic()