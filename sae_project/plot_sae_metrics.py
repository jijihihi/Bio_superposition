import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import numpy as np

# --- 일러스트레이터 편집 최적화 설정 ---
plt.rcParams['svg.fonttype'] = 'none'       # SVG 저장 시 텍스트를 패스로 변환하지 않고 텍스트 형태로 유지
plt.rcParams['pdf.fontencoding'] = 'rgb'     # PDF 인코딩 표준화
plt.rcParams['font.sans-serif'] = 'Arial'    # 일러스트레이터 기본 폰트와 호환성 높은 Arial 사용
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

def main():
    parser = argparse.ArgumentParser(description="Plot SAE Pareto Frontiers (L0 vs FVU, L0 vs Acc).")
    parser.add_argument("--base_dir", type=str, default="/home/ubuntu/model-east3/outputs",
                        help="Base directory containing MoCo_seed* folders")
    parser.add_argument("--linear_probe_csv", type=str, default="sae_linear_probe_1e5_results.csv",
                        help="CSV file from step18_eval_probe_from_cache.py")
    parser.add_argument("--save_dir", type=str, default="./",
                        help="Directory to save the plots and aggregated csv")
    args = parser.parse_args()

    # 1. Load Trainlogs for L0 and FVU
    pattern = os.path.join(args.base_dir, "MoCo_seed*", "SAE_dim*_lambda*_seed48_no_L2norm_loss", "*_trainlog.csv")
    csv_files = glob.glob(pattern)
    
    data_logs = []
    for csv_file in csv_files:
        match = re.search(r'MoCo_seed(\d+)[/\\]SAE_dim(\d+)_lambda(\d+)_seed48', csv_file)
        if not match: continue
        cnn_seed = int(match.group(1))
        dim = int(match.group(2))
        lam = int(match.group(3))

        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0: continue
            last_row = df.iloc[-1]
            data_logs.append({
                "CNN_Seed": cnn_seed,
                "Dimension": dim,
                "Lambda": lam,
                "L0": last_row.get("test_sparsity", np.nan),
                "FVU": last_row.get("test_fvu", np.nan)
            })
        except:
            pass

    df_logs = pd.DataFrame(data_logs)
    
    # 2. Load Linear Probe Accuracy
    linear_csv_path = os.path.join(args.save_dir, args.linear_probe_csv) if not os.path.isabs(args.linear_probe_csv) else args.linear_probe_csv
    if os.path.exists(linear_csv_path):
        df_acc = pd.read_csv(linear_csv_path)
    elif os.path.exists(args.linear_probe_csv):
        df_acc = pd.read_csv(args.linear_probe_csv)
    else:
        print(f"Warning: {args.linear_probe_csv} not found. Test_Acc will be NaN.")
        df_acc = pd.DataFrame(columns=["CNN_Seed", "Dimension", "Lambda", "Test_Acc"])
        
    if df_logs.empty:
        print("No trainlog data found!")
        return

    # 3. Merge Data
    if not df_acc.empty:
        df_merged = pd.merge(df_logs, df_acc, on=["CNN_Seed", "Dimension", "Lambda"], how="left")
    else:
        df_merged = df_logs.copy()
        df_merged["Test_Acc"] = np.nan
        
    df_merged.to_csv(os.path.join(args.save_dir, "aggregated_sae_pareto_metrics.csv"), index=False)
    print("Merged data saved to aggregated_sae_pareto_metrics.csv")

    # 4. Plotting Function for Pareto Frontiers
    def plot_pareto(df, y_col, y_label, base_save_name):
        plt.figure(figsize=(10, 7))
        
        # --- Line 1: Fixed Lambda = 800 ---
        df_dim = df[(df["Lambda"] == 800)].copy()
        if not df_dim.empty:
            mean_dim = df_dim.groupby("Dimension").mean().reset_index()
            std_dim = df_dim.groupby("Dimension").std().reset_index()
            mean_dim = mean_dim.sort_values(by="L0")
            
            # 배경 산점도 (zorder=2)
            sns.scatterplot(data=df_dim, x="L0", y=y_col, color="blue", alpha=0.3, marker="o", zorder=2)
            # 평균선 및 마커 (zorder=3)
            plt.plot(mean_dim["L0"], mean_dim[y_col], marker='o', color="blue", label="Fixed Lambda=800 (Vary Dim)", linewidth=2, zorder=3)
            # 에러바 (zorder=2)
            plt.errorbar(mean_dim["L0"], mean_dim[y_col], 
                         xerr=std_dim["L0"], yerr=std_dim[y_col], 
                         fmt='none', ecolor='blue', capsize=4, alpha=0.7, zorder=2)
            # 텍스트 주석 (zorder=5로 최상단 배치, 수식 기호 $ 제거하여 텍스트 편집기 호환성 확보)
            for _, row in mean_dim.iterrows():
                plt.text(row["L0"], row[y_col], f' d={int(row["Dimension"])}', color='blue', fontsize=10, verticalalignment='bottom', zorder=5)
                
        # --- Line 2: Fixed Dimension = 4096 ---
        df_lam = df[(df["Dimension"] == 4096)].copy()
        if not df_lam.empty:
            mean_lam = df_lam.groupby("Lambda").mean().reset_index()
            std_lam = df_lam.groupby("Lambda").std().reset_index()
            mean_lam = mean_lam.sort_values(by="L0")
            
            sns.scatterplot(data=df_lam, x="L0", y=y_col, color="red", alpha=0.3, marker="s", zorder=2)
            plt.plot(mean_lam["L0"], mean_lam[y_col], marker='s', color="red", label="Fixed d=4096 (Vary Lambda)", linewidth=2, zorder=3)
            plt.errorbar(mean_lam["L0"], mean_lam[y_col], 
                         xerr=std_lam["L0"], yerr=std_lam[y_col], 
                         fmt='none', ecolor='red', capsize=4, alpha=0.7, zorder=2)
            for _, row in mean_lam.iterrows():
                plt.text(row["L0"], row[y_col], f' Lambda={int(row["Lambda"])}', color='red', fontsize=10, verticalalignment='top', zorder=5)
                
        # --- CNN Proxy (d=600, lambda=50) ---
        df_proxy = df[(df["Dimension"] == 600) & (df["Lambda"] == 50)].copy()
        if not df_proxy.empty:
            mean_p = df_proxy.mean()
            std_p = df_proxy.std()
            
            sns.scatterplot(data=df_proxy, x="L0", y=y_col, color="green", alpha=0.3, marker="*", zorder=2)
            plt.scatter(mean_p["L0"], mean_p[y_col], color="green", marker="*", s=200, label="CNN Proxy (d=600, Lambda=50)", zorder=4)
            plt.errorbar(mean_p["L0"], mean_p[y_col], 
                         xerr=std_p["L0"], yerr=std_p[y_col], 
                         fmt='none', ecolor='green', capsize=4, alpha=0.7, zorder=2)
            plt.text(mean_p["L0"], mean_p[y_col], ' Proxy', color='green', fontsize=10, fontweight='bold', zorder=5)

        # 축 레이블 및 타이틀 설정
        plt.xlabel("L0 (Active Features)", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f"Pareto Frontier: L0 vs {y_label}", fontsize=15, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6, zorder=1) # 그리드를 가장 아래로
        plt.tight_layout()
        
        # --- SVG 및 PDF 형식으로 각각 저장 ---
        plt.savefig(os.path.join(args.save_dir, f"{base_save_name}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(os.path.join(args.save_dir, f"{base_save_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved plots: {base_save_name}.svg and {base_save_name}.pdf")

    # 파일명 확장자를 빼고 기본 이름만 전달하도록 변경
    plot_pareto(df_merged, "FVU", "Fraction of Variance Unexplained (FVU)", "Pareto_L0_vs_FVU")
    
    if not df_merged["Test_Acc"].isna().all():
        plot_pareto(df_merged, "Test_Acc", "Linear Probe Accuracy (Test)", "Pareto_L0_vs_Acc")
    else:
        print("Skipping L0 vs Acc plot because Test_Acc data is missing (Run step18 first!).")

if __name__ == "__main__":
    main()
