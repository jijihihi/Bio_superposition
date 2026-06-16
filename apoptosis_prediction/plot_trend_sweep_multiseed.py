import os
import json
import numpy as np
import pandas as pd

def main():
    base_dir = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity_dim_and_k_sweep"
    
    # 결과를 저장할 디렉토리 (기존 plots 대신 tables 폴더 생성)
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    seeds = [42, 87, 95, 123, 124, 256, 445, 457]
    dims = [1024, 2048, 4096, 8192]
    mutations = ["SNCA", "GBA", "LRRK2"]
    ks = [5, 10, 15, 20, 25]

    for mut in mutations:
        cnn_std_data = {k: [] for k in ks}
        sae_std_data = {d: {k: [] for k in ks} for d in dims}
        
        cnn_morans_data = {k: [] for k in ks}
        sae_morans_data = {d: {k: [] for k in ks} for d in dims}
        
        for seed in seeds:
            cnn_extracted = False
            for d in dims:
                json_path = os.path.join(base_dir, f"seed_{seed}", f"d{d}", "local_linearity_results.json")
                if not os.path.exists(json_path):
                    continue
                    
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                for k in ks:
                    for res in data["results"]:
                        if res["mutation"] == mut and res["k"] == k:
                            if res["source"] == "SAE":
                                sae_std_data[d][k].append(res["mean_ratio"])
                                if "morans_I" in res and res["morans_I"] is not None:
                                    sae_morans_data[d][k].append(res["morans_I"])
                            elif res["source"] == "CNN" and not cnn_extracted:
                                cnn_std_data[k].append(res["mean_ratio"])
                                if "morans_I" in res and res["morans_I"] is not None:
                                    cnn_morans_data[k].append(res["morans_I"])
                                    
                cnn_extracted = True

        # 표 구성을 위한 인덱스 정의 (CNN + SAE 차원들)
        row_names = ["CNN (Baseline)"] + [f"SAE (d={d})" for d in dims]
        col_names = [f"k={k}" for k in ks]

        # 데이터 프레임 초기화 데이터 구조
        def build_dataframe(sae_source, cnn_source, stat_func):
            matrix = []
            
            # 1. CNN 행 계산
            cnn_row = []
            for k in ks:
                vals = cnn_source[k]
                cnn_row.append(stat_func(vals) if vals else np.nan)
            matrix.append(cnn_row)
            
            # 2. SAE 행 계산
            for d in dims:
                sae_row = []
                for k in ks:
                    vals = sae_source[d][k]
                    sae_row.append(stat_func(vals) if vals else np.nan)
                matrix.append(sae_row)
                
            return pd.DataFrame(matrix, index=row_names, columns=col_names)

        # ---------------------------------------------------------
        # 1. Local Std Ratio 표 생성 (Mean & Median)
        # ---------------------------------------------------------
        df_std_mean = build_dataframe(sae_std_data, cnn_std_data, np.mean)
        df_std_median = build_dataframe(sae_std_data, cnn_std_data, np.median)

        # CSV 파일 저장
        df_std_mean.to_csv(os.path.join(tables_dir, f"table_knn_std_mean_{mut}.csv"))
        df_std_median.to_csv(os.path.join(tables_dir, f"table_knn_std_median_{mut}.csv"))

        # 터미널 출력용 콘솔 로그
        print(f"\n" + "="*50)
        print(f" Mutation: {mut} - Local Std Ratio")
        print( "="*50)
        print("[MEAN TABLE]")
        print(df_std_mean.round(4))
        print("\n[MEDIAN TABLE]")
        print(df_std_median.round(4))

        # ---------------------------------------------------------
        # 2. Moran's I 표 생성 (Mean & Median)
        # ---------------------------------------------------------
        # Moran's I 데이터가 실제로 존재하는지 확인
        has_morans = any(sae_morans_data[d][k] for d in dims for k in ks) or any(cnn_morans_data[k] for k in ks)
        
        if has_morans:
            df_moran_mean = build_dataframe(sae_morans_data, cnn_morans_data, np.mean)
            df_moran_median = build_dataframe(sae_morans_data, cnn_morans_data, np.median)

            # CSV 파일 저장
            df_moran_mean.to_csv(os.path.join(tables_dir, f"table_morans_I_mean_{mut}.csv"))
            df_moran_median.to_csv(os.path.join(tables_dir, f"table_morans_I_median_{mut}.csv"))

            print(f"\n" + "-"*50)
            print(f" Mutation: {mut} - Moran's I")
            print( "-"*50)
            print("[MEAN TABLE]")
            print(df_moran_mean.round(4))
            print("\n[MEDIAN TABLE]")
            print(df_moran_median.round(4))

if __name__ == "__main__":
    main()
