import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from run_CNN.data_bank import load_split_csv
from sae_project.step09_sae_eval import train_linear_probe


class LinearProbe(nn.Module):
    """Simple linear probe for classification."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Run Linear Probe on cached SAE representations with Dead Neuron Threshold"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/ubuntu/model-east3/caches",
        help="Root directory containing the extracted npz caches",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="sae_linear_probe_results.csv",
        help="Output CSV file name",
    )
    parser.add_argument(
        "--model_base_dir",
        type=str,
        default="/home/ubuntu/model-east3/outputs",
        help="Base directory containing MoCo_seedXX folders to find train/val/test CSV splits",
    )
    parser.add_argument(
        "--dead_threshold",
        type=float,
        default=1e-5,
        help="usage_ema threshold to filter dead neurons (e.g., 1e-5)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of original classes to evaluate on (e.g., 4 for Control, SNCA, GBA, LRRK2)",
    )
    parser.add_argument(
        "--target_configs",
        type=str,
        nargs="+",
        default=None,
        help="List of target configs (e.g. 600_50 1024_800) to evaluate. If none, evaluates all.",
    )
    args = parser.parse_args()

    # 모든 .npz 파일 검색 (CNN_seed*/SAE_dim* 등 하위 폴더 모두 포함)
    pattern = os.path.join(args.cache_dir, "**", "*.npz")
    npz_files = glob.glob(pattern, recursive=True)

    # SAE 파일만 필터링 (명확하게 sae_refactoring_gap_ 이 포함된 파일만 선택)
    sae_files = [
        f for f in npz_files if "sae_refactoring_gap_" in os.path.basename(f)
    ]

    if not sae_files:
        print(f"No SAE npz cache files found in {args.cache_dir}")
        return

    results = []
    print(
        f"Found {len(sae_files)} SAE cache files. Starting evaluation (Threshold={args.dead_threshold})..."
    )

    for fpath in tqdm(sae_files, desc="Evaluating"):
        # CNN Seed, Dimension, Lambda 추출
        # 지원하는 패턴: CNN_seed42, SAE_dim4096_lambda800 또는 sae_gap_d4096_lam800
        m_seed = re.search(r"(?:CNN_seed|seed)(\d+)", fpath)
        m_dim = re.search(r"(?:SAE_dim|_d|dim)(\d+)", fpath)
        m_lam = re.search(r"(?:_lambda|_lam)(\d+)", fpath)

        cnn_seed = int(m_seed.group(1)) if m_seed else -1
        dim = int(m_dim.group(1)) if m_dim else -1
        lam = int(m_lam.group(1)) if m_lam else -1

        if args.target_configs:
            config_str = f"{dim}_{lam}"
            if config_str not in args.target_configs:
                continue

        try:
            data = np.load(fpath, allow_pickle=True)
            X_all = data["X_all"]
            y_all = data["y"]
            lines_all = data["lines"]
            usage_ema = data["usage_ema"]

            # 1. 원래 학습에 쓰였던 4개 클래스 (0, 1, 2, 3) 데이터만 걸러내기
            orig_mask = y_all < args.num_classes
            X_orig = X_all[orig_mask]
            y_orig = y_all[orig_mask]
            uids_all = data["uids"]

            if len(X_orig) == 0:
                print(
                    f"  Skipping {fpath}: No original classes (y < {args.num_classes}) found."
                )
                continue

            # 2. Dead Neuron 필터링 (usage_ema >= 1e-5)
            alive_mask = usage_ema >= args.dead_threshold
            n_alive = int(alive_mask.sum())
            X_orig = X_orig[:, alive_mask]

            # 3. Train/Test 분할 (실제 모델이 학습에 사용했던 train_split.csv 기반)
            # cache에서 저장된 uid와 각 split에 해당하는 uid를 매칭하여 X_train과 X_test를 완벽 분리합니다.
            model_dir = os.path.join(args.model_base_dir, f"MoCo_seed{cnn_seed}")
            train_csv = os.path.join(model_dir, "train_split.csv")
            val_csv = os.path.join(model_dir, "val_split.csv")
            test_csv = os.path.join(model_dir, "test_split.csv")

            if not os.path.exists(train_csv):
                print(f"  Skipping {fpath}: train_split.csv not found at {train_csv}")
                continue

            train_uids = set(load_split_csv(train_csv))
            val_uids = (
                set(load_split_csv(val_csv)) if os.path.exists(val_csv) else set()
            )
            test_uids = (
                set(load_split_csv(test_csv)) if os.path.exists(test_csv) else set()
            )
            eval_uids = val_uids.union(test_uids)

            # npz 안에 저장된 uids 기반으로 인덱스 찾기
            # 캐시에 저장된 uid는 상대 경로일 수 있으므로 끝부분만 매칭하거나, 그대로 매칭
            def _normalize_uid(u):
                return u.replace("\\", "/")

            train_uids_norm = {_normalize_uid(u) for u in train_uids}
            val_uids_norm = {_normalize_uid(u) for u in val_uids}
            test_uids_norm = {_normalize_uid(u) for u in test_uids}

            train_idx, val_idx, test_idx = [], [], []
            for i, u in enumerate(uids_all[orig_mask]):
                un = _normalize_uid(str(u))
                if un in train_uids_norm or any(
                    un.endswith(tu) for tu in train_uids_norm
                ):
                    train_idx.append(i)
                elif un in val_uids_norm or any(
                    un.endswith(vu) for vu in val_uids_norm
                ):
                    val_idx.append(i)
                elif un in test_uids_norm or any(
                    un.endswith(tu) for tu in test_uids_norm
                ):
                    test_idx.append(i)

            eval_idx = val_idx + test_idx

            if len(train_idx) == 0 or len(eval_idx) == 0:
                print(
                    f"  Skipping {fpath}: Train({len(train_idx)}) or Eval({len(eval_idx)}) missing after UID matching."
                )
                continue

            X_train = X_orig[train_idx]
            y_train = y_orig[train_idx]
            X_test = X_orig[eval_idx]
            y_test = y_orig[eval_idx]

            # 4. Linear Probe 훈련 (step09_sae_eval.py 와 완벽하게 동일한 구조: 스케일 보정을 위해 벡터 L2 정규화 후 분류기 학습)
            import torch.nn.functional as F

            X_train_t = torch.from_numpy(X_train).float()
            X_test_t = torch.from_numpy(X_test).float()
            X_train = F.normalize(X_train_t, dim=1, eps=1e-12).numpy()
            X_test = F.normalize(X_test_t, dim=1, eps=1e-12).numpy()

            # step09_sae_eval의 train_linear_probe는 dict를 반환함: {"train_acc": X, "test_acc": Y, "d_probe": Z, ...}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            probe_results = train_linear_probe(
                X_train,
                y_train,
                X_test,
                y_test,
                num_classes=args.num_classes,
                device=device,
                normalize_repr=False,
                verbose=False,
            )
            train_acc = probe_results["train_acc"]
            test_acc = probe_results["test_acc"]

            results.append(
                {
                    "File": os.path.basename(fpath),
                    "CNN_Seed": cnn_seed,
                    "Dimension": dim,
                    "Lambda": lam,
                    "N_Alive": n_alive,
                    "N_Train": len(train_idx),
                    "N_Val": len(val_idx),
                    "N_Test": len(test_idx),
                    "Train_Acc": train_acc,
                    "Test_Acc": test_acc,
                }
            )

        except Exception as e:
            print(f"Failed on {fpath}: {e}")

    if results:
        df = pd.DataFrame(results)
        save_path = (
            os.path.join(os.path.dirname(args.cache_dir), args.save_csv)
            if args.cache_dir != "."
            else args.save_csv
        )
        df.to_csv(save_path, index=False)
        print(f"\nSaved detailed results to {save_path}")

        # Dimension과 Lambda 별 평균 정확도 요약 출력
        valid_df = df[df["CNN_Seed"] != -1]
        if not valid_df.empty:
            summary = valid_df.groupby(["Dimension", "Lambda"])["Test_Acc"].agg(
                ["mean", "std", "count"]
            )
            print("\n=== Summary (Test Accuracy by Dimension & Lambda) ===")
            print(summary)


if __name__ == "__main__":
    main()
