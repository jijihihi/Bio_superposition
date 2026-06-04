import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import argparse
import re

class LinearProbe(nn.Module):
    """Simple linear probe for classification."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.net(x)

def train_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int = 4, epochs: int = 50, lr: float = 0.1,
    batch_size: int = 256, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_alive = X_train.shape[1]
    if n_alive == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(42)
    # 클래스 불균형을 맞추기 위해 가장 적은 클래스의 샘플 수에 맞춰 학습 (Balanced Training)
    class_indices = {c: np.where(y_train == c)[0] for c in range(num_classes)}
    min_class_count = min(len(v) for v in class_indices.values() if len(v) > 0)
    samples_per_class = min_class_count

    probe = LinearProbe(n_alive, num_classes).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(epochs):
        epoch_indices = []
        for c in range(num_classes):
            c_idx = class_indices.get(c, [])
            if len(c_idx) == 0:
                continue
            sampled = rng.choice(c_idx, size=samples_per_class, replace=False)
            epoch_indices.extend(sampled.tolist())
        
        if not epoch_indices:
            continue
            
        epoch_indices = np.array(epoch_indices)
        rng.shuffle(epoch_indices)

        for s in range(0, len(epoch_indices), batch_size):
            ii = epoch_indices[s:s+batch_size]
            xb = torch.from_numpy(X_train[ii]).float().to(device)
            yb = torch.from_numpy(y_train[ii]).long().to(device)

            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        X_te = torch.from_numpy(X_test).float().to(device)
        y_te = torch.from_numpy(y_test).long().to(device)
        test_pred = probe(X_te).argmax(dim=1)
        test_acc = (test_pred == y_te).float().mean().item()
        
        eval_indices = []
        for c in range(num_classes):
            c_idx = class_indices.get(c, [])
            if len(c_idx) == 0:
                continue
            sampled = rng.choice(c_idx, size=min(samples_per_class, len(c_idx)), replace=False)
            eval_indices.extend(sampled.tolist())
            
        eval_indices = np.array(eval_indices)
        X_tr_eval = torch.from_numpy(X_train[eval_indices]).float().to(device)
        y_tr_eval = torch.from_numpy(y_train[eval_indices]).long().to(device)
        train_pred = probe(X_tr_eval).argmax(dim=1)
        train_acc = (train_pred == y_tr_eval).float().mean().item()

    return train_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description="Run Linear Probe on cached SAE representations with Dead Neuron Threshold")
    parser.add_argument("--cache_dir", type=str, default="/home/ubuntu/model-east3/caches", 
                        help="Root directory containing the extracted npz caches")
    parser.add_argument("--save_csv", type=str, default="sae_linear_probe_1e5_results.csv",
                        help="Output CSV file name")
    parser.add_argument("--dead_threshold", type=float, default=1e-5,
                        help="usage_ema threshold to filter dead neurons (e.g., 1e-5)")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of original classes to evaluate on (e.g., 4 for Control, SNCA, GBA, LRRK2)")
    args = parser.parse_args()

    # 모든 .npz 파일 검색 (CNN_seed*/SAE_dim* 등 하위 폴더 모두 포함)
    pattern = os.path.join(args.cache_dir, "**", "*.npz")
    npz_files = glob.glob(pattern, recursive=True)
    
    # SAE 파일만 필터링 (CNN cache 제외)
    sae_files = [f for f in npz_files if "SAE_dim" in f or "sae_" in os.path.basename(f)]
    
    if not sae_files:
        print(f"No SAE npz cache files found in {args.cache_dir}")
        return

    results = []
    print(f"Found {len(sae_files)} SAE cache files. Starting evaluation (Threshold={args.dead_threshold})...")
    
    for fpath in tqdm(sae_files, desc="Evaluating"):
        # CNN Seed, Dimension, Lambda 추출
        match = re.search(r'CNN_seed(\d+).*?SAE_dim(\d+)_lambda(\d+)', fpath)
        cnn_seed = int(match.group(1)) if match else -1
        dim = int(match.group(2)) if match else -1
        lam = int(match.group(3)) if match else -1
        
        try:
            data = np.load(fpath, allow_pickle=True)
            X_all = data['X_all']
            y_all = data['y']
            usage_ema = data['usage_ema']
            
            # 1. 원래 학습에 쓰였던 4개 클래스 (0, 1, 2, 3) 데이터만 걸러내기
            orig_mask = y_all < args.num_classes
            X_orig = X_all[orig_mask]
            y_orig = y_all[orig_mask]
            
            if len(X_orig) == 0:
                print(f"  Skipping {fpath}: No original classes (y < {args.num_classes}) found.")
                continue
            
            # 2. Dead Neuron 필터링 (usage_ema >= 1e-5)
            alive_mask = usage_ema >= args.dead_threshold
            n_alive = int(alive_mask.sum())
            X_orig = X_orig[:, alive_mask]
            
            # 3. Train/Test 분할 (80% Train, 20% Test) - 랜덤 시드 42 고정
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
            )
            
            # 4. Linear Probe 훈련 (step09_sae_eval.py 와 완벽하게 동일한 구조)
            train_acc, test_acc = train_linear_probe(
                X_train, y_train, X_test, y_test, num_classes=args.num_classes
            )
            
            results.append({
                "File": os.path.basename(fpath),
                "CNN_Seed": cnn_seed,
                "Dimension": dim,
                "Lambda": lam,
                "N_Alive": n_alive,
                "Train_Acc": train_acc,
                "Test_Acc": test_acc
            })
            
        except Exception as e:
            print(f"Failed on {fpath}: {e}")
            
    if results:
        df = pd.DataFrame(results)
        save_path = os.path.join(os.path.dirname(args.cache_dir), args.save_csv) if args.cache_dir != "." else args.save_csv
        df.to_csv(save_path, index=False)
        print(f"\nSaved detailed results to {save_path}")
        
        # Dimension과 Lambda 별 평균 정확도 요약 출력
        valid_df = df[df["CNN_Seed"] != -1]
        if not valid_df.empty:
            summary = valid_df.groupby(["Dimension", "Lambda"])["Test_Acc"].agg(["mean", "std", "count"])
            print("\n=== Summary (Test Accuracy by Dimension & Lambda) ===")
            print(summary)

if __name__ == "__main__":
    main()
