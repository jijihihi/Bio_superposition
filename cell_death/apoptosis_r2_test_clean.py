import argparse
import json
import os
import sys

import matplotlib
import numpy as np

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from run_CNN.logging_utils import get_logger
from trajectory_inference_pipeline.trajectory_utils import load_and_match_cell_death, load_features_cache

logger = get_logger("cell_death_r2_test_clean")


def get_args():
    p = argparse.ArgumentParser("Clean Cell Death Prediction from Cache")
    p.add_argument("--features_cache", type=str, required=True, help="Path to .npz cache (CNN or SAE)")
    p.add_argument("--cell_death_csv", type=str, required=True, help="Path to cell death CSV")
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--model", type=str, default="ridge", choices=["ridge", "xgboost"])
    
    p.add_argument("--gap_l2_norm", action="store_true", help="Apply L2 norm to features")
    p.add_argument("--gap_l1_norm", action="store_true", help="Apply L1 norm to features")
    
    p.add_argument("--pca_dim", type=int, default=250, help="PCA dim before regression. 0=no PCA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--n_permutations", type=int, default=0, help="Num permutations for null p-value")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--title_suffix", type=str, default="", help="Suffix for plot title")
    
    # XGBoost hyperparameters
    p.add_argument("--xgb_n_estimators", type=int, default=300)
    p.add_argument("--xgb_max_depth", type=int, default=5)
    p.add_argument("--xgb_learning_rate", type=float, default=0.04)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    p.add_argument("--xgb_early_stopping", type=int, default=30)
    
    # Context arguments for plotting
    p.add_argument("--config_name", type=str, default="UnknownConfig")
    p.add_argument("--seed_name", type=str, default="0")
    p.add_argument("--layer_name", type=str, default="unknown_layer")
    p.add_argument("--csv_out", type=str, default="")
    
    return p.parse_args()


def build_model_pipeline(args, seed):
    """Builds the scikit-learn compatible pipeline including PCA and Model"""
    steps = [("scaler", StandardScaler())]
    if args.pca_dim > 0:
        steps.append(("pca", PCA(n_components=args.pca_dim, random_state=seed)))
        
    if args.model == "ridge":
        steps.append(("model", RidgeCV(alphas=np.logspace(-3, 5, 50))))
    else:
        import torch
        xgb = XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            gamma=0.0,
            min_child_weight=1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        steps.append(("model", xgb))
        
    return Pipeline(steps)


def evaluate_cv(X, y, args, seed):
    """Run KFold CV and return true and predicted values"""
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y, dtype=np.float64)
    fold_r2s = []
    
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # 1. Apply scaling and PCA manually since XGBoost needs eval_set
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if args.pca_dim > 0:
            n_comp = min(args.pca_dim, X_train.shape[1], X_train.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        if args.model == "ridge":
            model = RidgeCV(alphas=np.logspace(-3, 5, 50))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        else:
            import torch
            model = XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
                early_stopping_rounds=args.xgb_early_stopping,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # Split train into train/val for early stopping
            n_train = len(X_train)
            n_val = max(1, int(n_train * 0.15))
            val_idx = np.random.RandomState(seed + fold_i).permutation(n_train)[:n_val]
            tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
            
            model.fit(
                X_train[tr_idx], 
                y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False
            )
            pred = model.predict(X_test)
            
        y_pred[test_idx] = pred
        fold_r2s.append(r2_score(y_test, pred))
        
    global_r2 = r2_score(y, y_pred)
    return global_r2, np.array(fold_r2s), y_pred


def permutation_test(X, y, args, real_r2, seed):
    """Run parallel permutation test to get p-value"""
    if args.n_permutations <= 0:
        return None, []
        
    def _single_perm(perm_seed):
        rng = np.random.RandomState(perm_seed)
        y_perm = rng.permutation(y)
        perm_r2, _, _ = evaluate_cv(X, y_perm, args, perm_seed)
        return perm_r2
        
    perm_seeds = [seed + 10000 + i * 7 for i in range(args.n_permutations)]
    logger.info(f"Running {args.n_permutations} permutations...")
    perm_r2s = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_single_perm)(ps) for ps in perm_seeds
    )
    
    perm_r2s = np.array(perm_r2s)
    pval = (np.sum(perm_r2s >= real_r2) + 1) / (len(perm_r2s) + 1)
    return pval, perm_r2s





def main():
    args = get_args()
    
    logger.info(f"Loading features from {args.features_cache}")
    data = np.load(args.features_cache, allow_pickle=True)
    
    if "X_gap" in data:
        X = data["X_gap"]
        uids = data["uids"].astype(str) if data["uids"].dtype.kind != "U" else data["uids"]
        logger.info(f"Detected CNN GAP cache: {X.shape}")
    elif "X_all" in data:
        X, _, _, uids, _, _ = load_features_cache(args.features_cache, args.dead_threshold)
        logger.info(f"Detected SAE cache: {X.shape}")
    else:
        raise ValueError("Unknown cache format. Expected 'X_gap' or 'X_all'.")
        
    logger.info("Loading cell death data")
    cell_death = load_and_match_cell_death(args.cell_death_csv, uids)
    
    # Filter valid samples
    valid_mask = ~np.isnan(cell_death)
    X_v = X[valid_mask]
    y_v = cell_death[valid_mask]
    logger.info(f"Valid matched samples: {len(X_v)}/{len(uids)}")
    
    # Normalization
    if args.gap_l1_norm:
        norms = np.sum(np.abs(X_v), axis=1, keepdims=True)
        X_v = X_v / np.where(norms == 0, 1e-12, norms)
        logger.info("Applied L1 normalization")
    if args.gap_l2_norm:
        norms = np.linalg.norm(X_v, axis=1, keepdims=True)
        X_v = X_v / np.where(norms == 0, 1e-12, norms)
        logger.info("Applied L2 normalization")
        
    # Setup Output
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.features_cache), "cell_death")
    os.makedirs(out_dir, exist_ok=True)
    
    def get_superclass(uid):
        uid_str = str(uid)
        if "Control" in uid_str: return "Control"
        if "SNCA" in uid_str: return "SNCA"
        if "GBA" in uid_str: return "GBA"
        if "LRRK2" in uid_str: return "LRRK2"
        return "Unknown"
        
    sc_arr = np.array([get_superclass(u) for u in uids[valid_mask]])
    csv_rows = []
    json_results = {}
    
    mutations = ["SNCA", "GBA", "LRRK2"]
    for mut in mutations:
        mut_mask = sc_arr == mut
        X_mut = X_v[mut_mask]
        y_mut = y_v[mut_mask]
        
        logger.info(f"\n--- Evaluating {mut} only (n={len(X_mut)}) ---")
        real_r2, fold_r2s, y_pred = evaluate_cv(X_mut, y_mut, args, args.seed)
        logger.info(f"R² = {real_r2:.4f} (Folds: {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f})")
        
        json_results[mut] = {
            "r2_mean": float(real_r2),
            "r2_folds": [float(r) for r in fold_r2s]
        }
        
        if args.csv_out:
            for fold_idx, r2 in enumerate(fold_r2s):
                csv_rows.append([args.config_name, args.seed_name, args.model, args.layer_name, f"{mut} only", fold_idx, r2])
                
    if args.csv_out and csv_rows:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        write_header = not os.path.exists(args.csv_out)
        import csv
        with open(args.csv_out, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Config", "Seed", "Model", "Layer", "Group", "Fold_idx", "R2"])
            writer.writerows(csv_rows)
        logger.info(f"Appended results to {args.csv_out}")
    
    # Save JSON
    json_path = os.path.join(out_dir, f"results_{args.model}.json")
    import json
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
