import os
import glob
import numpy as np
from collections import defaultdict
import argparse
import re

def normalize_uid(uid: str) -> str:
    KNOWN_ROOTS = [
        "/home/ubuntu/model-east3/wds_shards_tar/",
        "/home/ubuntu/model-east3/wds_shards_tar\\",
        "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/"
    ]
    for root in KNOWN_ROOTS:
        if uid.startswith(root):
            uid = uid[len(root):]
            break
    
    for cls_prefix in ["Control/", "SNCA/", "GBA/", "LRRK2/", 
                       "GBA_346/", "GBA_WIMP4/", "SNCA-G51D_isogenic/", 
                       "SNCA-G51D/", "SNCA_isogenic/", "SNCAx3_isogenic/",
                       "alpha_syn_1day/", "alpha_syn_7day/"]:
        idx = uid.find(cls_prefix)
        if idx >= 0:
            return uid[idx:]
    return uid

def load_split_uids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1, ndmin=2)
        if df.size == 0:
            return set()
        uids = df[:, 0].tolist()
        return {normalize_uid(str(u)) for u in uids}
    except Exception as e:
        print(f"[!] Error reading CSV {csv_path}: {e}")
        return set()

def main():
    parser = argparse.ArgumentParser(description="Check train/val/test splits in caches")
    parser.add_argument("--caches_dir", type=str, default="/home/ubuntu/model-east3/caches", help="Path to caches folder")
    # [수정 포인트] 기본 경로를 outputs 폴더로 변경했습니다.
    parser.add_argument("--model_base_dir", type=str, default="/home/ubuntu/model-east3/outputs", help="Base directory containing MoCo_seedXX folders")
    args = parser.parse_args()

    print(f"Caches Directory: {args.caches_dir}")
    print(f"Model Base Directory (Outputs): {args.model_base_dir}")

    npz_files = glob.glob(os.path.join(args.caches_dir, "**", "*.npz"), recursive=True)
    if not npz_files:
        print("No .npz files found in the specified caches_dir.")
        return

    CLASS_NAMES = {
        0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2",
        4: "GBA_346", 5: "GBA_WIMP4", 6: "SNCA-G51D_iso", 
        7: "SNCA-G51D", 8: "SNCA_iso", 9: "SNCAx3_iso",
        10: "alpha_1d", 11: "alpha_7d"
    }
    
    seed_to_splits = {}
    
    for npz in sorted(npz_files):
        try:
            m = re.search(r'seed(\d+)', npz, re.IGNORECASE)
            if not m:
                print(f"[!] Skipping {os.path.basename(npz)} (Could not determine seed from path)")
                continue
            
            seed = m.group(1)
            
            if seed not in seed_to_splits:
                # outputs/MoCo_seedXX 구조에 맞춤
                csv_dir = os.path.join(args.model_base_dir, f"MoCo_seed{seed}")
                
                train_csv = os.path.join(csv_dir, "train_split.csv")
                val_csv = os.path.join(csv_dir, "val_split.csv")
                test_csv = os.path.join(csv_dir, "test_split.csv")
                
                if not os.path.exists(train_csv):
                    print(f"[!] Warning: CSV files not found for seed {seed} in {csv_dir}")
                    seed_to_splits[seed] = (set(), set(), set())
                else:
                    t_set = load_split_uids(train_csv)
                    v_set = load_split_uids(val_csv)
                    te_set = load_split_uids(test_csv)
                    seed_to_splits[seed] = (t_set, v_set, te_set)
                    print(f"[✓] Loaded CSVs for seed {seed} (Train: {len(t_set)}, Val: {len(v_set)}, Test: {len(te_set)})")
            
            train_set, val_set, test_set = seed_to_splits[seed]
            
            # allow_pickle=True로 내부 데이터 로드 에러 원천 차단
            data = np.load(npz, allow_pickle=True)
            if 'uids' not in data or 'y' not in data:
                print(f"[!] Skipping {os.path.basename(npz)} (Missing 'uids' or 'y')")
                continue
                
            uids = data['uids']
            y = data['y']
            
            counts = {
                "train": defaultdict(int),
                "val": defaultdict(int),
                "test": defaultdict(int),
                "unknown": defaultdict(int)
            }
            
            for i, uid in enumerate(uids):
                if isinstance(uid, bytes):
                    uid_str = uid.decode('utf-8')
                else:
                    uid_str = str(uid)
                    
                norm_uid = normalize_uid(uid_str)
                label = int(y[i])
                
                if norm_uid in train_set:
                    counts["train"][label] += 1
                elif norm_uid in val_set:
                    counts["val"][label] += 1
                elif norm_uid in test_set:
                    counts["test"][label] += 1
                else:
                    counts["unknown"][label] += 1
            
            print("-" * 60)
            print(f"📂 {os.path.basename(os.path.dirname(npz))} / {os.path.basename(npz)}")
            print(f"  Total uids in cache: {len(uids)}")
            
            for split in ["train", "val", "test", "unknown"]:
                split_total = sum(counts[split].values())
                if split_total > 0:
                    details = ", ".join([f"{CLASS_NAMES.get(k, str(k))}={counts[split][k]}" for k in sorted(counts[split].keys())])
                    print(f"  {split.upper():7s} : {split_total:6d} -> [{details}]")
                    
        except Exception as e:
            print(f"Error processing {npz}: {e}")

if __name__ == "__main__":
    main()
