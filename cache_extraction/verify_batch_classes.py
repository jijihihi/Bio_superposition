import os
import torch
import numpy as np
from sae_project.step04_data_bank import load_all_sample_refs, build_uid_to_refidx
from cache_extraction.extract_features_lambda_labs import make_balanced_loader

class DummyArgs:
    def __init__(self):
        self.shard_root = "/home/ubuntu/model-east3/wds_shards_tar"
        self.save_dir = "/home/ubuntu/model-east3/outputs/MoCo_seed95"
        self.test_only = False
        self.ignore_splits = False
        self.img_size = 128
        self.batch_size = 64
        self.num_workers = 0
        self.seed = 42

def main():
    args = DummyArgs()
    
    print("Loading references...")
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    print("Creating sequential DataLoader (shuffle=False)...")
    loader = make_balanced_loader(
        args, 
        refs, 
        uid_to_refidx, 
        samples_per_class=50,  # Just take a small amount for quick verification
        seed=args.seed,
        include_train=True
    )
    
    print("\n--- Batch Verification (shuffle=False) ---")
    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        
        y_batch = batch[1].numpy()
        unique_classes, counts = np.unique(y_batch, return_counts=True)
        
        class_str = ", ".join([f"Class {cls}: {count}장" for cls, count in zip(unique_classes, counts)])
        print(f"Batch {batch_idx:02d} | Total {len(y_batch):2d}장 | 구성: {class_str}")
        
        if batch_idx >= 10:
            print("...")
            break

if __name__ == "__main__":
    main()
