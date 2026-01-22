# ================================================================
# Args / main
# ================================================================

import os
import argparse


def get_args():
    p = argparse.ArgumentParser("Pointwise Top-K SAE on CNN feature maps")

    # data/model paths
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_root", type=str, default=None, help="default uses DEFAULT_SHARD_ROOT from logging_utils")
    p.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume",
                   help="same save_dir used in contrastive training (contains train_split.csv)")
    p.add_argument("--model_state_path", type=str, default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume/best_model",
                   help="path to best_model.pt or last_model.pt (model_q.state_dict())")
    p.add_argument("--sae_save_dir", type=str, default="",
                   help="where to save SAE ckpt/logs (default: <save_dir>/SAE)")

    # which feature map
    p.add_argument("--which_layer", type=str, default="refine_out",
                   choices=["stage5_out", "refine_out"])

    # encoder architecture (must match training)
    p.add_argument("--blocks", type=str, default="2,3,4,4")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=3)

    # these are only needed to instantiate the same wrapper used in training
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # dataset / loader
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--augment", action="store_true",
                   help="apply rot90 aug before encoder (default off)")
    
    # --- strict balanced batching for SAE ---
    p.add_argument("--strict_plate_balance", action="store_true",
                help="use StrictPlateBalancedBatchSamplerOnBank for train loader")

    # --- token sampling ---
    p.add_argument("--tokens_per_image", type=int, default=2048,
               help="0 => use all H*W tokens per image, else sample this many per image")
    
    # --- token chunking ---
    p.add_argument("--token_batch", type=int, default=8192,
                help="process tokens in chunks of this size inside each image-batch step")
    p.add_argument("--shuffle_tokens", action="store_true",
                help="shuffle token rows before chunking (recommended)")



    # SAE config
    p.add_argument("--d_in", type=int, default=512)
    p.add_argument("--d_sae", type=int, default=4096)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sae_init_scale", type=float, default=0.02)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--sae_wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # optional token normalization
    p.add_argument("--token_l2_norm", action="store_true",
                   help="L2 normalize tokens before SAE")

    # sparsity / regularization
    p.add_argument("--l1_coeff", type=float, default=0.0,
                   help="optional L1 on activations (top-k already sparse)")

    # neuron resampling
    p.add_argument("--usage_ema", type=float, default=0.99)
    p.add_argument("--dead_threshold", type=float, default=1e-6,
                   help="usage_ema below this => dead")
    p.add_argument("--resample_every", type=int, default=500,
                   help="steps interval for resampling")
    p.add_argument("--max_resample_frac", type=float, default=0.05,
                   help="max fraction of features to resample at once")

    # precision
    p.add_argument("--use_bf16", action="store_true")

    # --- validation / test eval ---
    p.add_argument("--use_val", action="store_true",
                help="if set, evaluate on val_split.csv each epoch when exists")
    p.add_argument("--use_test", action="store_true",
                help="if set, evaluate on test_split.csv each epoch when exists (or at end)")

    p.add_argument("--val_batch_size", type=int, default=8,
                help="0 => use same as --batch_size")
    p.add_argument("--test_batch_size", type=int, default=8,
                help="0 => use same as --batch_size")

    p.add_argument("--save_best", action="store_true",
                help="save best ckpt by val_total_loss (or train if no val)")

    # --- FVU logging ---
    p.add_argument("--log_fvu", action="store_true",
                help="compute and log FVU (SSE/SST) for train/val/test")

    # logging/saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)

    return p.parse_args()


def resolve_paths(args):
    # default sae_save_dir
    if args.sae_save_dir == "":
        args.sae_save_dir = os.path.join(args.save_dir, "SAE")
    os.makedirs(args.sae_save_dir, exist_ok=True)

    # shard_root default is stored in logging_utils constants
    if args.shard_root is None:
        from sae_project.step02_logging_utils import DEFAULT_SHARD_ROOT
        args.shard_root = DEFAULT_SHARD_ROOT

    return args
