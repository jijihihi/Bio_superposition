# ================================================================
# Args / main
# ================================================================

import argparse
import os
import sys


def get_args(args_list=None):
    p = argparse.ArgumentParser("Pointwise Top-K SAE on CNN feature maps")

    # data/model paths
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shard_root",
        type=str,
        default=None,
        help="default uses DEFAULT_SHARD_ROOT from logging_utils",
    )
    # default 에 아무것도 안들어오면 DEFAULT_SHARD_ROOT 로 실행되게 내부 알고리즘으로 구현이 되어 있는데 이는 "/content/wds_shards" 즉 코랩 기반. 근데 안 적어주면 오류뜨네. 뭐지. 람다랩스에서 쓸거면 항상 경로 적어줘야해
    p.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image",
        help="same save_dir used in contrastive training (contains train_split.csv)",
    )
    p.add_argument(
        "--model_state_path",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image/best_model.pt",
        help="path to best_model.pt or last_model.pt (model_q.state_dict())",
    )
    p.add_argument(
        "--sae_save_dir",
        type=str,
        default="",
        help="where to save SAE ckpt/logs (default: <save_dir>/SAE)",
    )
    p.add_argument(
        "--eval_ckpt",
        type=str,
        default=None,
        help="Path to an existing SAE checkpoint to evaluate (skips training)",
    )

    # which feature map
    p.add_argument(
        "--which_layer",
        type=str,
        default="refine_out",
        choices=["stage5_mid", "stage5_out", "refine_out"],
    )

    # encoder architecture (must match training)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    # these are only needed to instantiate the same wrapper used in training
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # dataset / loader
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N batches before optimizer step (effective batch = batch_size * N)",
    )
    p.add_argument(
        "--augment",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="apply rot90 aug before encoder (default: True)",
    )
    p.add_argument(
        "--explicit_4x_augment",
        action="store_true",
        help="Use all 4 rotations explicitly (4x data) instead of random rotation",
    )

    # --- strict balanced batching for SAE ---
    p.add_argument(
        "--strict_plate_balance",
        action="store_true",
        help="use StrictPlateBalancedBatchSamplerOnBank for train loader",
    )

    # --- token sampling ---
    p.add_argument(
        "--tokens_per_image",
        type=int,
        default=4096,
        help="0 => use all H*W tokens per image, else sample this many per image",
    )

    # --- token chunking ---
    p.add_argument(
        "--token_batch",
        type=int,
        default=65536,
        help="process tokens in chunks of this size inside each image-batch step",
    )
    p.add_argument(
        "--shuffle_tokens",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="shuffle token rows before chunking (default: True)",
    )

    # SAE config
    p.add_argument("--d_in", type=int, default=512)
    p.add_argument("--d_sae", type=int, default=4096)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sae_init_scale", type=float, default=0.02)
    p.add_argument("--sae_lr", type=float, default=3e-4)
    p.add_argument(
        "--sae_wd",
        type=float,
        default=1e-4,
        help="Weight decay for SAE optimizer (default: 1e-4)",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # LR scheduling
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="LR scheduler: none, cosine (warmup + cosine decay), linear (warmup + linear decay)",
    )
    p.add_argument(
        "--lr_warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps for LR warmup (default: 0.1 = 10%)",
    )

    # optional token normalization
    p.add_argument(
        "--token_l2_norm",
        action="store_true",
        help="L2 normalize tokens before SAE (Old Mode)",
    )
    p.add_argument(
        "--token_norm_mode",
        type=str,
        default="gap-scalar",
        choices=[
            "per-token",
            "gap-scalar",
        ],  # 지금 기본적으로 그 레이어 GAP L2 norm으로 각 그리드 모두 나눠 준 후에 각 토큰도 L2 norm으로 나눠주는데 loss function mse에서 그 토큰 L2 norm 곱해줘서 L2 norm 큰거에 집중하도록 유도.
        help="per-token: old mode, gap-scalar: normalize whole FMap by GAP norm",
    )

    # sparsity / regularization
    p.add_argument(
        "--l1_coeff",
        type=float,
        default=0.0,
        help="optional L1 on activations (top-k already sparse)",
    )

    # ============== Gated SAE Configuration ==============

    # Sparsity warmup
    p.add_argument(
        "--sparsity_warmup_steps",
        type=int,
        default=100,
        help="Steps to keep sparsity coeff at 0 (warmup)",
    )
    p.add_argument(
        "--final_sparsity_coeff",
        type=float,
        default=5.0,
        help="Final sparsity coefficient after warmup",
    )
    p.add_argument(
        "--sparsity_ramp_fraction",
        type=float,
        default=0.1,
        help="Fraction of total_steps for sparsity ramp (default: 0.1 = 10%)",
    )

    # Gated SAE weight tying
    p.add_argument(
        "--tie_gate_weights",
        action="store_true",
        help="Share W_gate = W_mag for parameter efficiency",
    )

    # Clustering initialization
    p.add_argument(
        "--use_clustering_init",
        action="store_true",
        help="Initialize SAE from token clustering centroids",
    )
    p.add_argument(
        "--clustering_init_noise",
        type=float,
        default=0.1,
        help="Noise scale added to clustering initialization",
    )

    # Aux loss coefficient
    p.add_argument(
        "--aux_coeff",
        type=float,
        default=0.1,
        help="Auxiliary loss coefficient for dead neuron prevention",
    )
    p.add_argument(
        "--aux_k",
        type=int,
        default=32,
        help="Number of auxiliary features for aux loss",
    )
    # ============== End Gated SAE Configuration ==============

    # neuron resampling
    p.add_argument("--usage_ema", type=float, default=0.99)
    p.add_argument(
        "--dead_threshold",
        type=float,
        default=5e-4,
        help="usage_ema below this => dead",
    )
    p.add_argument(
        "--resample_every", type=int, default=500, help="steps interval for resampling"
    )
    p.add_argument(
        "--max_resample_frac",
        type=float,
        default=0.05,
        help="max fraction of features to resample at once",
    )

    # precision
    p.add_argument("--use_bf16", action="store_true")

    # --- validation / test eval ---
    p.add_argument(
        "--use_val",
        action="store_true",
        help="if set, evaluate on val_split.csv each epoch when exists",
    )
    p.add_argument(
        "--use_test",
        action="store_true",
        help="if set, evaluate on test_split.csv each epoch when exists (or at end)",
    )

    p.add_argument(
        "--val_batch_size", type=int, default=8, help="0 => use same as --batch_size"
    )
    p.add_argument(
        "--test_batch_size", type=int, default=8, help="0 => use same as --batch_size"
    )

    p.add_argument(
        "--save_best",
        action="store_true",
        help="save best ckpt by val_total_loss (or train if no val)",
    )

    # --- FVU logging ---
    p.add_argument(
        "--log_fvu",
        action="store_true",
        help="compute and log FVU (SSE/SST) for train/val/test",
    )

    # logging/saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)

    # Gated SAE grid search
    p.add_argument(
        "--grid_search",
        action="store_true",
        help="Run automated grid search over sparsity, aux_coeff, tie_weights",
    )
    p.add_argument(
        "--train_all_layers",
        action="store_true",
        help="Train SAE on both stage5_out and refine_out layers",
    )
    p.add_argument(
        "--eval_gap_random",
        action="store_true",
        help="Include GAP@Random baseline in evaluation (on/off)",
    )

    # Grid search parameters
    p.add_argument(
        "--d_sae_grid",
        type=int,
        nargs="+",
        default=[512 * 8],
        help="List of d_sae values for grid search",
    )
    p.add_argument(
        "--sparsity_grid",
        type=float,
        nargs="+",
        default=[5.0],
        help="List of sparsity coefficients for grid search",
    )
    p.add_argument(
        "--aux_coeff_grid",
        type=float,
        nargs="+",
        default=[1 / 32],
        help="List of aux loss coefficients for grid search",
    )
    p.add_argument(
        "--tie_weights_grid",
        type=int,
        nargs="+",
        default=[1],
        help="List of tie_weights (0 or 1) for grid search",
    )

    if "ipykernel" in sys.modules:
        return p.parse_args(
            args_list if args_list is not None else []
        )  # ✅ 주피터/코랩이면 외부 인자 무시
    return p.parse_args()


# ==============================================================================
# Step15: Filter Concepts by Gini Coefficient
# ==============================================================================
def get_step15_args(args_list=None):
    """
    Arguments for step15_filter_concepts.py
    - CSV 경로 입력 → Gini 계수 기준 필터링 → class-specific concepts만 포함된 새 CSV 생성
    """
    p = argparse.ArgumentParser(
        description="Filter SAE concepts by Gini impurity to extract class-specific concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gini impurity <= 0.5인 class-specific concepts만 추출
  python -m sae_project.step15_filter_concepts --input_csv gap_means.csv --max_gini 0.5
  
  # Gini <= 0.3 (더 엄격한 필터), 최소 50개 이미지에서 활성화된 concept만
  python -m sae_project.step15_filter_concepts --input_csv gap_means.csv --max_gini 0.3 --min_active 50
""",
    )

    # Required inputs
    p.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV with class-wise GAP means (output from step09)",
    )

    # Filtering options
    p.add_argument(
        "--max_gini",
        type=float,
        default=0.5,
        help="Maximum Gini impurity threshold (0=pure, 0.75=uniform for 4 classes). Default: 0.5",
    )
    p.add_argument(
        "--min_active",
        type=int,
        default=10,
        help="Minimum number of images where concept must be active in at least one class. Default: 10",
    )
    p.add_argument(
        "--alive_only",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Only include alive (non-dead) concepts. Default: True",
    )

    # Output
    p.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV path. Default: <input_dir>/filtered_gini<max_gini>.csv",
    )
    p.add_argument(
        "--include_all_columns",
        action="store_true",
        help="Include all original columns in output. Default: only key columns",
    )

    # Additional filtering options (optional)
    p.add_argument(
        "--min_max_gap",
        type=float,
        default=0.0,
        help="Minimum max(GAP) value to consider concept significant. Default: 0.0",
    )
    p.add_argument(
        "--sort_by",
        type=str,
        default="gini",
        choices=["gini", "max_gap", "concept_id"],
        help="Sort output by: gini (ascending), max_gap (descending), concept_id. Default: gini",
    )

    if "ipykernel" in sys.modules:
        return p.parse_args(args_list if args_list is not None else [])
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
