# !python -m trajectory_inference_pipeline.aggregate_figures \
# --features_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/CNN_seed445_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz" \
# --cell_death_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#  --vis_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/pairwise_phate" \
#  --dpt_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/DPT_445" \
#  --stats_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/DPT_445"

import argparse
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import add_trajectory_arguments, get_logger

logger = get_logger("aggregate_figures")


def get_args():
    p = argparse.ArgumentParser(
        description="Aggregate generated trajectory figures into a single panel."
    )
    p = add_trajectory_arguments(p)
    p.add_argument(
        "--vis_dir", type=str, default="", help="Directory containing PHATE/PAGA pngs"
    )
    p.add_argument(
        "--dpt_dir", type=str, default="", help="Directory containing DPT Scatter pngs"
    )
    p.add_argument(
        "--stats_dir",
        type=str,
        default="",
        help="Directory containing Terciles Bar pngs",
    )
    return p.parse_args()


def run_aggregate_figures(args):
    mutations = ["SNCA", "GBA", "LRRK2"]
    plot_types = ["PHATE", "PAGA", "DPT Scatter", "cell_death by Stage"]

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.dirname(args.features_cache)

    out_dir = os.path.join(base_dir, "aggregated_figures")
    os.makedirs(out_dir, exist_ok=True)

    # Paths for sub-modules
    vis_dir = (
        args.vis_dir
        if args.vis_dir
        else (
            os.path.join(base_dir, "pairwise_vis") if not args.output_dir else base_dir
        )
    )
    dpt_dir = (
        args.dpt_dir
        if args.dpt_dir
        else (
            os.path.join(base_dir, "pairwise_dpt") if not args.output_dir else base_dir
        )
    )
    stats_dir = (
        args.stats_dir
        if args.stats_dir
        else (
            os.path.join(base_dir, "downstream_stats")
            if not args.output_dir
            else base_dir
        )
    )

    # If the user specified a single output_dir for everything, all files might be there.
    # Otherwise they are in module-specific subdirectories.
    def get_img_path(plot_type, mut):
        prefix = (
            f"{args.norm}_{args.which_layer}_{mut}"
            if hasattr(args, "which_layer")
            else f"{args.norm}_none_{mut}"
        )
        # Fallback if which_layer is not available in args directly but we can try globbing
        # Wait, the pipeline scripts all extract which_layer from the cache.
        # Let's load it just to be safe.
        return ""

    import numpy as np

    data = np.load(args.features_cache, allow_pickle=True)
    which_layer = str(data["which_layer"]) if "which_layer" in data else "unknown"

    # Matplotlib setup
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for i, mut in enumerate(mutations):
        paths = [
            os.path.join(vis_dir, f"phate_{args.norm}_{which_layer}_{mut}.png"),
            os.path.join(vis_dir, f"paga_{args.norm}_{which_layer}_{mut}.png"),
            os.path.join(dpt_dir, f"dpt_scatter_{args.norm}_{which_layer}_{mut}.png"),
            os.path.join(stats_dir, f"terciles_jt_{args.norm}_{which_layer}_{mut}.png"),
        ]

        for j, (plot_type, img_path) in enumerate(zip(plot_types, paths)):
            ax = axes[i, j]
            ax.axis("off")

            if j == 0:
                ax.text(
                    -0.1,
                    0.5,
                    mut,
                    fontsize=20,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    rotation=90,
                )
            if i == 0:
                ax.set_title(plot_type, fontsize=18, fontweight="bold", pad=15)

            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing:\n{os.path.basename(img_path)}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )
                logger.warning(f"Missing image: {img_path}")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    out_png = os.path.join(out_dir, f"figure_summary_{args.norm}_{which_layer}.png")
    out_pdf = os.path.join(out_dir, f"figure_summary_{args.norm}_{which_layer}.pdf")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Successfully created aggregated figure panel:")
    logger.info(f"  PNG: {out_png}")
    logger.info(f"  PDF: {out_pdf}")


if __name__ == "__main__":
    args = get_args()
    if not args.norm:
        args.norm = "log_std"
    run_aggregate_figures(args)
