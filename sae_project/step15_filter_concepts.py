# ==============================================================================
# Step 15: Filter Concepts by Gini Impurity
# - Load class-wise GAP means CSV
# - Filter concepts by Gini impurity threshold (lower = more class-specific)
# - Output CSV with only class-specific concepts, preserving all original info
#
# Usage:
#   python -m sae_project.step15_filter_concepts --input_csv gap_means.csv --max_gini 0.5
# ==============================================================================

import csv
import logging
import os
from typing import Any, Dict, List

from sae_project.step01_configs import get_step15_args

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("filter_concepts")


# ==============================================================================
# Constants
# ==============================================================================
CLASS_COLS = ["Control", "SNCA", "GBA", "LRRK2"]
COUNT_COLS = ["n_Control", "n_SNCA", "n_GBA", "n_LRRK2"]


# ==============================================================================
# Gini Impurity Calculation
# ==============================================================================
def compute_gini_impurity(gaps: List[float], eps: float = 1e-8) -> float:
    """
    Compute Gini impurity from GAP values.

    Gini = 1 - sum(p_i^2) where p_i = GAP_i / sum(GAPs)

    - 0.0 = pure (only one class has non-zero GAP)
    - 0.75 = uniform (all 4 classes have equal GAP)

    Args:
        gaps: List of GAP values for each class
        eps: Small value to avoid division by zero

    Returns:
        Gini impurity value in [0, 0.75] for 4 classes
    """
    total = sum(gaps) + eps
    probs = [g / total for g in gaps]
    gini = 1.0 - sum(p * p for p in probs)
    return gini


def get_max_class(gaps: List[float], class_names: List[str]) -> str:
    """Get the class name with maximum GAP value."""
    max_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    return class_names[max_idx]


# ==============================================================================
# Load and Filter Concepts
# ==============================================================================
def load_and_filter_csv(
    input_csv: str,
    max_gini: float,
    min_active: int,
    alive_only: bool,
    min_max_gap: float,
) -> tuple:
    """
    Load CSV and filter concepts by Gini impurity.

    Args:
        input_csv: Path to input CSV
        max_gini: Maximum Gini impurity (concepts with gini <= max_gini pass)
        min_active: Minimum active images in at least one class
        alive_only: Only include alive concepts
        min_max_gap: Minimum max(GAP) value

    Returns:
        (filtered_rows, all_fieldnames, stats)
    """
    filtered_rows = []
    stats = {
        "total": 0,
        "dead_skipped": 0,
        "low_active_skipped": 0,
        "low_gap_skipped": 0,
        "high_gini_skipped": 0,
        "passed": 0,
        "per_class": {"Control": 0, "SNCA": 0, "GBA": 0, "LRRK2": 0},
    }

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            stats["total"] += 1

            # Check is_alive
            if alive_only:
                is_alive = int(row.get("is_alive", 1))
                if is_alive == 0:
                    stats["dead_skipped"] += 1
                    continue

            # Get GAP values
            gaps = []
            for col in CLASS_COLS:
                try:
                    gaps.append(float(row[col]))
                except (KeyError, ValueError):
                    gaps.append(0.0)

            # Check min_max_gap
            max_gap = max(gaps)
            if max_gap < min_max_gap:
                stats["low_gap_skipped"] += 1
                continue

            # Check min_active (if columns exist)
            if COUNT_COLS[0] in row:
                try:
                    active_counts = [int(row[col]) for col in COUNT_COLS]
                    max_active = max(active_counts)
                except (KeyError, ValueError):
                    max_active = 999
            else:
                max_active = 999 if max_gap > 0.01 else 0

            if max_active < min_active:
                stats["low_active_skipped"] += 1
                continue

            # Compute Gini impurity
            gini = compute_gini_impurity(gaps)

            # Filter by Gini
            if gini > max_gini:
                stats["high_gini_skipped"] += 1
                continue

            # Passed all filters!
            stats["passed"] += 1

            # Add computed fields to row
            row["gini_impurity"] = f"{gini:.6f}"
            row["max_class"] = get_max_class(gaps, CLASS_COLS)
            row["max_gap"] = f"{max_gap:.6f}"

            # Count per max class
            max_class = row["max_class"]
            stats["per_class"][max_class] += 1

            filtered_rows.append(row)

    return filtered_rows, fieldnames, stats


# ==============================================================================
# Sort Results
# ==============================================================================
def sort_rows(rows: List[Dict], sort_by: str) -> List[Dict]:
    """Sort rows by specified field."""
    if sort_by == "gini":
        return sorted(rows, key=lambda r: float(r.get("gini_impurity", 1.0)))
    elif sort_by == "max_gap":
        return sorted(rows, key=lambda r: float(r.get("max_gap", 0.0)), reverse=True)
    elif sort_by == "concept_id":
        return sorted(rows, key=lambda r: int(r.get("concept_id", 0)))
    else:
        return rows


# ==============================================================================
# Save Filtered CSV
# ==============================================================================
def save_filtered_csv(
    output_csv: str,
    rows: List[Dict],
    original_fieldnames: List[str],
    include_all_columns: bool,
) -> None:
    """Save filtered concepts to CSV."""

    if include_all_columns:
        # Use all original columns plus computed ones
        fieldnames = list(original_fieldnames)
        for extra in ["gini_impurity", "max_class", "max_gap"]:
            if extra not in fieldnames:
                fieldnames.append(extra)
    else:
        # Key columns only
        fieldnames = [
            "concept_id",
            "is_alive",
            "Control",
            "SNCA",
            "GBA",
            "LRRK2",
        ]
        # Add count columns if they exist in first row
        if rows and COUNT_COLS[0] in rows[0]:
            fieldnames.extend(COUNT_COLS)
        # Add computed fields
        fieldnames.extend(["gini_impurity", "max_class", "max_gap", "entropy"])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} concepts to: {output_csv}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_step15_args()

    logger.info("=" * 60)
    logger.info("Step 15: Filter Concepts by Gini Impurity")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Max Gini threshold: {args.max_gini}")
    logger.info(f"Min active images: {args.min_active}")
    logger.info(f"Alive only: {args.alive_only}")
    logger.info(f"Min max GAP: {args.min_max_gap}")
    logger.info(f"Sort by: {args.sort_by}")
    logger.info("-" * 60)

    # Validate input
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return

    # Load and filter
    filtered_rows, original_fieldnames, stats = load_and_filter_csv(
        input_csv=args.input_csv,
        max_gini=args.max_gini,
        min_active=args.min_active,
        alive_only=args.alive_only,
        min_max_gap=args.min_max_gap,
    )

    # Sort
    filtered_rows = sort_rows(filtered_rows, args.sort_by)

    # Determine output path
    if args.output_csv == "":
        input_dir = os.path.dirname(args.input_csv)
        input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
        output_csv = os.path.join(
            input_dir, f"{input_basename}_filtered_gini{args.max_gini}.csv"
        )
    else:
        output_csv = args.output_csv

    # Save
    save_filtered_csv(
        output_csv=output_csv,
        rows=filtered_rows,
        original_fieldnames=original_fieldnames,
        include_all_columns=args.include_all_columns,
    )

    # Print statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("FILTERING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total concepts in input:    {stats['total']}")
    logger.info(f"Dead concepts skipped:      {stats['dead_skipped']}")
    logger.info(f"Low active images skipped:  {stats['low_active_skipped']}")
    logger.info(f"Low max GAP skipped:        {stats['low_gap_skipped']}")
    logger.info(f"High Gini (>{args.max_gini}) skipped: {stats['high_gini_skipped']}")
    logger.info("-" * 40)
    logger.info(f"PASSED (class-specific):    {stats['passed']}")
    logger.info("")
    logger.info("Concepts per dominant class:")
    for cls, cnt in stats["per_class"].items():
        logger.info(f"  {cls}: {cnt}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Gini Impurity Reference:")
    logger.info("  0.00 = Pure (only 1 class has GAP > 0)")
    logger.info("  0.50 = Moderately class-specific")
    logger.info("  0.75 = Uniform (all 4 classes equal)")
    logger.info("=" * 60)

    # Print top 10 most class-specific concepts
    if filtered_rows:
        logger.info("")
        logger.info("Top 10 most class-specific concepts:")
        logger.info("-" * 60)
        for i, row in enumerate(filtered_rows[:10]):
            logger.info(
                f"  {row['concept_id']:>4s}: Gini={row['gini_impurity']} "
                f"max_class={row['max_class']:8s} "
                f"gaps=[{row['Control']}, {row['SNCA']}, {row['GBA']}, {row['LRRK2']}]"
            )


if __name__ == "__main__":
    main()
