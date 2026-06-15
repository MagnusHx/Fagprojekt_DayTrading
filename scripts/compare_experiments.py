#!/usr/bin/env python3
"""
Compare two sets of experimental results with statistical tests.

This script performs paired t-tests across folds to determine if differences
between experiments are statistically significant.

Usage:
    # Compare E1-timebar vs E1-cusum
    uv run python scripts/compare_experiments.py \\
      --results-a results/E1_timebar.csv \\
      --results-b results/E1_cusum.csv \\
      --name-a "E1-timebar" \\
      --name-b "E1-cusum" \\
      --metrics test_accuracy test_f1_macro test_sharpe
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import wandb

from kvant.ml_framework.utils.statistical_tests import (
    paired_ttest,
    format_ttest_result,
    calculate_ci,
    format_ci,
)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results CSV and sort by fold."""
    df = pd.read_csv(csv_path)
    if "fold" in df.columns:
        df = df.sort_values("fold")
    return df


def compare_metrics(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    metrics: list[str],
    name_a: str,
    name_b: str,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> dict:
    """
    Compare two sets of results with paired t-tests.

    Args:
        results_a: Results dataframe for model A
        results_b: Results dataframe for model B
        metrics: List of metric column names to compare
        name_a: Display name for model A
        name_b: Display name for model B
        wandb_project: Optional W&B project for logging
        wandb_name: Optional W&B run name

    Returns:
        dict mapping metric_name -> ttest_result
    """
    # Initialize W&B if specified
    if wandb_project and wandb_name:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "model_a": name_a,
                "model_b": name_b,
                "metrics": metrics,
            },
        )

    results = {}

    print(f"\n{'='*80}")
    print(f"Comparing {name_a} vs {name_b}")
    print(f"{'='*80}\n")

    for metric in metrics:
        if metric not in results_a.columns or metric not in results_b.columns:
            print(f"⚠️  Metric '{metric}' not found in both result sets")
            continue

        values_a = results_a[metric].values
        values_b = results_b[metric].values

        if len(values_a) != len(values_b):
            print(f"⚠️  Unequal fold counts for '{metric}': {len(values_a)} vs {len(values_b)}")
            continue

        # Paired t-test
        ttest_result = paired_ttest(values_a, values_b, alternative="two-sided")

        # Also compute per-model stats for display
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        ci_a_lower, ci_a_upper = calculate_ci(values_a)
        ci_b_lower, ci_b_upper = calculate_ci(values_b)

        results[metric] = ttest_result

        # Print results
        print(f"\n{metric}:")
        print(f"  {name_a}: {format_ci(mean_a, ci_a_lower, ci_a_upper)}")
        print(f"  {name_b}: {format_ci(mean_b, ci_b_lower, ci_b_upper)}")
        print(format_ttest_result(ttest_result, f"  {name_a} vs {name_b}"))

        # Log to W&B if enabled
        if wandb_project and wandb_name:
            wandb.log({
                f"{metric}/{name_a}/mean": mean_a,
                f"{metric}/{name_a}/ci_lower": ci_a_lower,
                f"{metric}/{name_a}/ci_upper": ci_a_upper,
                f"{metric}/{name_b}/mean": mean_b,
                f"{metric}/{name_b}/ci_lower": ci_b_lower,
                f"{metric}/{name_b}/ci_upper": ci_b_upper,
                f"{metric}/ttest_p_value": ttest_result["p_value"],
                f"{metric}/ttest_mean_diff": ttest_result["mean_diff"],
                f"{metric}/ttest_significant": ttest_result["significant"],
            })

    print(f"\n{'='*80}")

    # Finish W&B if started
    if wandb_project and wandb_name:
        wandb.log({"status": "complete"})
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare two experimental results with statistical tests."
    )
    parser.add_argument(
        "--results-a",
        type=Path,
        required=True,
        help="Path to results CSV for model A",
    )
    parser.add_argument(
        "--results-b",
        type=Path,
        required=True,
        help="Path to results CSV for model B",
    )
    parser.add_argument(
        "--name-a",
        type=str,
        required=True,
        help="Display name for model A (e.g., E1-timebar)",
    )
    parser.add_argument(
        "--name-b",
        type=str,
        required=True,
        help="Display name for model B (e.g., E1-cusum)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Metric names to compare (e.g., test_accuracy test_f1_macro)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional W&B project for logging comparison results",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional W&B run name for comparison",
    )
    args = parser.parse_args()

    # Load results
    results_a = load_results(args.results_a)
    results_b = load_results(args.results_b)

    # Compare
    compare_metrics(
        results_a,
        results_b,
        args.metrics,
        args.name_a,
        args.name_b,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    print("\n✅ Comparison complete.")


if __name__ == "__main__":
    main()
