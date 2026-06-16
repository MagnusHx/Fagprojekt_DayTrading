#!/usr/bin/env python3
"""
Threshold sweep at evaluation time: evaluate a saved checkpoint at different meta-acceptance thresholds.

This allows exploring the confidence-vs-Sharpe tradeoff (E4) without retraining.

Usage:
    uv run python scripts/reconcile_metrics.py \\
      --checkpoint artifacts/E3_conv1d-best.ckpt.pt \\
      --thresholds 0.0 0.55 0.65 \\
      --wandb-name E4-threshold-sweep
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.train import ExperimentEvaluator, EvalConfig
from kvant.ml_framework.utils.statistical_tests import calculate_ci
from kvant.ml_framework.wandb_defaults import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, wandb_init_kwargs


def _plot_threshold_tradeoff(results: dict) -> plt.Figure:
    """
    Plot frequency vs Sharpe ratio across thresholds.
    RQ3 figure: shows the tradeoff between trade frequency and risk-adjusted returns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)

    thresholds = []
    n_trades = []
    sharpe_ratios = []
    hit_rates = []

    for threshold_key, fold_results in sorted(results.items()):
        threshold = float(threshold_key.split("_")[-1])
        # Aggregate across folds
        trades = [fold.get("n_trades", np.nan) for fold in fold_results.values()]
        sharpes = [fold.get("sharpe_ratio", np.nan) for fold in fold_results.values()]
        hits = [fold.get("hit_rate", np.nan) for fold in fold_results.values()]

        thresholds.append(threshold)
        n_trades.append(np.nanmean(trades))
        sharpe_ratios.append(np.nanmean(sharpes))
        hit_rates.append(np.nanmean(hits))

    # Plot 1: Frequency vs Sharpe
    axes[0].plot(thresholds, n_trades, "o-", linewidth=2, markersize=8, label="Avg trades")
    axes[0].set_xlabel("Meta-acceptance threshold", fontsize=11)
    axes[0].set_ylabel("Avg n_trades per fold", fontsize=11)
    axes[0].set_title("E4: Trade Frequency vs Threshold", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Sharpe vs Threshold
    ax2 = axes[1]
    ax2.plot(thresholds, sharpe_ratios, "o-", color="green", linewidth=2, markersize=8, label="Sharpe ratio")
    ax2.set_xlabel("Meta-acceptance threshold", fontsize=11)
    ax2.set_ylabel("Sharpe ratio (annualized)", fontsize=11)
    ax2.set_title("E4: Risk-adjusted Returns vs Threshold", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def load_checkpoint_bundle(checkpoint_path: Path) -> dict:
    """Load a checkpoint bundle saved by train_experiment.py."""
    bundle = torch.load(checkpoint_path, map_location="cpu")
    return bundle


def evaluate_checkpoint_at_thresholds(
    checkpoint_path: Path,
    thresholds: list[float],
    wandb_name: str,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
) -> dict:
    """
    Evaluate a checkpoint at different meta-acceptance thresholds.

    For each threshold, the evaluator re-evaluates the saved predictions
    with a new threshold, producing trade-level economics without retraining.
    """
    bundle = load_checkpoint_bundle(checkpoint_path)

    exp_dir = Path(bundle["exp_dir"])
    model_state = bundle["model_state"]
    model_name = bundle["model_name"]
    model_kwargs = bundle["model_kwargs"]
    eval_cfg_dict = bundle.get("eval_config", {})

    # Load prepared experiment
    exp = PreparedExperiment(exp_dir)

    results = {}

    for threshold in thresholds:
        # Update threshold in eval config
        eval_cfg_dict["meta_accept_threshold"] = float(threshold)
        eval_cfg = EvalConfig(**eval_cfg_dict)

        # Create evaluator
        evaluator = ExperimentEvaluator(
            exp=exp,
            model_state=model_state,
            model_name=model_name,
            model_kwargs=model_kwargs,
            eval_cfg=eval_cfg,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Evaluate all folds
        fold_results = evaluator.evaluate_cv(verbose=True)

        results[f"threshold_{threshold}"] = fold_results

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep meta-acceptance thresholds on a saved checkpoint (E4)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best-checkpoint bundle (*.ckpt.pt).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="Thresholds to sweep (e.g., 0.0 0.55 0.65).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help="W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        required=True,
        help="W&B run name (e.g., E4-threshold-sweep).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output file for aggregated results.",
    )
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(
        **wandb_init_kwargs(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "checkpoint": str(args.checkpoint),
                "thresholds": args.thresholds,
            },
        ),
    )

    # Evaluate at all thresholds
    results = evaluate_checkpoint_at_thresholds(
        args.checkpoint,
        thresholds=args.thresholds,
        wandb_name=args.wandb_name,
        wandb_project=args.wandb_project,
    )

    # Generate and log threshold tradeoff figure (RQ3 figure)
    fig_tradeoff = _plot_threshold_tradeoff(results)
    wandb.log({"figures/threshold_tradeoff": wandb.Image(fig_tradeoff)})
    plt.close(fig_tradeoff)

    # Log results to W&B and print summary
    for threshold_key, fold_results in results.items():
        threshold = float(threshold_key.split("_")[-1])

        # Aggregate across folds
        metrics_to_agg = [
            "n_trades",
            "hit_rate",
            "avg_return_per_trade",
            "sharpe_ratio",
            "max_drawdown",
            "cumulative_return",
        ]

        summary = {}
        for metric in metrics_to_agg:
            values = [fold.get(metric, np.nan) for fold in fold_results.values()]
            values = values[~np.isnan(values)]  # Filter out NaNs for CI calculation
            if len(values) > 0:
                mean_val = float(np.nanmean(values))
                std_val = float(np.nanstd(values))
                ci_lower, ci_upper = calculate_ci(values, confidence=0.95)
                summary[metric] = {"mean": mean_val, "std": std_val, "ci_lower": ci_lower, "ci_upper": ci_upper}

                wandb.log({
                    f"{threshold_key}/{metric}/mean": mean_val,
                    f"{threshold_key}/{metric}/std": std_val,
                    f"{threshold_key}/{metric}/ci_lower": ci_lower,
                    f"{threshold_key}/{metric}/ci_upper": ci_upper,
                })

        print(f"\n=== Threshold {threshold} ===")
        for metric, stats in summary.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}  [95% CI: {stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

    # Save results to CSV if requested
    if args.output:
        df_list = []
        for threshold_key, fold_results in results.items():
            threshold = float(threshold_key.split("_")[-1])
            for fold_id, fold_metrics in fold_results.items():
                row = {"threshold": threshold, "fold": fold_id}
                row.update(fold_metrics)
                df_list.append(row)
        df = pd.DataFrame(df_list)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    wandb.log({"status": "complete"})
    wandb.finish()

    print("\n✅ Threshold sweep complete.")


if __name__ == "__main__":
    main()
