#!/usr/bin/env python3
"""
Simple baseline models: majority class and logistic regression.

These serve as E0 (floor) for validating that deep learning models beat trivial baselines.

Usage:
    uv run python scripts/simple_baselines.py \\
      --model majority \\
      --prepared-data-dir src/kvant/ml_framework/prepared \\
      --cv-manifest <path-to-cv-manifest.json> \\
      --wandb-name E0-majority
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.utils.statistical_tests import calculate_ci, format_ci


def load_cv_manifest(manifest_path: Path) -> dict:
    """Load cross-validation manifest."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, split: str, title: str) -> plt.Figure:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(["Down", "Up"])
    ax.set_yticklabels(["Down", "Up"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im, ax=ax)
    return fig


def run_majority_class_baseline(experiment: PreparedExperiment) -> dict:
    """
    Majority class baseline: always predict the most common class.
    """
    results = {}

    for fold_idx, (train_set, val_set, test_set) in enumerate(
        zip(experiment.train_folds, experiment.val_folds, experiment.test_folds)
    ):
        train_labels = np.array([int(experiment._labels_primary[i]) for i in train_set])
        val_labels = np.array([int(experiment._labels_primary[i]) for i in val_set])
        test_labels = np.array([int(experiment._labels_primary[i]) for i in test_set])

        # Majority class is the most common class in training
        unique, counts = np.unique(train_labels, return_counts=True)
        majority_class = unique[np.argmax(counts)]

        # Predict majority class for all examples
        train_preds = np.full_like(train_labels, majority_class)
        val_preds = np.full_like(val_labels, majority_class)
        test_preds = np.full_like(test_labels, majority_class)

        # Evaluate
        fold_result = {
            "fold": fold_idx,
            "train_accuracy": float(accuracy_score(train_labels, train_preds)),
            "train_f1_macro": float(f1_score(train_labels, train_preds, average="macro", zero_division=0)),
            "val_accuracy": float(accuracy_score(val_labels, val_preds)),
            "val_f1_macro": float(f1_score(val_labels, val_preds, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(test_labels, test_preds)),
            "test_f1_macro": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
        }

        # Log confusion matrices to W&B
        fig_test = _plot_confusion_matrix(test_labels, test_preds, "test", f"Majority class - Fold {fold_idx} - Test")
        wandb.log({f"confusion_matrix/test_fold_{fold_idx}": wandb.Image(fig_test)})
        plt.close(fig_test)

        results[f"fold_{fold_idx}"] = fold_result

    return results


def run_logistic_regression_baseline(experiment: PreparedExperiment) -> dict:
    """
    Logistic regression baseline: train on flattened feature windows.
    """
    results = {}

    for fold_idx, (train_set, val_set, test_set) in enumerate(
        zip(experiment.train_folds, experiment.val_folds, experiment.test_folds)
    ):
        # Load features and labels
        train_features = np.array([experiment.features[i] for i in train_set])
        val_features = np.array([experiment.features[i] for i in val_set])
        test_features = np.array([experiment.features[i] for i in test_set])

        train_labels = np.array([int(experiment._labels_primary[i]) for i in train_set])
        val_labels = np.array([int(experiment._labels_primary[i]) for i in val_set])
        test_labels = np.array([int(experiment._labels_primary[i]) for i in test_set])

        # Flatten feature windows (L, F) -> (L*F,)
        train_flat = train_features.reshape(train_features.shape[0], -1)
        val_flat = val_features.reshape(val_features.shape[0], -1)
        test_flat = test_features.reshape(test_features.shape[0], -1)

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=1337, n_jobs=-1)
        clf.fit(train_flat, train_labels)

        # Predict
        train_preds = clf.predict(train_flat)
        val_preds = clf.predict(val_flat)
        test_preds = clf.predict(test_flat)

        # Evaluate
        fold_result = {
            "fold": fold_idx,
            "train_accuracy": float(accuracy_score(train_labels, train_preds)),
            "train_f1_macro": float(f1_score(train_labels, train_preds, average="macro", zero_division=0)),
            "val_accuracy": float(accuracy_score(val_labels, val_preds)),
            "val_f1_macro": float(f1_score(val_labels, val_preds, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(test_labels, test_preds)),
            "test_f1_macro": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
        }

        # Log confusion matrices to W&B
        fig_test = _plot_confusion_matrix(test_labels, test_preds, "test", f"Logistic Regression - Fold {fold_idx} - Test")
        wandb.log({f"confusion_matrix/test_fold_{fold_idx}": wandb.Image(fig_test)})
        plt.close(fig_test)

        results[f"fold_{fold_idx}"] = fold_result

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run simple baseline models (majority class, logistic regression)."
    )
    parser.add_argument(
        "--model",
        choices=["majority", "logreg"],
        required=True,
        help="Which baseline model to run.",
    )
    parser.add_argument(
        "--cv-manifest",
        type=Path,
        required=True,
        help="Path to CV manifest JSON from prepare_experiment.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="day-trading-experiments",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        required=True,
        help="W&B run name (e.g., E0-majority, E0-logreg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output file for results.",
    )
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "model": args.model,
            "cv_manifest": str(args.cv_manifest),
        },
    )

    # Load prepared experiment
    manifest = load_cv_manifest(args.cv_manifest)
    exp_dirs = [fold["exp_dir"] for fold in manifest["folds"]]
    experiment = PreparedExperiment.load_cv(*exp_dirs)

    # Run baseline
    if args.model == "majority":
        results = run_majority_class_baseline(experiment)
    else:  # logreg
        results = run_logistic_regression_baseline(experiment)

    # Compute fold statistics
    folds_data = []
    for fold_key, fold_result in results.items():
        folds_data.append(fold_result)

    # Aggregate across folds: mean ± std
    metrics_to_agg = [
        "train_accuracy",
        "train_f1_macro",
        "val_accuracy",
        "val_f1_macro",
        "test_accuracy",
        "test_f1_macro",
    ]

    for metric in metrics_to_agg:
        values = [fold[metric] for fold in folds_data]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower, ci_upper = calculate_ci(values, confidence=0.95)

        # Log to W&B
        wandb.log({
            f"{metric}/mean": mean_val,
            f"{metric}/std": std_val,
            f"{metric}/ci_lower": ci_lower,
            f"{metric}/ci_upper": ci_upper,
        })

        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

    # Save to CSV if requested
    if args.output:
        df = pd.DataFrame(folds_data)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    # Log final summary
    wandb.log({"status": "complete"})
    wandb.finish()

    print("\n✅ Baseline complete.")


if __name__ == "__main__":
    main()
