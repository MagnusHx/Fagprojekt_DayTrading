#!/usr/bin/env python3
"""Simple scikit-learn baseline models for prepared walk-forward folds.

Usage:
    uv run python scripts/simple_baselines.py \\
      --model majority \\
      --cv-manifest <path-to-cv-manifest.json> \\
      --wandb-name E0-majority
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import wandb

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.utils.statistical_tests import calculate_ci
from kvant.ml_framework.wandb_defaults import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, wandb_init_kwargs


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
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.colorbar(im, ax=ax)
    return fig


def _make_estimator(model: str, seed: int) -> BaseEstimator:
    """Create a scikit-learn estimator for a named baseline model."""
    if model == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model == "random":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if model == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=50,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
    raise ValueError(f"Unknown baseline model: {model}")


def _flatten_split(experiment: PreparedExperiment, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened window features and primary side labels for one split."""
    features = []
    labels = []
    for tid, tpos in np.asarray(index, dtype=np.int64):
        x_window, _event_label = experiment.store.window_and_label(int(tid), int(tpos), experiment.L)
        features.append(x_window)
        labels.append(experiment.store.side_label(int(tid), int(tpos)))
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return features.reshape(features.shape[0], -1), labels


def _score_split(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute classification metrics for a split."""
    out = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    n = int(len(y_true))
    for label in (0, 1):
        true_count = int(np.sum(y_true == label))
        pred_count = int(np.sum(y_pred == label))
        out[f"{prefix}_true_side_class_{label}_count"] = true_count
        out[f"{prefix}_true_side_class_{label}_pct"] = float(true_count / n) if n else 0.0
        out[f"{prefix}_pred_side_class_{label}_count"] = pred_count
        out[f"{prefix}_pred_side_class_{label}_pct"] = float(pred_count / n) if n else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(np.int64, copy=False)
    for true_label in (0, 1):
        row_total = int(np.sum(cm[true_label, :]))
        for pred_label in (0, 1):
            count = int(cm[true_label, pred_label])
            out[f"{prefix}_confusion_true{true_label}_pred{pred_label}_count"] = count
            out[f"{prefix}_confusion_true{true_label}_pred{pred_label}_row_pct"] = (
                float(count / row_total) if row_total else 0.0
            )
    return out


def run_sklearn_baseline(manifest: dict, *, model: str, seed: int) -> list[dict[str, float | int | str]]:
    """Run a scikit-learn baseline across all prepared walk-forward folds."""
    results: list[dict[str, float | int | str]] = []

    for fold in manifest["folds"]:
        fold_idx = int(fold["fold_idx"])
        experiment = PreparedExperiment(Path(fold["exp_dir"]))

        train_flat, train_labels = _flatten_split(experiment, experiment.index_train)
        val_flat, val_labels = _flatten_split(experiment, experiment.index_val)
        test_flat, test_labels = _flatten_split(experiment, experiment.index_test)

        clf = _make_estimator(model, seed)
        clf.fit(train_flat, train_labels)

        train_preds = clf.predict(train_flat)
        val_preds = clf.predict(val_flat)
        test_preds = clf.predict(test_flat)

        fold_result = {
            "fold": fold_idx,
            "exp_dir": str(fold["exp_dir"]),
            **_score_split(train_labels, train_preds, "train"),
            **_score_split(val_labels, val_preds, "val"),
            **_score_split(test_labels, test_preds, "test"),
        }

        fig_test = _plot_confusion_matrix(test_labels, test_preds, "test", f"{model} baseline - Fold {fold_idx} - Test")
        wandb.log({f"confusion_matrix/test_fold_{fold_idx}": wandb.Image(fig_test)})
        plt.close(fig_test)

        wandb.log(
            {
                f"fold{fold_idx:02d}/test/accuracy": fold_result["test_accuracy"],
                f"fold{fold_idx:02d}/test/f1_macro": fold_result["test_f1_macro"],
                f"fold{fold_idx:02d}/test/distribution/pred_side/class_0_pct": fold_result[
                    "test_pred_side_class_0_pct"
                ],
                f"fold{fold_idx:02d}/test/distribution/pred_side/class_1_pct": fold_result[
                    "test_pred_side_class_1_pct"
                ],
                f"fold{fold_idx:02d}/test/distribution/true_side/class_0_pct": fold_result[
                    "test_true_side_class_0_pct"
                ],
                f"fold{fold_idx:02d}/test/distribution/true_side/class_1_pct": fold_result[
                    "test_true_side_class_1_pct"
                ],
            }
        )
        results.append(fold_result)

    return results


def main():
    """Run the requested scikit-learn baseline and save fold-level metrics."""
    parser = argparse.ArgumentParser(description="Run simple scikit-learn baseline models on prepared CV artifacts.")
    parser.add_argument(
        "--model",
        choices=["majority", "random", "logreg", "random_forest", "hist_gb"],
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
        help="W&B run name (e.g., E0-majority, E0-logreg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output file for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for stochastic scikit-learn baselines.",
    )
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(
        **wandb_init_kwargs(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "model": args.model,
                "cv_manifest": str(args.cv_manifest),
                "seed": int(args.seed),
            },
        ),
    )

    manifest = load_cv_manifest(args.cv_manifest)
    folds_data = run_sklearn_baseline(manifest, model=args.model, seed=int(args.seed))

    # Aggregate across folds: mean ± std
    metrics_to_agg = [
        "train_accuracy",
        "train_f1_macro",
        "train_precision_macro",
        "train_recall_macro",
        "train_true_side_class_0_pct",
        "train_true_side_class_1_pct",
        "train_pred_side_class_0_pct",
        "train_pred_side_class_1_pct",
        "train_confusion_true0_pred0_count",
        "train_confusion_true0_pred1_count",
        "train_confusion_true1_pred0_count",
        "train_confusion_true1_pred1_count",
        "val_accuracy",
        "val_f1_macro",
        "val_precision_macro",
        "val_recall_macro",
        "val_true_side_class_0_pct",
        "val_true_side_class_1_pct",
        "val_pred_side_class_0_pct",
        "val_pred_side_class_1_pct",
        "val_confusion_true0_pred0_count",
        "val_confusion_true0_pred1_count",
        "val_confusion_true1_pred0_count",
        "val_confusion_true1_pred1_count",
        "test_accuracy",
        "test_f1_macro",
        "test_precision_macro",
        "test_recall_macro",
        "test_true_side_class_0_pct",
        "test_true_side_class_1_pct",
        "test_pred_side_class_0_pct",
        "test_pred_side_class_1_pct",
        "test_confusion_true0_pred0_count",
        "test_confusion_true0_pred1_count",
        "test_confusion_true1_pred0_count",
        "test_confusion_true1_pred1_count",
    ]

    for metric in metrics_to_agg:
        values = [fold[metric] for fold in folds_data]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower, ci_upper = calculate_ci(values, confidence=0.95)

        # Log to W&B
        wandb.log(
            {
                f"{metric}/mean": mean_val,
                f"{metric}/std": std_val,
                f"{metric}/ci_lower": ci_lower,
                f"{metric}/ci_upper": ci_upper,
            }
        )

        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

    # Save to CSV if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(folds_data)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    # Log final summary
    wandb.log({"status": "complete"})
    wandb.finish()

    print("\n✅ Baseline complete.")


if __name__ == "__main__":
    main()
