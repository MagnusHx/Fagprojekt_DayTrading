import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvant.labels import SIDE_LABEL_DOWN, label_semantics_payload
from kvant.ml_framework.scripts.train_experiment import (
    _apply_baseline_preset,
    _auto_wandb_name,
    _compatible_default_exp_dir,
    _compatible_default_manifest,
    _default_results_path,
    _device_status_message,
    _fold_result_row,
    _make_lr_scheduler,
    _manifest_results_path,
    _parse_meta_features,
    _validate_primary_side_train_labels,
    _should_run_cv,
    parse_args,
)


def test_recommended_training_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("sys.argv", ["train_experiment", "--exp-dir", str(tmp_path)])

    args = parse_args()

    assert args.epochs == 30
    assert args.lr == 1e-3
    assert args.lr_scheduler == "cosine"
    assert args.min_lr == 1e-5
    assert args.full_eval_every == 3
    assert args.early_stopping_patience is None
    assert args.early_stopping_min_delta == 0.0
    assert args.kelly_fraction == 0.25
    assert args.fixed_bet_size == 1.0
    assert args.portfolio_max_position_fraction == 0.02
    assert args.meta_accept_threshold == 0.5
    assert args.results_out is None


def test_parse_args_can_enable_early_stopping(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train_experiment", "--exp-dir", str(tmp_path), "--early-stopping-patience", "5"],
    )

    args = parse_args()

    assert args.early_stopping_patience == 5


def test_baseline_preset_forces_conv1d_and_zero_cost() -> None:
    args = argparse.Namespace(
        baseline=True,
        model="resnet_lstm",
        transaction_cost=0.0,
        wandb_name=None,
    )

    out = _apply_baseline_preset(args)

    assert out.model == "conv1d"
    assert out.transaction_cost == 0.0
    assert out.wandb_name is None


def test_baseline_preset_keeps_explicit_wandb_name() -> None:
    args = argparse.Namespace(
        baseline=True,
        model="resnet_lstm",
        transaction_cost=0.0,
        wandb_name="my-baseline",
    )

    out = _apply_baseline_preset(args)

    assert out.model == "conv1d"
    assert out.transaction_cost == 0.0
    assert out.wandb_name == "my-baseline"


def test_auto_wandb_name_builds_baseline_cv_name() -> None:
    args = argparse.Namespace(
        wandb_name=None,
        baseline=True,
        model="conv1d",
        cv_manifest="src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_cv_manifest.json",
        exp_dir=None,
        epochs=3,
        transaction_cost=0.0,
        pipeline_stage="primary_side",
    )

    out = _auto_wandb_name(args, run_cv=True)

    assert out == "baseline-sb_L_12_w180_h1.5_TBPD30_cv_manifest-ep3-tc0"


def test_auto_wandb_name_builds_single_fold_name() -> None:
    args = argparse.Namespace(
        wandb_name=None,
        baseline=False,
        model="resnet_lstm",
        cv_manifest=None,
        exp_dir="src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_fold00",
        epochs=10,
        transaction_cost=0.0,
        pipeline_stage="primary_side",
    )

    out = _auto_wandb_name(args, run_cv=False)

    assert out == "resnet_lstm-metaprimary_side-sb_L_12_w180_h1.5_TBPD30_fold00-ep10-tc0"


def test_auto_wandb_name_keeps_explicit_name() -> None:
    args = argparse.Namespace(
        wandb_name="manual-name",
        baseline=True,
        model="conv1d",
        cv_manifest="manifest.json",
        exp_dir=None,
        epochs=3,
        transaction_cost=0.0,
        pipeline_stage="primary_side",
    )

    assert _auto_wandb_name(args, run_cv=True) == "manual-name"


def test_parse_meta_features_supports_repeated_and_csv_inputs() -> None:
    out = _parse_meta_features(["proba,embedding", "prepared_last:f0"])

    assert out == ("proba", "embedding", "prepared_last:f0")


def test_make_lr_scheduler_builds_cosine_scheduler() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = argparse.Namespace(lr_scheduler="cosine", epochs=10, min_lr=1e-5)

    scheduler = _make_lr_scheduler(args, optimizer)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_make_lr_scheduler_can_be_disabled() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = argparse.Namespace(lr_scheduler="none", epochs=10, min_lr=1e-5)

    assert _make_lr_scheduler(args, optimizer) is None


def _write_prepared_config(exp_dir, *, binary: bool) -> None:
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "pipeline_stage": "event_outcome",
                "label_semantics": label_semantics_payload(drop_time_exit_label=binary),
            }
        )
    )


def test_compatible_default_exp_dir_prefers_event_outcome_sibling(tmp_path) -> None:
    binary_dir = tmp_path / "sb_fold00_droptexit"
    event_dir = tmp_path / "sb_fold00"
    _write_prepared_config(binary_dir, binary=True)
    _write_prepared_config(event_dir, binary=False)

    out = _compatible_default_exp_dir(binary_dir)

    assert out == event_dir


def test_compatible_default_manifest_prefers_event_outcome_sibling(tmp_path) -> None:
    binary_dir = tmp_path / "sb_fold00_droptexit"
    event_dir = tmp_path / "sb_fold00"
    _write_prepared_config(binary_dir, binary=True)
    _write_prepared_config(event_dir, binary=False)

    binary_manifest = tmp_path / "exp_droptexit_cv_manifest.json"
    binary_manifest.write_text(json.dumps({"folds": [{"fold_idx": 0, "exp_dir": str(binary_dir)}]}))
    event_manifest = tmp_path / "exp_cv_manifest.json"
    event_manifest.write_text(json.dumps({"folds": [{"fold_idx": 0, "exp_dir": str(event_dir)}]}))

    out = _compatible_default_manifest(binary_manifest)

    assert out == event_manifest


def test_should_run_cv_respects_explicit_exp_dir_over_default_manifest(tmp_path) -> None:
    exp_dir = tmp_path / "sb_fold00"
    manifest = tmp_path / "exp_cv_manifest.json"
    _write_prepared_config(exp_dir, binary=False)
    manifest.write_text(json.dumps({"folds": [{"fold_idx": 0, "exp_dir": str(exp_dir)}]}))

    args = argparse.Namespace(exp_dir=exp_dir, cv_manifest=manifest)

    assert _should_run_cv(args, ["--exp-dir", str(exp_dir)]) is False
    assert _should_run_cv(args, ["--cv-manifest", str(manifest)]) is True
    assert _should_run_cv(args, []) is True


def test_device_status_message_for_cpu() -> None:
    out = _device_status_message(torch.device("cpu"))

    assert out == "Using device: cpu (CUDA not available)"


def test_validate_primary_side_train_labels_rejects_single_effective_class() -> None:
    exp = SimpleNamespace(
        index_train=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        store=SimpleNamespace(
            side_labels_for_index=lambda index: np.full(len(index), SIDE_LABEL_DOWN, dtype=np.int64),
        ),
    )

    with pytest.raises(RuntimeError, match="requires both down and up labels"):
        _validate_primary_side_train_labels(exp)


def test_default_results_path_uses_wandb_name() -> None:
    args = argparse.Namespace(wandb_name="E1-timebar")

    out = _default_results_path(args, None)

    assert str(out) == "results/E1-timebar.csv"


def test_manifest_results_path_strips_cv_manifest_suffix() -> None:
    out = _manifest_results_path(Path("src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json"))

    assert str(out) == "results/E1_timebar.csv"


def test_fold_result_row_extracts_comparison_metrics() -> None:
    row = _fold_result_row(
        2,
        {
            "test/classification/accuracy": 0.51,
            "test/meta/f1": 0.42,
            "test/portfolio/total_return_pct": -0.1,
            "test/not_exported": 123,
        },
    )

    assert row["fold"] == 2
    assert row["test_accuracy"] == 0.51
    assert row["test_meta_f1"] == 0.42
    assert row["test_portfolio_total_return_pct"] == -0.1
    assert "test/not_exported" not in row
