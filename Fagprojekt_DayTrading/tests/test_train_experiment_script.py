import argparse

from kvant.ml_framework.scripts.train_experiment import _apply_baseline_preset, _parse_meta_features


def test_baseline_preset_forces_conv1d_and_zero_cost() -> None:
    args = argparse.Namespace(
        baseline=True,
        model="resnet_lstm",
        transaction_cost=0.001,
        wandb_name=None,
    )

    out = _apply_baseline_preset(args)

    assert out.model == "conv1d"
    assert out.transaction_cost == 0.0
    assert out.wandb_name == "baseline-conv1d-cost0"


def test_baseline_preset_keeps_explicit_wandb_name() -> None:
    args = argparse.Namespace(
        baseline=True,
        model="resnet_lstm",
        transaction_cost=0.001,
        wandb_name="my-baseline",
    )

    out = _apply_baseline_preset(args)

    assert out.model == "conv1d"
    assert out.transaction_cost == 0.0
    assert out.wandb_name == "my-baseline"


def test_parse_meta_features_supports_repeated_and_csv_inputs() -> None:
    out = _parse_meta_features(["proba,embedding", "prepared_last:f0"])

    assert out == ("proba", "embedding", "prepared_last:f0")
