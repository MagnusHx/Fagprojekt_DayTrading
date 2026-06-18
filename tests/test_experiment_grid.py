from pathlib import Path

from kvant.ml_framework.scripts.run_experiment_grid import (
    GridConfig,
    GridRun,
    iter_grid_configs,
    prepare_command,
    train_command,
)


def test_grid_contains_expected_number_of_data_configs() -> None:
    """The calibration grid should cover all CUSUM, barrier-height, and width combinations."""
    configs = list(iter_grid_configs())

    assert len(configs) == 12
    assert GridConfig(cusum_h=0.01, barrier_height=0.01, barrier_width=240) in configs
    assert GridConfig(cusum_h=0.03, barrier_height=0.06, barrier_width=240) in configs


def test_grid_config_matches_prepare_experiment_label_convention() -> None:
    """Grid labels should match the manifest names produced by prepare_experiment.py."""
    config = GridConfig(cusum_h=0.01, barrier_height=0.01, barrier_width=240)

    assert config.barrier_height_pct == 1.0
    assert config.label == "sb_L_12_w240_h1_fixedCUSUM0.01"
    assert config.manifest_path.name == "sb_L_12_w240_h1_fixedCUSUM0.01_cv_manifest.json"


def test_prepare_command_uses_decimal_barrier_height_as_percent() -> None:
    """The user-facing decimal barrier height must be converted to the preparation CLI percent unit."""
    config = GridConfig(cusum_h=0.02, barrier_height=0.04, barrier_width=240)

    cmd = prepare_command(config)

    assert cmd[:14] == [
        "uv",
        "run",
        "python",
        "-m",
        "kvant.ml_prepare_data.prepare_experiment",
        "--sampler",
        "fixed_cusum",
        "--cusum-h",
        "0.02",
        "--barrier-width",
        "240",
        "--barrier-height-pct",
        "4",
        "--cv-manifest",
    ]
    assert Path(cmd[14]).name == "sb_L_12_w240_h4_fixedCUSUM0.02_cv_manifest.json"


def test_train_command_builds_conv1d_threshold_run() -> None:
    """Training commands should target the expected CV manifest and W&B run name."""
    config = GridConfig(cusum_h=0.02, barrier_height=0.01, barrier_width=240)
    run = GridRun(config=config, model="conv1d")

    cmd = train_command(
        run,
        epochs=20,
        transaction_cost=0.0,
        wandb_project="day-trading-experiments",
        wandb_entity="team-entity",
        extra_args=("--no-save-best-checkpoint",),
    )

    assert "--cv-manifest" in cmd
    manifest_arg = cmd[cmd.index("--cv-manifest") + 1]
    assert Path(manifest_arg).name == "sb_L_12_w240_h1_fixedCUSUM0.02_cv_manifest.json"
    assert "--model" in cmd
    assert "conv1d" in cmd
    assert "--bet-sizing" in cmd
    assert cmd[cmd.index("--bet-sizing") + 1] == "fixed"
    assert "--no-meta" in cmd
    assert "--fixed-bet-size" in cmd
    assert cmd[cmd.index("--fixed-bet-size") + 1] == "1"
    assert "--results-out" in cmd
    assert Path(cmd[cmd.index("--results-out") + 1]).name == "E2-grid-conv1d-w240-tb1-cusum2-nometa.csv"
    assert "--wandb-name" in cmd
    assert "E2-grid-conv1d-w240-tb1-cusum2-nometa" in cmd
    assert "--wandb-project" in cmd
    assert "day-trading-experiments" in cmd
    assert "--wandb-entity" in cmd
    assert "team-entity" in cmd
    assert "--transaction-cost" in cmd
    assert cmd[cmd.index("--transaction-cost") + 1] == "0"
    assert "--no-save-best-checkpoint" in cmd


def test_train_command_builds_resnet_confidence_run() -> None:
    """Confidence sweep runs should be distinguishable from the no-meta ResNet baseline."""
    config = GridConfig(cusum_h=0.01, barrier_height=0.02, barrier_width=240)
    run = GridRun(
        config=config,
        model="resnet_lstm",
        no_meta=True,
        primary_confidence_threshold=0.55,
    )

    cmd = train_command(
        run,
        epochs=30,
        transaction_cost=0.001,
        wandb_project="day-trading-experiments",
        wandb_entity="team-entity",
    )

    assert cmd[cmd.index("--model") + 1] == "resnet_lstm"
    assert cmd[cmd.index("--epochs") + 1] == "30"
    assert "--no-meta" in cmd
    assert "--primary-confidence-threshold" in cmd
    assert cmd[cmd.index("--primary-confidence-threshold") + 1] == "0.55"
    assert Path(cmd[cmd.index("--results-out") + 1]).name == (
        "E2-grid-resnet_lstm-w240-tb2-cusum1-nometa-ct0p55.csv"
    )


def test_train_command_builds_resnet_meta_run() -> None:
    """Meta sweep runs should use the selected ResNet-LSTM architecture."""
    config = GridConfig(cusum_h=0.01, barrier_height=0.02, barrier_width=240)
    run = GridRun(config=config, model="resnet_lstm", no_meta=False, meta_threshold=0.6)

    cmd = train_command(
        run,
        epochs=30,
        transaction_cost=0.001,
        wandb_project="day-trading-experiments",
        wandb_entity="team-entity",
    )

    assert cmd[cmd.index("--model") + 1] == "resnet_lstm"
    assert cmd[cmd.index("--epochs") + 1] == "30"
    assert "--no-meta" not in cmd
    assert "--meta-accept-threshold" in cmd
    assert cmd[cmd.index("--meta-accept-threshold") + 1] == "0.6"
    assert Path(cmd[cmd.index("--results-out") + 1]).name == "E2-grid-resnet_lstm-w240-tb2-cusum1-mt0p6.csv"
