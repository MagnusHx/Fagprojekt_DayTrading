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
        transaction_cost=0.001,
        wandb_project="day-trading-experiments",
        wandb_entity="team-entity",
        extra_args=("--no-save-best-checkpoint",),
    )

    assert "--cv-manifest" in cmd
    manifest_arg = cmd[cmd.index("--cv-manifest") + 1]
    assert Path(manifest_arg).name == "sb_L_12_w240_h1_fixedCUSUM0.02_cv_manifest.json"
    assert "--model" in cmd
    assert "conv1d" in cmd
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
    assert "--no-save-best-checkpoint" in cmd
