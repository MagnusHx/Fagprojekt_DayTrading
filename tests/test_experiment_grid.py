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

    assert len(configs) == 9
    assert GridConfig(cusum_h=0.01, barrier_height=0.025, barrier_width_periods=24) in configs
    assert GridConfig(cusum_h=0.03, barrier_height=0.06, barrier_width_periods=24) in configs


def test_grid_config_matches_prepare_experiment_label_convention() -> None:
    """Grid labels should match the manifest names produced by prepare_experiment.py."""
    config = GridConfig(cusum_h=0.02, barrier_height=0.05, barrier_width_periods=24)

    assert config.barrier_height_pct == 5.0
    assert config.label == "sb_L_96_wp24_h5_fixedCUSUM0.02"
    assert config.manifest_path.name == "sb_L_96_wp24_h5_fixedCUSUM0.02_cv_manifest.json"


def test_prepare_command_uses_decimal_barrier_height_as_percent() -> None:
    """The user-facing decimal barrier height must be converted to the preparation CLI percent unit."""
    config = GridConfig(cusum_h=0.02, barrier_height=0.05, barrier_width_periods=24)

    cmd = prepare_command(config)

    assert cmd == [
        "uv",
        "run",
        "python",
        "-m",
        "kvant.ml_prepare_data.prepare_experiment",
        "--sampler",
        "fixed_cusum",
        "--cusum-h",
        "0.02",
        "--lookback",
        "96",
        "--barrier-width-periods",
        "24",
        "--barrier-height-pct",
        "5",
    ]


def test_train_command_builds_conv1d_threshold_run() -> None:
    """Training commands should target the expected CV manifest and W&B run name."""
    config = GridConfig(cusum_h=0.03, barrier_height=0.06, barrier_width_periods=24)
    run = GridRun(config=config, model="conv1d", meta_threshold=0.55)

    cmd = train_command(
        run,
        epochs=20,
        transaction_cost=0.001,
        wandb_project="Kvant",
        extra_args=("--no-save-best-checkpoint",),
    )

    assert "--cv-manifest" in cmd
    manifest_arg = cmd[cmd.index("--cv-manifest") + 1]
    assert Path(manifest_arg).name == "sb_L_96_wp24_h6_fixedCUSUM0.03_cv_manifest.json"
    assert "--model" in cmd
    assert "conv1d" in cmd
    assert "--meta-accept-threshold" in cmd
    assert "0.55" in cmd
    assert "--wandb-name" in cmd
    assert "grid-conv1d-wp24-bh0p06-ch0p03-mt0p55" in cmd
    assert "--no-save-best-checkpoint" in cmd
