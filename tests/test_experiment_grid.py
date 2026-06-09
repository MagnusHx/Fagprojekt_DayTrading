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

    assert len(configs) == 27
    assert GridConfig(cusum_h=0.005, barrier_height=0.005, barrier_width=60) in configs
    assert GridConfig(cusum_h=0.02, barrier_height=0.015, barrier_width=180) in configs


def test_grid_config_matches_prepare_experiment_label_convention() -> None:
    """Grid labels should match the manifest names produced by prepare_experiment.py."""
    config = GridConfig(cusum_h=0.005, barrier_height=0.005, barrier_width=60)

    assert config.barrier_height_pct == 0.5
    assert config.label == "sb_L_12_w60_h0.5_fixedCUSUM0.005"
    assert config.manifest_path.name == "sb_L_12_w60_h0.5_fixedCUSUM0.005_cv_manifest.json"


def test_prepare_command_uses_decimal_barrier_height_as_percent() -> None:
    """The user-facing decimal barrier height must be converted to the preparation CLI percent unit."""
    config = GridConfig(cusum_h=0.01, barrier_height=0.015, barrier_width=180)

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
        "0.01",
        "--barrier-width",
        "180",
        "--barrier-height-pct",
        "1.5",
    ]


def test_train_command_builds_conv1d_threshold_run() -> None:
    """Training commands should target the expected CV manifest and W&B run name."""
    config = GridConfig(cusum_h=0.02, barrier_height=0.01, barrier_width=120)
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
    assert Path(manifest_arg).name == "sb_L_12_w120_h1_fixedCUSUM0.02_cv_manifest.json"
    assert "--model" in cmd
    assert "conv1d" in cmd
    assert "--meta-accept-threshold" in cmd
    assert "0.55" in cmd
    assert "--wandb-name" in cmd
    assert "grid-conv1d-w120-bh0p01-ch0p02-mt0p55" in cmd
    assert "--no-save-best-checkpoint" in cmd
