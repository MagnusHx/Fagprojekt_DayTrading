from tasks import _baseline_train_command


def test_baseline_train_command_uses_baseline_preset_defaults() -> None:
    cmd = _baseline_train_command()

    assert "--baseline" in cmd
    assert "--epochs 3" in cmd
    assert "--train-batch-size 256" in cmd
    assert "--eval-batch-size 512" in cmd


def test_baseline_train_command_allows_exp_dir_and_extra_args() -> None:
    cmd = _baseline_train_command(
        epochs=5,
        train_batch_size=128,
        eval_batch_size=256,
        exp_dir="prepared/fold00",
        extra_args="--seed 7 --wandb-name test-baseline",
    )

    assert '--exp-dir "prepared/fold00"' in cmd
    assert "--seed 7 --wandb-name test-baseline" in cmd
    assert "--epochs 5" in cmd
