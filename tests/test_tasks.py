import pytest

from tasks import TRAINING_PRESETS, _training_preset_command


def test_shared_training_presets_cover_the_five_team_runs() -> None:
    assert set(TRAINING_PRESETS) == {
        "smoke",
        "baseline-no-cost",
        "baseline-cost",
        "main-no-cost",
        "main-cost",
    }


@pytest.mark.parametrize(
    ("preset_name", "model", "epochs", "transaction_cost"),
    [
        ("baseline-no-cost", "conv1d", "30", "0"),
        ("baseline-cost", "conv1d", "30", "0"),
        ("main-no-cost", "resnet_lstm", "30", "0"),
        ("main-cost", "resnet_lstm", "30", "0"),
    ],
)
def test_shared_training_preset_command_uses_expected_core_settings(
    preset_name: str,
    model: str,
    epochs: str,
    transaction_cost: str,
) -> None:
    command = _training_preset_command(preset_name=preset_name, cv_manifest="prepared/cv.json")

    assert f"--model {model}" in command
    assert f"--epochs {epochs}" in command
    assert "--lr 0.001" in command
    assert "--full-eval-every 3" in command
    assert "--bet-sizing fixed" in command
    assert "--fixed-bet-size 1" in command
    assert "--kelly-fraction" not in command
    assert "--portfolio-max-position-fraction 0.02" in command
    assert f"--transaction-cost {transaction_cost}" in command
    assert "--cv-manifest prepared/cv.json" in command
    assert f"--wandb-name {preset_name}" in command


def test_smoke_preset_is_small_and_skips_expensive_return_stats() -> None:
    command = _training_preset_command(preset_name="smoke", exp_dir="prepared/fold00")

    assert "--model conv1d" in command
    assert "--epochs 1" in command
    assert "--full-eval-every 1" in command
    assert "--no-return-stats" in command
    assert "--no-save-best-checkpoint" in command
    assert "--exp-dir prepared/fold00" in command


def test_training_preset_allows_explicit_overrides() -> None:
    command = _training_preset_command(
        preset_name="main-cost",
        extra_args="--seed 7 --wandb-name team-member-run",
    )

    assert "--seed 7" in command
    assert command.endswith("--wandb-name team-member-run")


def test_training_preset_rejects_two_data_scopes() -> None:
    with pytest.raises(ValueError, match="either exp_dir or cv_manifest"):
        _training_preset_command(
            preset_name="smoke",
            exp_dir="prepared/fold00",
            cv_manifest="prepared/cv.json",
        )
