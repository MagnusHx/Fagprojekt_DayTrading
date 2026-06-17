import os
import shlex
from dataclasses import dataclass

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "kvant"
PYTHON_VERSION = "3.12"


@dataclass(frozen=True)
class TrainingPreset:
    """Shared training configuration used by an invoke task."""

    name: str
    args: tuple[str, ...]


TRAINING_PRESETS = {
    "smoke": TrainingPreset(
        name="smoke",
        args=(
            "--model",
            "conv1d",
            "--epochs",
            "1",
            "--full-eval-every",
            "1",
            "--no-return-stats",
            "--no-save-best-checkpoint",
        ),
    ),
    "baseline-no-cost": TrainingPreset(
        name="baseline-no-cost",
        args=(
            "--model",
            "conv1d",
            "--epochs",
            "30",
            "--lr",
            "0.001",
            "--full-eval-every",
            "3",
            "--bet-sizing",
            "fixed",
            "--fixed-bet-size",
            "1",
            "--portfolio-max-position-fraction",
            "0.02",
            "--transaction-cost",
            "0",
        ),
    ),
    "baseline-cost": TrainingPreset(
        name="baseline-cost",
        args=(
            "--model",
            "conv1d",
            "--epochs",
            "30",
            "--lr",
            "0.001",
            "--full-eval-every",
            "3",
            "--bet-sizing",
            "fixed",
            "--fixed-bet-size",
            "1",
            "--portfolio-max-position-fraction",
            "0.02",
            "--transaction-cost",
            "0",
        ),
    ),
    "main-no-cost": TrainingPreset(
        name="main-no-cost",
        args=(
            "--model",
            "resnet_lstm",
            "--epochs",
            "30",
            "--lr",
            "0.001",
            "--full-eval-every",
            "3",
            "--resnet-channels",
            "64",
            "--resnet-blocks",
            "2",
            "--resnet-kernel-size",
            "5",
            "--lstm-hidden-size",
            "64",
            "--lstm-layers",
            "1",
            "--model-dropout",
            "0.3",
            "--bet-sizing",
            "fixed",
            "--fixed-bet-size",
            "1",
            "--portfolio-max-position-fraction",
            "0.02",
            "--transaction-cost",
            "0",
        ),
    ),
    "main-cost": TrainingPreset(
        name="main-cost",
        args=(
            "--model",
            "resnet_lstm",
            "--epochs",
            "30",
            "--lr",
            "0.001",
            "--full-eval-every",
            "3",
            "--resnet-channels",
            "64",
            "--resnet-blocks",
            "2",
            "--resnet-kernel-size",
            "5",
            "--lstm-hidden-size",
            "64",
            "--lstm-layers",
            "1",
            "--model-dropout",
            "0.3",
            "--bet-sizing",
            "fixed",
            "--fixed-bet-size",
            "1",
            "--portfolio-max-position-fraction",
            "0.02",
            "--transaction-cost",
            "0",
        ),
    ),
}


def _training_preset_command(
    *,
    preset_name: str,
    exp_dir: str | None = None,
    cv_manifest: str | None = None,
    extra_args: str = "",
) -> str:
    """Build a command for one shared training preset."""
    if preset_name not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset {preset_name!r}.")
    if exp_dir and cv_manifest:
        raise ValueError("Pass either exp_dir or cv_manifest, not both.")

    preset = TRAINING_PRESETS[preset_name]
    command = ["uv", "run", "python", "-m", "kvant.ml_framework.scripts.train_experiment", *preset.args]
    if exp_dir:
        command.extend(["--exp-dir", exp_dir])
    if cv_manifest:
        command.extend(["--cv-manifest", cv_manifest])
    command.extend(["--wandb-name", preset.name])
    if extra_args.strip():
        command.extend(shlex.split(extra_args))
    return shlex.join(command)


def _run_training_preset(
    ctx: Context,
    *,
    preset_name: str,
    exp_dir: str,
    cv_manifest: str,
    extra_args: str,
) -> None:
    """Run one shared training preset."""
    command = _training_preset_command(
        preset_name=preset_name,
        exp_dir=exp_dir or None,
        cv_manifest=cv_manifest or None,
        extra_args=extra_args,
    )
    ctx.run(command, echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def smoke(ctx: Context, exp_dir: str = "", cv_manifest: str = "", extra_args: str = "") -> None:
    """Run the shared one-epoch smoke configuration."""
    _run_training_preset(ctx, preset_name="smoke", exp_dir=exp_dir, cv_manifest=cv_manifest, extra_args=extra_args)


@task(name="baseline-no-cost")
def baseline_no_cost(ctx: Context, exp_dir: str = "", cv_manifest: str = "", extra_args: str = "") -> None:
    """Run the shared zero-cost Conv1D baseline."""
    _run_training_preset(
        ctx, preset_name="baseline-no-cost", exp_dir=exp_dir, cv_manifest=cv_manifest, extra_args=extra_args
    )


@task(name="baseline-cost")
def baseline_cost(ctx: Context, exp_dir: str = "", cv_manifest: str = "", extra_args: str = "") -> None:
    """Run the legacy cost-named Conv1D baseline under the zero-cost protocol."""
    _run_training_preset(
        ctx, preset_name="baseline-cost", exp_dir=exp_dir, cv_manifest=cv_manifest, extra_args=extra_args
    )


@task(name="main-no-cost")
def main_no_cost(ctx: Context, exp_dir: str = "", cv_manifest: str = "", extra_args: str = "") -> None:
    """Run the shared zero-cost ResNet-LSTM candidate."""
    _run_training_preset(
        ctx, preset_name="main-no-cost", exp_dir=exp_dir, cv_manifest=cv_manifest, extra_args=extra_args
    )


@task(name="main-cost")
def main_cost(ctx: Context, exp_dir: str = "", cv_manifest: str = "", extra_args: str = "") -> None:
    """Run the legacy cost-named ResNet-LSTM candidate under the zero-cost protocol."""
    _run_training_preset(ctx, preset_name="main-cost", exp_dir=exp_dir, cv_manifest=cv_manifest, extra_args=extra_args)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
