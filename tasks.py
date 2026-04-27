import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "kvant"
PYTHON_VERSION = "3.12"


def _baseline_train_command(
    *,
    epochs: int = 3,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    exp_dir: str | None = None,
    extra_args: str = "",
) -> str:
    """Build the quick baseline training command."""
    cmd = (
        "uv run python -m kvant.ml_framework.scripts.train_experiment "
        "--baseline "
        f"--epochs {int(epochs)} "
        f"--train-batch-size {int(train_batch_size)} "
        f"--eval-batch-size {int(eval_batch_size)}"
    )
    if exp_dir:
        cmd += f' --exp-dir "{exp_dir}"'
    if extra_args.strip():
        cmd += f" {extra_args.strip()}"
    return cmd

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
def baseline(
    ctx: Context,
    epochs: int = 3,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    exp_dir: str = "",
    extra_args: str = "",
) -> None:
    """Run a quick baseline with conv1d and zero transaction cost."""
    cmd = _baseline_train_command(
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        exp_dir=exp_dir or None,
        extra_args=extra_args,
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)

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
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
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
