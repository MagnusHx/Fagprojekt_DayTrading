from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from kvant.ml_prepare_data import prepared_data_root
from kvant.ml_framework.wandb_defaults import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


CUSUM_THRESHOLDS = (0.01, 0.02, 0.03)
BARRIER_HEIGHTS = (0.01, 0.02, 0.04, 0.06)
BARRIER_WIDTHS = (240,)
DECISION_THRESHOLDS = (0.45, 0.50, 0.55, 0.60)

PROMISING_TEMPLATE = Path("reports/promising_grid_configs.json")


@dataclass(frozen=True)
class GridConfig:
    cusum_h: float
    barrier_height: float
    barrier_width: int

    @property
    def barrier_height_pct(self) -> float:
        """Return the barrier height in the percentage unit expected by preparation CLI."""
        return float(self.barrier_height) * 100.0

    @property
    def label(self) -> str:
        """Return the prepared experiment label produced by prepare_experiment.py."""
        return f"sb_L_12_w{int(self.barrier_width)}_h{self.barrier_height_pct:g}_" f"fixedCUSUM{float(self.cusum_h):g}"

    @property
    def manifest_path(self) -> Path:
        """Return the expected CV manifest path for this prepared experiment."""
        return prepared_data_root / f"{self.label}_cv_manifest.json"


@dataclass(frozen=True)
class GridRun:
    config: GridConfig
    model: Literal["conv1d", "resnet_lstm"]
    meta_threshold: float | None = None
    no_meta: bool = True
    primary_confidence_threshold: float = 0.0

    @property
    def run_name(self) -> str:
        """Return a compact W&B run name for this grid item."""
        pieces = [
            "E2-grid",
            self.model,
            f"w{int(self.config.barrier_width)}",
            f"tb{_fmt_percent_token(self.config.barrier_height)}",
            f"cusum{_fmt_percent_token(self.config.cusum_h)}",
        ]
        if self.meta_threshold is not None:
            pieces.append(f"mt{_fmt_token(self.meta_threshold)}")
        elif self.no_meta:
            pieces.append("nometa")
        if float(self.primary_confidence_threshold) > 0.0:
            pieces.append(f"ct{_fmt_token(self.primary_confidence_threshold)}")
        return "-".join(pieces)

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for this run."""
        return Path("artifacts") / self.run_name

    @property
    def results_path(self) -> Path:
        """Return the stable fold-results CSV path for this run."""
        return Path("results") / "grid_search" / f"{self.run_name}.csv"


def _fmt_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _fmt_percent_token(value: float) -> str:
    return f"{float(value) * 100:g}"


def iter_grid_configs() -> Iterable[GridConfig]:
    """Yield the full CUSUM/barrier calibration grid."""
    for cusum_h in CUSUM_THRESHOLDS:
        for barrier_height in BARRIER_HEIGHTS:
            for barrier_width in BARRIER_WIDTHS:
                yield GridConfig(
                    cusum_h=float(cusum_h),
                    barrier_height=float(barrier_height),
                    barrier_width=int(barrier_width),
                )


def prepare_command(config: GridConfig) -> list[str]:
    """Build the command that prepares one CUSUM/barrier configuration."""
    return [
        "uv",
        "run",
        "python",
        "-m",
        "kvant.ml_prepare_data.prepare_experiment",
        "--sampler",
        "fixed_cusum",
        "--cusum-h",
        f"{config.cusum_h:g}",
        "--barrier-width",
        str(int(config.barrier_width)),
        "--barrier-height-pct",
        f"{config.barrier_height_pct:g}",
        "--cv-manifest",
        str(config.manifest_path),
    ]


def train_command(
    run: GridRun,
    *,
    epochs: int,
    transaction_cost: float,
    wandb_project: str,
    wandb_entity: str,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """Build the training command for one manifest/model/threshold combination."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "kvant.ml_framework.scripts.train_experiment",
        "--cv-manifest",
        str(run.config.manifest_path),
        "--model",
        run.model,
        "--epochs",
        str(int(epochs)),
        "--transaction-cost",
        f"{float(transaction_cost):g}",
        "--wandb-project",
        wandb_project,
        "--wandb-entity",
        wandb_entity,
        "--wandb-name",
        run.run_name,
        "--checkpoint-out-dir",
        str(run.checkpoint_dir),
        "--results-out",
        str(run.results_path),
    ]
    if run.meta_threshold is not None:
        cmd.extend(["--meta-accept-threshold", f"{float(run.meta_threshold):g}"])
    elif run.no_meta:
        cmd.append("--no-meta")
        cmd.extend(["--fixed-bet-size", "1"])
    if float(run.primary_confidence_threshold) > 0.0:
        cmd.extend(["--primary-confidence-threshold", f"{float(run.primary_confidence_threshold):g}"])
    cmd.extend(extra_args)
    return cmd


def load_promising_configs(path: Path) -> list[GridConfig]:
    """Load ResNet-LSTM follow-up configs selected from Conv1D grid results."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("configs", [])
    configs: list[GridConfig] = []
    for row in payload:
        configs.append(
            GridConfig(
                cusum_h=float(row["cusum_h"]),
                barrier_height=float(row["barrier_height"]),
                barrier_width=int(row["barrier_width"]),
            )
        )
    return configs


def write_promising_template(path: Path) -> None:
    """Write a small template for manually selected ResNet-LSTM follow-up configs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Fill this after the Conv1D grid. Keep only configurations worth testing with ResNet-LSTM.",
        "configs": [
            {"cusum_h": 0.01, "barrier_height": 0.02, "barrier_width": 240},
            {"cusum_h": 0.02, "barrier_height": 0.02, "barrier_width": 240},
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _selected(items: list[list[str]], *, start_index: int, max_runs: int | None) -> list[list[str]]:
    start = max(int(start_index), 0)
    selected = items[start:]
    if max_runs is not None:
        selected = selected[: max(int(max_runs), 0)]
    return selected


def _run_or_print(commands: list[list[str]], *, execute: bool, plan_out: Path | None) -> None:
    if plan_out is not None:
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        plan_out.write_text(json.dumps([{"command": command} for command in commands], indent=2))
        print(f"Wrote command plan to {plan_out}")

    for command in commands:
        print(" ".join(command))
        if execute:
            subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the experiment-grid runner."""
    parser = argparse.ArgumentParser(description="Prepare and run the CUSUM/barrier calibration grid.")
    parser.add_argument(
        "mode",
        choices=(
            "plan",
            "prepare",
            "train-conv1d",
            "train-resnet",
            "train-confidence",
            "train-meta",
            "write-promising-template",
        ),
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually run commands. Default only prints/writes them."
    )
    parser.add_argument("--plan-out", type=Path, default=Path("artifacts/run_debug/experiment_grid_plan.json"))
    parser.add_argument("--start-index", type=int, default=0, help="Skip commands before this zero-based index.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit how many commands are printed or executed.")
    parser.add_argument("--conv1d-epochs", type=int, default=20)
    parser.add_argument("--resnet-epochs", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--promising-configs", type=Path, default=PROMISING_TEMPLATE)
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Include preparation commands even when the expected CV manifest already exists.",
    )
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="Additional training argument token. Repeat for argument names and values.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested grid stage."""
    args = parse_args()
    configs = list(iter_grid_configs())
    extra_train_args = tuple(str(arg) for arg in args.extra_train_arg)

    if args.mode == "write-promising-template":
        write_promising_template(args.promising_configs)
        print(f"Wrote template to {args.promising_configs}")
        return

    if args.mode in {"plan", "prepare"}:
        prepare_commands = [
            prepare_command(config) for config in configs if args.force_prepare or not config.manifest_path.exists()
        ]
        if args.mode == "prepare":
            commands = prepare_commands
        else:
            train_commands = [
                train_command(
                    GridRun(config=config, model="conv1d", no_meta=True),
                    epochs=args.conv1d_epochs,
                    transaction_cost=args.transaction_cost,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    extra_args=extra_train_args,
                )
                for config in configs
            ]
            commands = prepare_commands + train_commands
    elif args.mode == "train-conv1d":
        commands = [
            train_command(
                GridRun(config=config, model="conv1d", no_meta=True),
                epochs=args.conv1d_epochs,
                transaction_cost=args.transaction_cost,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                extra_args=extra_train_args,
            )
            for config in configs
        ]
    elif args.mode == "train-resnet":
        configs = load_promising_configs(args.promising_configs)
        commands = [
            train_command(
                GridRun(config=config, model="resnet_lstm", no_meta=True),
                epochs=args.resnet_epochs,
                transaction_cost=args.transaction_cost,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                extra_args=extra_train_args,
            )
            for config in configs
        ]
    elif args.mode == "train-confidence":
        configs = load_promising_configs(args.promising_configs)
        commands = [
            train_command(
                GridRun(
                    config=config,
                    model="conv1d",
                    no_meta=True,
                    primary_confidence_threshold=threshold,
                ),
                epochs=args.conv1d_epochs,
                transaction_cost=args.transaction_cost,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                extra_args=extra_train_args,
            )
            for config in configs
            for threshold in DECISION_THRESHOLDS
        ]
    else:
        configs = load_promising_configs(args.promising_configs)
        commands = [
            train_command(
                GridRun(config=config, model="conv1d", no_meta=False, meta_threshold=threshold),
                epochs=args.conv1d_epochs,
                transaction_cost=args.transaction_cost,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                extra_args=extra_train_args,
            )
            for config in configs
            for threshold in DECISION_THRESHOLDS
        ]

    selected = _selected(commands, start_index=args.start_index, max_runs=args.max_runs)
    metadata = {
        "mode": args.mode,
        "execute": bool(args.execute),
        "start_index": int(args.start_index),
        "max_runs": args.max_runs,
        "total_commands_before_selection": len(commands),
        "selected_commands": len(selected),
        "grid": {
            "cusum_thresholds": list(CUSUM_THRESHOLDS),
            "barrier_heights": list(BARRIER_HEIGHTS),
            "barrier_widths": list(BARRIER_WIDTHS),
            "decision_thresholds": list(DECISION_THRESHOLDS),
        },
        "commands": [{"command": command} for command in selected],
    }
    if args.plan_out is not None:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(metadata, indent=2))
        print(f"Wrote command plan to {args.plan_out}")

    _run_or_print(selected, execute=bool(args.execute), plan_out=None)


if __name__ == "__main__":
    main()
