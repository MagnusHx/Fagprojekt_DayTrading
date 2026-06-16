from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kvant.labels import SIDE_LABEL_DOWN, SIDE_LABEL_UP
from kvant.ml_framework.run_validation import (
    git_commit_or_none,
    validate_cv_manifest,
    validate_prepared_experiment,
)
from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.models import create_model
from kvant.ml_framework.train import (
    Trainer,
    TrainConfig,
    ExperimentEvaluator,
    EvalConfig,
    normalize_meta_features,
)
from kvant.ml_framework.train.utils import class_weights_from_dataset
from kvant.ml_framework.logging import WandbLogger

import os
from dotenv import load_dotenv

load_dotenv()
default_project = os.environ.get("WANDB_PROJECT", "Kvant")
entity = os.environ.get("WANDB_ENTITY", "s245509-danmarks-tekniske-universitet-dtu")


RESULT_METRICS = {
    "test_accuracy": "test/classification/accuracy",
    "test_f1_macro": "test/classification/f1_macro",
    "test_meta_f1": "test/meta/f1",
    "test_take_rate": "test/meta/take_rate",
    "test_trade_signal_rate": "test/decision/trade_signal_rate",
    "test_directional_acted_accuracy": "test/decision/directional_acted_accuracy",
    "test_portfolio_total_return_pct": "test/portfolio/total_return_pct",
    "test_portfolio_sharpe_ratio_annualized": "test/portfolio/sharpe_ratio_annualized",
    "test_portfolio_max_drawdown_pct": "test/portfolio/max_drawdown_pct",
    "test_portfolio_annualized_return_pct": "test/portfolio/annualized_return_pct",
    "test_portfolio_average_trade_return_pct": "test/portfolio/average_trade_return_pct",
    "test_portfolio_n_executed_trades": "test/portfolio/n_executed_trades",
    "test_paper_net_return_total_pct": "test/paper/executed_trade_net_return_total_pct",
    "test_paper_sharpe_ratio_annualized": "test/paper/sharpe_ratio_annualized",
}


def _apply_baseline_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the baseline preset."""
    if not getattr(args, "baseline", False):
        return args

    args.model = "conv1d"
    args.transaction_cost = 0.0
    return args


def _parse_meta_features(values: list[str] | None) -> tuple[str, ...]:
    return normalize_meta_features(values)


def _make_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the configured learning-rate scheduler."""
    scheduler_name = getattr(args, "lr_scheduler", "cosine")
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )
    raise ValueError(f"Unknown learning-rate scheduler {scheduler_name!r}.")


def _droptexit_sibling(path: Path | None) -> Path | None:
    if path is None:
        return None
    name = path.name
    if "_droptexit" not in name:
        return None
    return path.with_name(name.replace("_droptexit", "", 1))


def _config_declares_event_outcome(cfg: dict) -> bool:
    semantics = (cfg.get("label_semantics") or {}).get("labels") or {}
    try:
        label_ids = tuple(sorted(int(k) for k in semantics.keys()))
    except Exception:
        return False
    pipeline_stage = str(cfg.get("pipeline_stage", "event_outcome"))
    return pipeline_stage == "event_outcome" and label_ids == (0, 1, 2)


def _is_event_outcome_exp_dir(exp_dir: Path | None) -> bool:
    if exp_dir is None:
        return False
    cfg_path = Path(exp_dir) / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return False
    return _config_declares_event_outcome(cfg)


def _resolved_manifest_path(manifest_path: Path | None) -> Path | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    if manifest_path.suffix != ".txt":
        return manifest_path
    try:
        resolved = Path(manifest_path.read_text().strip())
    except Exception:
        return None
    return resolved if resolved.exists() else None


def _manifest_declares_event_outcome(manifest_path: Path | None) -> bool:
    resolved = _resolved_manifest_path(manifest_path)
    if resolved is None:
        return False
    try:
        payload = json.loads(resolved.read_text())
    except Exception:
        return False
    folds = payload.get("folds", [])
    if not folds:
        return False
    exp_dir = Path(folds[0].get("exp_dir", ""))
    return _is_event_outcome_exp_dir(exp_dir)


def _compatible_default_exp_dir(candidate: Path | None) -> Path | None:
    if _is_event_outcome_exp_dir(candidate):
        return candidate
    sibling = _droptexit_sibling(candidate)
    if _is_event_outcome_exp_dir(sibling):
        return sibling
    return candidate


def _compatible_default_manifest(candidate: Path | None) -> Path | None:
    resolved = _resolved_manifest_path(candidate)
    if _manifest_declares_event_outcome(resolved):
        return resolved
    sibling = _droptexit_sibling(resolved)
    if _manifest_declares_event_outcome(sibling):
        return sibling
    return resolved


def _cli_flag_present(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _should_run_cv(args: argparse.Namespace, argv: list[str] | None = None) -> bool:
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit_exp_dir = _cli_flag_present(argv, "--exp-dir")
    explicit_cv_manifest = _cli_flag_present(argv, "--cv-manifest")
    if explicit_exp_dir and not explicit_cv_manifest:
        return False
    return bool(args.cv_manifest is not None and args.cv_manifest.exists())


def _slugify_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = re.sub(r"-{2,}", "-", token).strip("-._")
    return token or "run"


def _fmt_compact_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def _scope_token(path: Path | str | None) -> str:
    if path is None:
        return "default"
    resolved = Path(path)
    name = resolved.name
    if name.endswith((".json", ".txt")):
        name = resolved.stem
    return _slugify_token(name)


def _auto_wandb_name(args: argparse.Namespace, *, run_cv: bool) -> str:
    if getattr(args, "wandb_name", None):
        return str(args.wandb_name)

    scope_path = args.cv_manifest if run_cv else args.exp_dir
    scope_name = _scope_token(scope_path)
    parts = [
        "baseline" if getattr(args, "baseline", False) else args.model,
        scope_name,
        f"ep{int(args.epochs)}",
        f"tc{_fmt_compact_float(float(args.transaction_cost))}",
    ]
    if not getattr(args, "baseline", False):
        parts.insert(1, f"meta{_slugify_token(args.pipeline_stage)}")
    return "-".join(parts)


def parse_args() -> argparse.Namespace:
    from kvant.ml_prepare_data import prepared_data_root

    exp_ptr = prepared_data_root / "last_experiment.txt"
    default_exp_dir = None
    if exp_ptr.exists():
        exp_id = exp_ptr.read_text().strip()
        if exp_id:
            default_exp_dir = _compatible_default_exp_dir(prepared_data_root / exp_id)
    default_cv_manifest = None
    cv_ptr = prepared_data_root / "last_experiment_cv_manifest.txt"
    if cv_ptr.exists():
        default_cv_manifest = _compatible_default_manifest(cv_ptr)

    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=False, default=default_exp_dir)
    p.add_argument("--cv-manifest", type=Path, required=False, default=default_cv_manifest)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", choices=("cosine", "none"), default="cosine")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Preset a baseline launch with model=conv1d and transaction_cost=0.0.",
    )
    p.add_argument("--model", type=str, choices=("conv1d", "resnet_lstm"), default="conv1d")
    p.add_argument("--model-dropout", type=float, default=0.3)
    p.add_argument("--resnet-channels", type=int, default=64)
    p.add_argument("--resnet-blocks", type=int, default=2)
    p.add_argument("--resnet-kernel-size", type=int, default=5)
    p.add_argument("--lstm-hidden-size", type=int, default=64)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--wandb-project", type=str, default=default_project)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-api-timeout", type=int, default=29)
    p.add_argument(
        "--full-eval-every",
        type=int,
        default=3,
        help="Run expensive full evaluation every N epochs; epoch 1 and the final epoch are always evaluated.",
    )
    p.add_argument("--no-return-stats", action="store_true")
    p.add_argument("--pipeline-stage", type=str, choices=("primary_side",), default="primary_side")
    p.add_argument("--initial-portfolio", type=float, default=1.0)
    p.add_argument("--portfolio-initial-cash", type=float, default=10_000.0)
    p.add_argument("--portfolio-max-position-fraction", type=float, default=0.02)
    p.add_argument("--portfolio-max-total-exposure", type=float, default=1.0)
    p.add_argument("--portfolio-max-positions", type=int, default=10)
    p.add_argument("--transaction-cost", type=float, default=0.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--kelly-payoff-ratio", type=float, default=1.0)
    p.add_argument("--risk-free-rate", type=float, default=0.0314)
    p.add_argument("--days-per-year", type=float, default=365.0)
    p.add_argument(
        "--no-meta",
        action="store_true",
        help="Disable meta-label decision layer; act on every primary signal with fixed bet size (E5 nometa arm).",
    )
    p.add_argument(
        "--meta-accept-threshold",
        type=float,
        default=0.5,
        help="Minimum meta-label probability required before taking the primary side prediction.",
    )
    p.add_argument(
        "--meta-features",
        action="append",
        default=None,
        help="Repeatable or comma-separated meta-label feature tokens. "
        "Supported: proba, logits, embedding, prediction_margin, prediction_entropy, "
        "time_since_last_event, prepared_last:<feature_name>, ticker_rolling_win_rate_<N>, "
        "ticker_directional_win_rate_<N>, ticker_recent_net_return_<N>. "
        "Prepared aliases include prepared_last:volatility and prepared_last:recent_return.",
    )
    p.add_argument("--topk-ticker-plots", type=int, default=50)
    p.add_argument(
        "--wandb-optional-media",
        action="store_true",
        help="Enable heavier optional W&B charts and media artifacts.",
    )
    p.add_argument(
        "--print-dataset-summary",
        action="store_true",
        help="Print split summaries to stdout during training startup.",
    )
    p.add_argument("--checkpoint-out-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument(
        "--results-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-fold CV results. Defaults to results/<wandb-name>.csv for CV runs.",
    )
    p.add_argument(
        "--no-save-best-checkpoint",
        action="store_true",
        help="Disable writing a local best-checkpoint bundle for offline metric reconciliation.",
    )
    args = p.parse_args()
    args = _apply_baseline_preset(args)
    if args.exp_dir is None and args.cv_manifest is None:
        raise SystemExit(
            "No prepared experiment found. Run `uv run python -m kvant.ml_prepare_data.prepare_experiment` "
            "or pass --exp-dir/--cv-manifest explicitly."
        )
    args.wandb_name = _auto_wandb_name(args, run_cv=_should_run_cv(args))
    # Handle --no-meta: disable all meta features
    if getattr(args, "no_meta", False):
        args.meta_features = ()
        args.meta_accept_threshold = 0.0
    else:
        args.meta_features = _parse_meta_features(args.meta_features)
    if not (0.0 <= args.meta_accept_threshold <= 1.0):
        raise SystemExit("--meta-accept-threshold must be between 0 and 1.")
    if args.kelly_fraction < 0.0:
        raise SystemExit("--kelly-fraction must be non-negative.")
    if args.kelly_payoff_ratio <= 0.0:
        raise SystemExit("--kelly-payoff-ratio must be positive.")
    if args.portfolio_initial_cash <= 0.0:
        raise SystemExit("--portfolio-initial-cash must be positive.")
    if not (0.0 < args.portfolio_max_position_fraction <= 1.0):
        raise SystemExit("--portfolio-max-position-fraction must be in (0, 1].")
    if args.portfolio_max_total_exposure <= 0.0:
        raise SystemExit("--portfolio-max-total-exposure must be positive.")
    if args.portfolio_max_positions <= 0:
        raise SystemExit("--portfolio-max-positions must be positive.")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("epochs and batch sizes must be positive.")
    if args.full_eval_every <= 0:
        raise SystemExit("--full-eval-every must be positive.")
    if args.min_lr < 0.0:
        raise SystemExit("--min-lr must be non-negative.")
    if args.min_lr > args.lr:
        raise SystemExit("--min-lr must be less than or equal to --lr.")
    return args


def _model_kwargs(args: argparse.Namespace) -> dict:
    return {
        "conv_channels": args.resnet_channels,
        "num_blocks": args.resnet_blocks,
        "kernel_size": args.resnet_kernel_size,
        "lstm_hidden_size": args.lstm_hidden_size,
        "lstm_layers": args.lstm_layers,
        "dropout": args.model_dropout,
    }


def _device_status_message(device: torch.device) -> str:
    """Build a human-readable runtime message for the selected torch device."""
    if device.type != "cuda":
        return "Using device: cpu (CUDA not available)"

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    return f"Using device: cuda:{device_index} ({device_name})"


def _save_best_checkpoint_bundle(
    *,
    args: argparse.Namespace,
    exp_dir: Path,
    fold_tag: str | None,
    best_state: dict | None,
    best_metric: float,
    exp: PreparedExperiment,
    labeler_cfg: dict,
) -> Path | None:
    if args.no_save_best_checkpoint or best_state is None:
        return None

    out_dir = Path(args.checkpoint_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.wandb_name or "stocks-run"
    if fold_tag:
        stem = f"{stem}-{fold_tag}"
    bundle_path = out_dir / f"{stem}-best.ckpt.pt"
    bundle = {
        "exp_dir": str(exp_dir),
        "fold_tag": fold_tag,
        "best_metric": float(best_metric),
        "checkpoint_metric": "val/meta/f1",
        "run_stage": str(args.pipeline_stage),
        "event_label_semantics": exp.label_semantics,
        "event_label_ids": list(exp.label_ids),
        "primary_label_ids": [0, 1],
        "model_name": args.model,
        "model_kwargs": _model_kwargs(args),
        "model_state": best_state,
        "run_metadata": {
            "git_commit": git_commit_or_none(Path(__file__).resolve().parents[4]),
            "seed": int(getattr(args, "seed", 0)),
        },
        "eval_config": {
            "compute_per_ticker_accuracy": True,
            "compute_profit_stats": not args.no_return_stats,
            "compute_paper_trading_metrics": not args.no_return_stats,
            "meta_model": "logreg",
            "meta_features": list(args.meta_features),
            "meta_random_state": int(args.seed),
            "meta_accept_threshold": float(args.meta_accept_threshold),
            "initial_portfolio": args.initial_portfolio,
            "portfolio_initial_cash": float(args.portfolio_initial_cash),
            "portfolio_max_position_fraction": float(args.portfolio_max_position_fraction),
            "portfolio_max_total_exposure": float(args.portfolio_max_total_exposure),
            "portfolio_max_positions": int(args.portfolio_max_positions),
            "transaction_cost": args.transaction_cost,
            "kelly_fraction": float(args.kelly_fraction),
            "kelly_payoff_ratio": float(args.kelly_payoff_ratio),
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "backtest_width_minutes": int(labeler_cfg.get("width_minutes", 0)),
            "backtest_barrier_height": float(labeler_cfg.get("height", 0.0)),
        },
    }
    torch.save(bundle, bundle_path)
    print(f"Saved best-checkpoint bundle to {bundle_path}")
    return bundle_path


def _make_logger(
    args: argparse.Namespace,
    *,
    exp_dir: Path,
    fold_tag: str | None = None,
) -> WandbLogger:
    return WandbLogger(
        project=args.wandb_project,
        entity=entity,
        name=(args.wandb_name or "stocks-run")
        if fold_tag is None
        else f"{(args.wandb_name or 'stocks-run')}-{fold_tag}",
        api_timeout=args.wandb_api_timeout,
        config={
            "exp_dir": str(exp_dir),
            "fold_tag": fold_tag,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "model": args.model,
            "model_dropout": args.model_dropout,
            "resnet_channels": args.resnet_channels,
            "resnet_blocks": args.resnet_blocks,
            "resnet_kernel_size": args.resnet_kernel_size,
            "lstm_hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "L": None,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "full_eval_every": getattr(args, "full_eval_every", 3),
            "wandb_metric_mode": "compact",
            "class_weights": None,
            "pipeline_stage": args.pipeline_stage,
            "meta_model": "logreg",
            "meta_features": list(args.meta_features),
            "meta_accept_threshold": float(args.meta_accept_threshold),
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "kelly_fraction": float(args.kelly_fraction),
            "kelly_payoff_ratio": float(args.kelly_payoff_ratio),
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "backtest_width_minutes": None,
            "backtest_barrier_height": None,
            "checkpoint_metric": "val/meta/f1",
            "metric_pipeline": "side_model -> meta_label -> trade_decision -> execution -> economics",
        },
        enable_optional_media=args.wandb_optional_media,
        per_ticker_chart_limit=args.topk_ticker_plots,
        compact_metrics=True,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _result_stem(args: argparse.Namespace, manifest_path: Path | None = None) -> str:
    raw = str(getattr(args, "wandb_name", None) or "")
    if not raw and manifest_path is not None:
        raw = Path(manifest_path).stem
    if not raw:
        raw = "training_results"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return stem or "training_results"


def _default_results_path(args: argparse.Namespace, manifest_path: Path | None = None) -> Path:
    return Path("results") / f"{_result_stem(args, manifest_path)}.csv"


def _manifest_results_path(manifest_path: Path) -> Path:
    stem = Path(manifest_path).stem
    for suffix in ("_cv_manifest", "-cv-manifest", "_manifest", "-manifest"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or Path(manifest_path).stem
    return Path("results") / f"{stem}.csv"


def _fold_result_row(fold_idx: int, metrics: dict) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {"fold": int(fold_idx)}
    for out_name, metric_name in RESULT_METRICS.items():
        value = metrics.get(metric_name)
        row[out_name] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return row


def _write_cv_results_csv(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fold", *RESULT_METRICS.keys()]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CV fold results to {path}", flush=True)


def _write_default_cv_result_csvs(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    rows: list[dict[str, float | int | None]],
) -> None:
    primary_path = Path(args.results_out) if args.results_out is not None else _default_results_path(args, manifest_path)
    _write_cv_results_csv(primary_path, rows)
    if args.results_out is not None:
        return

    manifest_path_out = _manifest_results_path(manifest_path)
    if manifest_path_out != primary_path:
        _write_cv_results_csv(manifest_path_out, rows)


def _runtime_metadata(args: argparse.Namespace, *, exp_dir: Path, fold_tag: str | None, preflight: dict) -> dict:
    label_regime = preflight.get("label_regime")
    return {
        "git_commit": git_commit_or_none(Path(__file__).resolve().parents[4]),
        "exp_dir": str(exp_dir),
        "fold_tag": fold_tag,
        "label_regime": label_regime,
        "pipeline_stage": str(args.pipeline_stage),
        "model_name": args.model,
        "model_kwargs": _model_kwargs(args),
        "seed_python": int(args.seed),
        "seed_numpy": int(args.seed),
        "seed_torch": int(args.seed),
        "epochs": int(args.epochs),
        "meta_model": "logreg",
        "meta_features": list(args.meta_features),
        "meta_accept_threshold": float(args.meta_accept_threshold),
        "transaction_cost": float(args.transaction_cost),
        "kelly_fraction": float(args.kelly_fraction),
        "kelly_payoff_ratio": float(args.kelly_payoff_ratio),
        "require_market_data": bool(not args.no_return_stats),
        "preflight": preflight,
    }


def _validate_primary_side_train_labels(exp: PreparedExperiment) -> None:
    side_labels = exp.store.side_labels_for_index(exp.index_train)
    side_labels = side_labels[side_labels >= 0]
    counts = {
        SIDE_LABEL_DOWN: int(np.sum(side_labels == SIDE_LABEL_DOWN)),
        SIDE_LABEL_UP: int(np.sum(side_labels == SIDE_LABEL_UP)),
    }
    if counts[SIDE_LABEL_DOWN] == 0 or counts[SIDE_LABEL_UP] == 0:
        raise RuntimeError(
            "Primary-side training requires both down and up labels after mapping event labels to side labels. "
            f"Got counts down={counts[SIDE_LABEL_DOWN]}, up={counts[SIDE_LABEL_UP]}. "
            "If this was prepared with --labeler next_bar before the label fix, regenerate the prepared experiment."
        )


def run_single_fold(
    args: argparse.Namespace,
    exp_dir: Path,
    fold_tag: str | None = None,
    logger: WandbLogger | None = None,
) -> dict:
    preflight = validate_prepared_experiment(exp_dir, require_market_data=not args.no_return_stats).to_jsonable()
    runtime_dir = Path("artifacts/run_debug") / Path(exp_dir).name
    _write_json_artifact(runtime_dir / "preflight.json", preflight)

    exp = PreparedExperiment(exp_dir)
    if exp.store.pipeline_stage != "event_outcome" or exp.n_classes != 3:
        raise RuntimeError(
            "The Lopez de Prado pipeline expects event-outcome prepared artifacts with raw triple-barrier labels. "
            "Regenerate the prepared experiment before training."
        )

    dl_train, dl_val, dl_test = exp.get_primary_side_loaders(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Train eval loader (no shuffle) so "train" metrics are stable and comparable
    dl_train_eval = DataLoader(
        dl_train.dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Optional local sanity check
    ds_train, ds_val, ds_test = exp.get_primary_side_datasets()
    _validate_primary_side_train_labels(exp)

    if args.print_dataset_summary:
        for ds, split_name in [(ds_train, "train"), (ds_val, "val"), (ds_test, "test")]:
            print(f"Dataset {split_name}")
            ds.summary(display=True)
            print("-" * 10, "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(_device_status_message(device), flush=True)
    model = create_model(
        model_name=args.model,
        n_features=exp.store.n_features,
        n_classes=2,
        **_model_kwargs(args),
    ).to(device)
    labeler_cfg = exp.cfg.get("labeler", {})

    w = class_weights_from_dataset(ds_train, n_classes=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device), ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _make_lr_scheduler(args, optimizer)

    owns_logger = logger is None
    logger = logger or _make_logger(args, exp_dir=exp_dir, fold_tag=fold_tag)
    logger.log_config(
        {
            "exp_dir": str(exp_dir),
            "fold_tag": fold_tag,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "model": args.model,
            "model_dropout": args.model_dropout,
            "resnet_channels": args.resnet_channels,
            "resnet_blocks": args.resnet_blocks,
            "resnet_kernel_size": args.resnet_kernel_size,
            "lstm_hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "L": exp.L,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "full_eval_every": getattr(args, "full_eval_every", 3),
            "wandb_metric_mode": "compact",
            "class_weights": w.tolist(),
            "n_classes": 2,
            "event_label_semantics": exp.label_semantics,
            "pipeline_stage": args.pipeline_stage,
            "meta_model": "logreg",
            "meta_features": list(args.meta_features),
            "meta_accept_threshold": float(args.meta_accept_threshold),
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "kelly_fraction": float(args.kelly_fraction),
            "kelly_payoff_ratio": float(args.kelly_payoff_ratio),
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "backtest_width_minutes": int(labeler_cfg.get("width_minutes", 0)),
            "backtest_barrier_height": float(labeler_cfg.get("height", 0.0)),
            "runtime_metadata": _runtime_metadata(args, exp_dir=exp_dir, fold_tag=fold_tag, preflight=preflight),
        }
    )

    try:
        logger.setup(
            exp=exp,
            loaders={"train": dl_train_eval, "val": dl_val, "test": dl_test},
        )

        evaluator = ExperimentEvaluator(
            store=exp.store,
            device=device,
            logger=logger,
            cfg=EvalConfig(
                compute_per_ticker_accuracy=True,
                compute_profit_stats=not args.no_return_stats,
                compute_paper_trading_metrics=not args.no_return_stats,
                compute_portfolio_metrics=not args.no_return_stats,
                meta_model="logreg",
                meta_features=args.meta_features,
                meta_random_state=int(args.seed),
                meta_accept_threshold=float(args.meta_accept_threshold),
                initial_portfolio=args.initial_portfolio,
                portfolio_initial_cash=float(args.portfolio_initial_cash),
                portfolio_max_position_fraction=float(args.portfolio_max_position_fraction),
                portfolio_max_total_exposure=float(args.portfolio_max_total_exposure),
                portfolio_max_positions=int(args.portfolio_max_positions),
                transaction_cost=args.transaction_cost,
                kelly_fraction=float(args.kelly_fraction),
                kelly_payoff_ratio=float(args.kelly_payoff_ratio),
                risk_free_rate=args.risk_free_rate,
                days_per_year=args.days_per_year,
                backtest_width_minutes=int(labeler_cfg.get("width_minutes", 0)),
                backtest_barrier_height=float(labeler_cfg.get("height", 0.0)),
                labels=(0, 1),
            ),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            evaluator=evaluator,
            logger=logger,
            scheduler=scheduler,
        )

        cfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            checkpoint_metric="val/meta/f1",
            full_eval_every=getattr(args, "full_eval_every", 3),
        )

        out = trainer.fit(
            train_loader=dl_train,
            train_eval_loader=dl_train_eval,
            val_loader=dl_val,
            test_loader=dl_test,
            cfg=cfg,
        )

        if out["best_state"] is not None:
            model.load_state_dict(out["best_state"])
        _save_best_checkpoint_bundle(
            args=args,
            exp_dir=exp_dir,
            fold_tag=fold_tag,
            best_state=out["best_state"],
            best_metric=float(out["best_metric"]),
            exp=exp,
            labeler_cfg=labeler_cfg,
        )

        best_metrics = evaluator.evaluate_all(
            model,
            {"train": dl_train_eval, "val": dl_val, "test": dl_test},
            step=args.epochs + 1,
            metric_splits=("val", "test"),
            detailed=True,
        )
        logger.child(namespace="best").log(best_metrics, step=args.epochs + 1)
        fold_idx = int(str(fold_tag).removeprefix("fold")) if fold_tag is not None else 0
        return {
            "best_metric": float(out["best_metric"]),
            "result_row": _fold_result_row(fold_idx, best_metrics),
        }
    finally:
        if owns_logger:
            logger.stop()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[4]
    require_market_data = not args.no_return_stats

    if _should_run_cv(args):
        manifest_diagnostics = validate_cv_manifest(args.cv_manifest, require_market_data=require_market_data)
        _write_json_artifact(project_root / "artifacts/run_debug/cv_manifest_preflight.json", manifest_diagnostics)

        resolved_manifest_path = Path(manifest_diagnostics["manifest_path"])
        payload = json.loads(resolved_manifest_path.read_text())
        folds = payload.get("folds", [])
        base_seed = int(args.seed)
        root_logger = _make_logger(args, exp_dir=Path(payload.get("exp_dir", folds[0]["exp_dir"])), fold_tag="cv")
        root_logger.log_config(
            {
                "cv_manifest": str(resolved_manifest_path),
                "cv_folds": len(folds),
                "git_commit": git_commit_or_none(project_root),
                "seed_base": base_seed,
            }
        )

        try:
            bests = []
            result_rows = []
            steps_per_fold = args.epochs + 2
            for i, fold in enumerate(folds):
                fold_idx = int(fold["fold_idx"])
                exp_dir = Path(fold["exp_dir"])
                fold_tag = f"fold{fold_idx:02d}"
                fold_seed = int(base_seed + fold_idx)
                _seed_everything(fold_seed)
                print(f"\n=== Training {fold_tag} on {exp_dir} (seed={fold_seed}) ===", flush=True)
                fold_logger = root_logger.child(namespace=fold_tag, step_offset=i * steps_per_fold)
                fold_logger.log_config({"fold_seed": fold_seed})
                fold_args = argparse.Namespace(**vars(args))
                fold_args.seed = fold_seed
                fold_result = run_single_fold(fold_args, exp_dir=exp_dir, fold_tag=fold_tag, logger=fold_logger)
                best_metric = float(fold_result["best_metric"])
                bests.append(best_metric)
                result_rows.append(fold_result["result_row"])
                root_logger.log(
                    {f"summary/{fold_tag}/best_val_meta_f1": float(best_metric)}, step=(i + 1) * steps_per_fold
                )

            mean_best = sum(bests) / len(bests)
            var_best = sum((x - mean_best) ** 2 for x in bests) / len(bests)
            std_best = var_best**0.5
            root_logger.log(
                {
                    "summary/cv/best_val_meta_f1_mean": float(mean_best),
                    "summary/cv/best_val_meta_f1_std": float(std_best),
                    "summary/cv/folds": len(bests),
                },
                step=len(folds) * steps_per_fold + 1,
            )
            print("\nCross-validation summary:", flush=True)
            print(f"  folds={len(bests)}", flush=True)
            print(f"  best val/meta/f1 mean={mean_best:.6f}", flush=True)
            print(f"  best val/meta/f1 std={std_best:.6f}", flush=True)
            _write_default_cv_result_csvs(args=args, manifest_path=resolved_manifest_path, rows=result_rows)
            return
        finally:
            root_logger.stop()

    _seed_everything(int(args.seed))
    validate_prepared_experiment(args.exp_dir, require_market_data=require_market_data)
    run_single_fold(args, exp_dir=args.exp_dir, fold_tag=None)


if __name__ == "__main__":
    main()
