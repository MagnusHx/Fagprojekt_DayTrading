from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kvant.ml_framework.run_validation import (
    git_commit_or_none,
    validate_cv_manifest,
    validate_prepared_experiment,
)
from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.models import create_model
from kvant.ml_framework.train import Trainer, TrainConfig, ExperimentEvaluator, EvalConfig
from kvant.ml_framework.train.utils import class_weights_from_dataset
from kvant.ml_framework.logging import WandbLogger

import os
from dotenv import load_dotenv

load_dotenv()
default_project = os.environ.get("WANDB_PROJECT", "Kvant")
entity = os.environ.get("WANDB_ENTITY", "s245509-danmarks-tekniske-universitet-dtu")


def parse_args() -> argparse.Namespace:
    from kvant.ml_prepare_data import prepared_data_root

    with open(prepared_data_root / "last_experiment.txt", "r") as f:
        exp_id = f.read().strip()

    from kvant.ml_prepare_data import prepared_data_root

    default_exp_dir = prepared_data_root / exp_id
    default_cv_manifest = None
    cv_ptr = prepared_data_root / "last_experiment_cv_manifest.txt"
    if cv_ptr.exists():
        p = Path(cv_ptr.read_text().strip())
        if p.exists():
            default_cv_manifest = p
    # default_exp_dir = Path("../src/kvant/ml_framework/prepared") / exp_id

    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=False, default=default_exp_dir)
    p.add_argument("--cv-manifest", type=Path, required=False, default=default_cv_manifest)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
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
    p.add_argument("--no-return-stats", action="store_true")
    p.add_argument("--initial-portfolio", type=float, default=1.0)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--risk-free-rate", type=float, default=0.0314)
    p.add_argument("--days-per-year", type=float, default=365.0)
    p.add_argument(
        "--trade-confidence-threshold",
        type=float,
        default=0.5,
        help="Legacy alias for --trade-action-threshold; kept for backwards compatibility.",
    )
    p.add_argument(
        "--trade-action-threshold",
        type=float,
        default=None,
        help="Minimum p(up)+p(down) required before the model is allowed to trade.",
    )
    p.add_argument(
        "--trade-direction-threshold",
        type=float,
        default=0.6,
        help="Directional confidence threshold on q_up = p(up)/(p(up)+p(down)); "
        "shorts use the symmetric lower band 1-threshold.",
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
        "--no-save-best-checkpoint",
        action="store_true",
        help="Disable writing a local best-checkpoint bundle for offline metric reconciliation.",
    )
    args = p.parse_args()
    args.trade_action_threshold = (
        args.trade_action_threshold
        if args.trade_action_threshold is not None
        else args.trade_confidence_threshold
    )
    args.trade_confidence_threshold = args.trade_action_threshold
    if not (0.0 <= args.trade_action_threshold <= 1.0):
        raise SystemExit("--trade-action-threshold must be between 0 and 1.")
    if not (0.5 <= args.trade_direction_threshold <= 1.0):
        raise SystemExit("--trade-direction-threshold must be between 0.5 and 1.0.")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("epochs and batch sizes must be positive.")
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
        "checkpoint_metric": "val/accuracy",
        "label_semantics": exp.label_semantics,
        "label_ids": list(exp.label_ids),
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
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "trade_confidence_threshold": args.trade_confidence_threshold,
            "trade_action_threshold": args.trade_action_threshold,
            "trade_direction_threshold": args.trade_direction_threshold,
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
            "class_weights": None,
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "trade_confidence_threshold": args.trade_confidence_threshold,
            "trade_action_threshold": args.trade_action_threshold,
            "trade_direction_threshold": args.trade_direction_threshold,
            "backtest_width_minutes": None,
            "backtest_barrier_height": None,
        },
        enable_optional_media=args.wandb_optional_media,
        per_ticker_chart_limit=args.topk_ticker_plots,
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


def _runtime_metadata(args: argparse.Namespace, *, exp_dir: Path, fold_tag: str | None, preflight: dict) -> dict:
    label_regime = preflight.get("label_regime")
    return {
        "git_commit": git_commit_or_none(Path(__file__).resolve().parents[4]),
        "exp_dir": str(exp_dir),
        "fold_tag": fold_tag,
        "label_regime": label_regime,
        "model_name": args.model,
        "model_kwargs": _model_kwargs(args),
        "seed_python": int(args.seed),
        "seed_numpy": int(args.seed),
        "seed_torch": int(args.seed),
        "epochs": int(args.epochs),
        "transaction_cost": float(args.transaction_cost),
        "trade_action_threshold": float(args.trade_action_threshold),
        "trade_direction_threshold": float(args.trade_direction_threshold),
        "require_market_data": bool(not args.no_return_stats),
        "preflight": preflight,
    }


def run_single_fold(
    args: argparse.Namespace,
    exp_dir: Path,
    fold_tag: str | None = None,
    logger: WandbLogger | None = None,
) -> float:
    preflight = validate_prepared_experiment(exp_dir, require_market_data=not args.no_return_stats).to_jsonable()
    runtime_dir = Path("artifacts/run_debug") / Path(exp_dir).name
    _write_json_artifact(runtime_dir / "preflight.json", preflight)

    exp = PreparedExperiment(exp_dir)
    dl_train, dl_val, dl_test = exp.get_loaders(
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
    ds_train, ds_val, ds_test = exp.get_datasets()

    if args.print_dataset_summary:
        for ds, split_name in [(ds_train, "train"), (ds_val, "val"), (ds_test, "test")]:
            print(f"Dataset {split_name}")
            ds.summary(display=True)
            print("-" * 10, "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=args.model,
        n_features=exp.store.n_features,
        n_classes=exp.n_classes,
        **_model_kwargs(args),
    ).to(device)
    labeler_cfg = exp.cfg.get("labeler", {})

    w = class_weights_from_dataset(ds_train, n_classes=exp.n_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    owns_logger = logger is None
    logger = logger or _make_logger(args, exp_dir=exp_dir, fold_tag=fold_tag)
    logger.log_config(
        {
            "exp_dir": str(exp_dir),
            "fold_tag": fold_tag,
            "epochs": args.epochs,
            "lr": args.lr,
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
            "class_weights": w.tolist(),
            "n_classes": exp.n_classes,
            "label_semantics": exp.label_semantics,
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
            "trade_confidence_threshold": args.trade_confidence_threshold,
            "trade_action_threshold": args.trade_action_threshold,
            "trade_direction_threshold": args.trade_direction_threshold,
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
                initial_portfolio=args.initial_portfolio,
                transaction_cost=args.transaction_cost,
                risk_free_rate=args.risk_free_rate,
                days_per_year=args.days_per_year,
                trade_confidence_threshold=args.trade_confidence_threshold,
                trade_action_threshold=args.trade_action_threshold,
                trade_direction_threshold=args.trade_direction_threshold,
                backtest_width_minutes=int(labeler_cfg.get("width_minutes", 0)),
                backtest_barrier_height=float(labeler_cfg.get("height", 0.0)),
                labels=exp.label_ids,
                label_semantics=exp.label_semantics,
            ),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            evaluator=evaluator,
            logger=logger,
        )

        cfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            checkpoint_metric="val/accuracy",
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
        )
        logger.child(namespace="best").log(best_metrics, step=args.epochs + 1)
        return float(out["best_metric"])
    finally:
        if owns_logger:
            logger.stop()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[4]
    require_market_data = not args.no_return_stats

    if args.cv_manifest is not None and args.cv_manifest.exists():
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
            steps_per_fold = args.epochs + 2
            for i, fold in enumerate(folds):
                fold_idx = int(fold["fold_idx"])
                exp_dir = Path(fold["exp_dir"])
                fold_tag = f"fold{fold_idx:02d}"
                fold_seed = int(base_seed + fold_idx)
                _seed_everything(fold_seed)
                print(f"\n=== Training {fold_tag} on {exp_dir} (seed={fold_seed}) ===")
                fold_logger = root_logger.child(namespace=fold_tag, step_offset=i * steps_per_fold)
                fold_logger.log_config({"fold_seed": fold_seed})
                fold_args = argparse.Namespace(**vars(args))
                fold_args.seed = fold_seed
                best_metric = run_single_fold(fold_args, exp_dir=exp_dir, fold_tag=fold_tag, logger=fold_logger)
                bests.append(best_metric)
                root_logger.log(
                    {f"summary/{fold_tag}/best_val_accuracy": float(best_metric)}, step=(i + 1) * steps_per_fold
                )

            mean_best = sum(bests) / len(bests)
            var_best = sum((x - mean_best) ** 2 for x in bests) / len(bests)
            std_best = var_best**0.5
            root_logger.log(
                {
                    "summary/cv/best_val_accuracy_mean": float(mean_best),
                    "summary/cv/best_val_accuracy_std": float(std_best),
                    "summary/cv/folds": len(bests),
                },
                step=len(folds) * steps_per_fold + 1,
            )
            print("\nCross-validation summary:")
            print(f"  folds={len(bests)}")
            print(f"  best val/accuracy mean={mean_best:.6f}")
            print(f"  best val/accuracy std={std_best:.6f}")
            return
        finally:
            root_logger.stop()

    _seed_everything(int(args.seed))
    validate_prepared_experiment(args.exp_dir, require_market_data=require_market_data)
    run_single_fold(args, exp_dir=args.exp_dir, fold_tag=None)


if __name__ == "__main__":
    main()
