from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.models import Conv1DClassifier
from kvant.ml_framework.train import Trainer, TrainConfig, ExperimentEvaluator, EvalConfig
from kvant.ml_framework.train.utils import class_weights_from_dataset
from kvant.ml_framework.logging import WandbLogger

import os
from dotenv import load_dotenv

load_dotenv()
project = os.environ.get("WANDB_PROJECT", "Kvant")
entity = os.environ.get("WANDB_ENTITY", "s245509-danmarks-tekniske-universitet-dtu")


def parse_args() -> argparse.Namespace:
    exp_id = "11bc7f8b735e5936"
    exp_id = "sb_L_400_w120_h2pct"
    exp_id = "sb_L_40_w120_h1.5_TBPD20"
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
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--wandb-project", type=str, default="kvant-stocks")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-api-timeout", type=int, default=29)
    p.add_argument("--no-return-stats", action="store_true")
    p.add_argument("--initial-portfolio", type=float, default=1.0)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--risk-free-rate", type=float, default=0.0314)
    p.add_argument("--days-per-year", type=float, default=365.0)
    p.add_argument("--topk-ticker-plots", type=int, default=50)
    return p.parse_args()


def _make_logger(
    args: argparse.Namespace,
    *,
    exp_dir: Path,
    fold_tag: str | None = None,
) -> WandbLogger:
    return WandbLogger(
        project=project,
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
            "L": None,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "class_weights": None,
            "initial_portfolio": args.initial_portfolio,
            "transaction_cost": args.transaction_cost,
            "risk_free_rate": args.risk_free_rate,
            "days_per_year": args.days_per_year,
        },
    )


def run_single_fold(
    args: argparse.Namespace,
    exp_dir: Path,
    fold_tag: str | None = None,
    logger: WandbLogger | None = None,
) -> float:
    exp = PreparedExperiment(exp_dir)
    dl_train, dl_val, dl_test = exp.get_loaders(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # Train eval loader (no shuffle) so "train" metrics are stable and comparable
    dl_train_eval = DataLoader(
        dl_train.dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Optional local sanity check
    ds_train, ds_val, ds_test = exp.get_datasets()

    for ds, split_name in [(ds_train, "train"), (ds_val, "val"), (ds_test, "test")]:
        print(f"Dataset {split_name}")
        ds.summary(display=True)
        print("-" * 10, "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1DClassifier(n_features=exp.store.n_features, n_classes=3).to(device)

    w = class_weights_from_dataset(ds_train, n_classes=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    owns_logger = logger is None
    logger = logger or _make_logger(args, exp_dir=exp_dir, fold_tag=fold_tag)
    if owns_logger:
        logger.log_config(
            {
                "exp_dir": str(exp_dir),
                "fold_tag": fold_tag,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "L": exp.L,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "class_weights": w.tolist(),
                "initial_portfolio": args.initial_portfolio,
                "transaction_cost": args.transaction_cost,
                "risk_free_rate": args.risk_free_rate,
                "days_per_year": args.days_per_year,
            }
        )

    # Log static dataset stats + configure ticker charts
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
            compute_paper_trading_metrics=True,
            initial_portfolio=args.initial_portfolio,
            transaction_cost=args.transaction_cost,
            risk_free_rate=args.risk_free_rate,
            days_per_year=args.days_per_year,
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

    # Restore best checkpoint
    if out["best_state"] is not None:
        model.load_state_dict(out["best_state"])

    # Final evaluation of the restored best checkpoint.
    best_metrics = evaluator.evaluate_all(
        model,
        {"train": dl_train_eval, "val": dl_val, "test": dl_test},
        step=args.epochs + 1,
    )
    logger.child(namespace="best").log(best_metrics, step=args.epochs + 1)
    if owns_logger:
        logger.stop()
    return float(out["best_metric"])


def main() -> None:
    args = parse_args()

    if args.cv_manifest is not None and args.cv_manifest.exists():
        payload = json.loads(args.cv_manifest.read_text())
        folds = payload.get("folds", [])
        if not folds:
            raise RuntimeError(f"No folds found in cv manifest: {args.cv_manifest}")

        root_logger = _make_logger(args, exp_dir=Path(payload.get("exp_dir", folds[0]["exp_dir"])), fold_tag="cv")
        root_logger.log_config(
            {
                "cv_manifest": str(args.cv_manifest),
                "cv_folds": len(folds),
            }
        )

        bests = []
        steps_per_fold = args.epochs + 2
        for i, fold in enumerate(folds):
            fold_idx = int(fold["fold_idx"])
            exp_dir = Path(fold["exp_dir"])
            fold_tag = f"fold{fold_idx:02d}"
            print(f"\n=== Training {fold_tag} on {exp_dir} ===")
            fold_logger = root_logger.child(namespace=fold_tag, step_offset=i * steps_per_fold)
            best_metric = run_single_fold(args, exp_dir=exp_dir, fold_tag=fold_tag, logger=fold_logger)
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
        root_logger.stop()
        print("\nCross-validation summary:")
        print(f"  folds={len(bests)}")
        print(f"  best val/accuracy mean={mean_best:.6f}")
        print(f"  best val/accuracy std={std_best:.6f}")
        return

    run_single_fold(args, exp_dir=args.exp_dir, fold_tag=None)


if __name__ == "__main__":
    main()
