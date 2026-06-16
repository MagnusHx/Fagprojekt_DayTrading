from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import time

from .evaluator import ExperimentEvaluator


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_batch_size: int = 256
    eval_batch_size: int = 512
    checkpoint_metric: str = "val/accuracy"

    full_eval_every: int = 1
    progress_batches: int = 250
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        evaluator: Optional[ExperimentEvaluator] = None,
        logger: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.evaluator = evaluator
        self.logger = logger
        self.scheduler = scheduler

    def current_lr(self) -> float:
        """Return the learning rate for the first optimizer parameter group."""
        param_groups = getattr(self.optimizer, "param_groups", [])
        if not param_groups:
            return 0.0
        return float(param_groups[0].get("lr", 0.0))

    def train_one_epoch(
        self,
        loader: DataLoader,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
        progress_batches: int = 250,
    ) -> float:
        """Train the model for one epoch and return mean batch loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        total_batches = len(loader)
        started_at = time.time()
        last_report_at = started_at

        for batch_idx, batch in enumerate(loader, start=1):
            x, y = batch[0], batch[1]  # ignore tid/tpos

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            valid_mask = y >= 0
            if not bool(torch.any(valid_mask)):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            should_report = (
                int(total_batches) > 0
                and int(batch_idx) < int(total_batches)
                and int(batch_idx) % max(1, int(progress_batches)) == 0
            )
            if should_report:
                avg_loss = total_loss / max(n_batches, 1)
                elapsed = time.time() - started_at
                delta = time.time() - last_report_at
                epoch_label = (
                    f"epoch {int(epoch)}/{int(total_epochs)}"
                    if epoch is not None and total_epochs is not None
                    else f"epoch {int(epoch)}"
                    if epoch is not None
                    else "epoch"
                )
                print(
                    f"{epoch_label} progress batch={batch_idx}/{total_batches} "
                    f"avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s (+{delta:.1f}s)",
                    flush=True,
                )
                last_report_at = time.time()

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def mean_loss(self, loader: DataLoader) -> float:
        """Compute the mean batch loss for a loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            valid_mask = y >= 0
            if not bool(torch.any(valid_mask)):
                continue

            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        *,
        train_loader: DataLoader,
        train_eval_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        cfg: TrainConfig,
    ) -> Dict[str, Any]:
        best_state = None
        best_metric = -float("inf")
        best_epoch: int | None = None
        epochs_ran = 0
        evaluations_without_improvement = 0
        stopped_early = False

        import time
        import numpy as np
        from collections import defaultdict

        tspend = defaultdict(list)

        def do_full_eval(epoch: int) -> bool:
            every = max(1, int(cfg.full_eval_every))
            return (epoch % every == 0) or (epoch == 1) or (epoch == cfg.epochs)

        for ep in range(1, cfg.epochs + 1):
            epochs_ran = ep
            t0 = time.time()
            print(f"Starting epoch {ep}/{cfg.epochs}...", flush=True)
            train_loss = self.train_one_epoch(
                train_loader,
                epoch=ep,
                total_epochs=cfg.epochs,
                progress_batches=cfg.progress_batches,
            )
            tspend["train"].append(time.time() - t0)

            metrics: Dict[str, Any] = {}
            metrics["epoch"] = int(ep)
            metrics["train/training/loss"] = float(train_loss)
            metrics["train/lr"] = self.current_lr()

            full_eval = (self.evaluator is not None) and do_full_eval(ep)

            if val_loader is not None:
                metrics["val/training/loss"] = self.mean_loss(val_loader)

            t0 = time.time()

            if full_eval:
                loaders = {}
                if train_eval_loader is not None:
                    loaders["train"] = train_eval_loader
                if val_loader is not None:
                    loaders["val"] = val_loader
                if test_loader is not None:
                    loaders["test"] = test_loader

                metrics.update(self.evaluator.evaluate_all(self.model, loaders, step=ep))

            tspend["eval"].append(time.time() - t0)

            if self.logger is not None:
                self.logger.log(dict(metrics), step=ep)

            metric_val = float(metrics.get(cfg.checkpoint_metric, -float("inf")))
            metric_available = cfg.checkpoint_metric in metrics
            if metric_available:
                improved = metric_val > (best_metric + float(cfg.early_stopping_min_delta))
                if improved:
                    best_metric = metric_val
                    best_epoch = ep
                    evaluations_without_improvement = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                elif cfg.early_stopping_patience is not None:
                    evaluations_without_improvement += 1

            totals = {k: sum(v) for k, v in tspend.items()}
            totals = {k + "(pct)": v / sum(totals.values()) for k, v in totals.items()}
            timing_metrics = {"train/epoch": np.mean(tspend["train"])} | totals
            timing_metrics = [f"{k}: {np.mean(v):.2f}" for k, v in timing_metrics.items()]

            print(
                f"epoch={ep:04d} train_loss={train_loss:.4f} "
                f"lr={metrics['train/lr']:.6g} {cfg.checkpoint_metric}={metric_val:.4f} best={best_metric:.4f} "
                f"[{' '.join(timing_metrics)}]",
                flush=True,
            )
            if self.scheduler is not None:
                self.scheduler.step()

            if (
                metric_available
                and cfg.early_stopping_patience is not None
                and evaluations_without_improvement >= cfg.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "Early stopping triggered "
                    f"at epoch {ep}: no {cfg.checkpoint_metric} improvement for "
                    f"{evaluations_without_improvement} evaluation(s).",
                    flush=True,
                )
                break

        return {
            "best_state": best_state,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "stopped_early": stopped_early,
        }
