from types import SimpleNamespace

import torch

from kvant.ml_framework.train.trainer import Trainer, TrainConfig


class _Evaluator:
    def __init__(self, values: list[float] | None = None) -> None:
        self.epochs = []
        self.values = list(values or [])

    def evaluate_all(self, model, loaders, *, step=None):
        self.epochs.append(step)
        if self.values:
            return {"val/meta/f1": float(self.values.pop(0))}
        return {"val/meta/f1": float(step)}


class _Logger:
    def __init__(self) -> None:
        self.payloads = []

    def log(self, metrics, *, step=None):
        self.payloads.append((metrics, step))


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_trainer_only_runs_expensive_metrics_on_full_evaluation_epochs() -> None:
    evaluator = _Evaluator()
    logger = _Logger()
    scheduler = _Scheduler()
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        criterion=SimpleNamespace(),
        device=torch.device("cpu"),
        evaluator=evaluator,
        logger=logger,
        scheduler=scheduler,
    )
    trainer.train_one_epoch = lambda loader, **kwargs: 0.5
    loss_calls = []
    trainer.mean_loss = lambda loader: loss_calls.append(loader.name) or 0.25

    train = SimpleNamespace(name="train", dataset=[1])
    train_eval = SimpleNamespace(name="train_eval", dataset=[1])
    val = SimpleNamespace(name="val", dataset=[1])
    test = SimpleNamespace(name="test", dataset=[1])

    trainer.fit(
        train_loader=train,
        train_eval_loader=train_eval,
        val_loader=val,
        test_loader=test,
        cfg=TrainConfig(epochs=6, checkpoint_metric="val/meta/f1", full_eval_every=5),
    )

    assert evaluator.epochs == [1, 5, 6]
    assert loss_calls.count("val") == 6
    assert loss_calls.count("train_eval") == 0
    assert loss_calls.count("test") == 0
    assert scheduler.steps == 6
    assert set(logger.payloads[1][0]) == {"epoch", "train/training/loss", "train/lr", "val/training/loss"}


def test_trainer_stops_early_after_patience_exhausted() -> None:
    evaluator = _Evaluator(values=[0.6, 0.59, 0.58])
    logger = _Logger()
    scheduler = _Scheduler()
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        criterion=SimpleNamespace(),
        device=torch.device("cpu"),
        evaluator=evaluator,
        logger=logger,
        scheduler=scheduler,
    )
    trainer.train_one_epoch = lambda loader, **kwargs: 0.5
    trainer.mean_loss = lambda loader: 0.25

    data = SimpleNamespace(name="train", dataset=[1])

    out = trainer.fit(
        train_loader=data,
        train_eval_loader=data,
        val_loader=data,
        test_loader=data,
        cfg=TrainConfig(
            epochs=10,
            checkpoint_metric="val/meta/f1",
            full_eval_every=1,
            early_stopping_patience=2,
        ),
    )

    assert evaluator.epochs == [1, 2, 3]
    assert scheduler.steps == 3
    assert out["best_metric"] == 0.6
    assert out["best_epoch"] == 1
    assert out["epochs_ran"] == 3
    assert out["stopped_early"] is True
