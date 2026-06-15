from types import SimpleNamespace

import torch

from kvant.ml_framework.train.trainer import Trainer, TrainConfig


class _Evaluator:
    def __init__(self) -> None:
        self.epochs = []

    def evaluate_all(self, model, loaders, *, step=None):
        self.epochs.append(step)
        return {"val/meta/f1": float(step)}


class _Logger:
    def __init__(self) -> None:
        self.payloads = []

    def log(self, metrics, *, step=None):
        self.payloads.append((metrics, step))


def test_trainer_only_runs_expensive_metrics_on_full_evaluation_epochs() -> None:
    evaluator = _Evaluator()
    logger = _Logger()
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        optimizer=SimpleNamespace(),
        criterion=SimpleNamespace(),
        device=torch.device("cpu"),
        evaluator=evaluator,
        logger=logger,
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
    assert set(logger.payloads[1][0]) == {"epoch", "train/training/loss", "val/training/loss"}
