from kvant.ml_framework.logging.wandb_logger import WandbLogger


class _DummyConfig:
    def __init__(self) -> None:
        self.updated = []

    def update(self, payload, allow_val_change=True) -> None:
        self.updated.append((payload, allow_val_change))


class _DummyRun:
    def __init__(self) -> None:
        self.project = "proj"
        self.name = "run"
        self.config = _DummyConfig()
        self.logged = []
        self.defined_metrics = []
        self.finished = False

    def define_metric(self, name: str) -> None:
        self.defined_metrics.append(name)

    def log(self, payload, step=None) -> None:
        self.logged.append((payload, step))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_uses_run_object_for_metrics_config_and_finish() -> None:
    run = _DummyRun()
    logger = WandbLogger(project="proj", run=run, compact_metrics=False)

    logger.log_config({"alpha": 1})
    logger.log({"metric": 2.0}, step=3)
    logger.stop()

    assert run.defined_metrics == ["global_epoch", "epoch"]
    assert run.config.updated == [({"alpha": 1}, True)]
    assert run.logged == [({"metric": 2.0, "epoch": 3, "global_epoch": 3}, 3)]
    assert run.finished is True


def test_wandb_logger_compact_metrics_keeps_informative_metrics_only() -> None:
    run = _DummyRun()
    logger = WandbLogger(project="proj", run=run)

    logger.log(
        {
            "train/training/loss": 0.5,
            "val/classification/f1_macro": 0.6,
            "test/meta/f1": 0.7,
            "test/portfolio/total_return_pct": 2.0,
            "test/precision_class_0": 0.4,
            "test/paper/tp": 12,
        },
        step=3,
    )

    payload, step = run.logged[0]
    assert step == 3
    assert payload == {
        "train/training/loss": 0.5,
        "val/classification/f1_macro": 0.6,
        "test/meta/f1": 0.7,
        "test/portfolio/total_return_pct": 2.0,
        "epoch": 3,
        "global_epoch": 3,
    }
