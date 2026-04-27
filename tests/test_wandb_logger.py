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
    logger = WandbLogger(project="proj", run=run)

    logger.log_config({"alpha": 1})
    logger.log({"metric": 2.0}, step=3)
    logger.stop()

    assert run.defined_metrics == ["global_epoch", "epoch"]
    assert run.config.updated == [({"alpha": 1}, True)]
    assert run.logged == [({"metric": 2.0, "epoch": 3, "global_epoch": 3}, 3)]
    assert run.finished is True

