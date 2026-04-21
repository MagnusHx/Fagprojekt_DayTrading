from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kvant.labels import label_semantics_payload
from kvant.ml_framework.logging import wandb_logger as wandb_logger_module
from kvant.ml_framework.logging.wandb_logger import WandbLogger
from kvant.ml_framework.train.classification_metrics import classification_metrics
from kvant.ml_framework.train.evaluator import EvalConfig, ExperimentEvaluator


class _PredictionDataset(Dataset):
    def __init__(self, y_true: list[int], tids: list[int], tpos: list[int]) -> None:
        self._y_true = [int(v) for v in y_true]
        self._tids = [int(v) for v in tids]
        self._tpos = [int(v) for v in tpos]

    def __len__(self) -> int:
        return len(self._y_true)

    def __getitem__(self, idx: int):
        x = torch.tensor([[[float(idx)]]], dtype=torch.float32).squeeze(0)
        y = torch.tensor(self._y_true[idx], dtype=torch.long)
        tid = torch.tensor(self._tids[idx], dtype=torch.int32)
        tpos = torch.tensor(self._tpos[idx], dtype=torch.int32)
        return x, y, tid, tpos


class _IndexLogitModel(torch.nn.Module):
    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x[:, 0, 0].long()
        return self.logits[idx]


class _FakeStore:
    def __init__(self, *, tickers_all: list[str], metadata: dict[tuple[int, int], dict | None], market_data: dict[int, dict]):
        self.tickers_all = list(tickers_all)
        self._metadata = dict(metadata)
        self._market_data = dict(market_data)

    def metadata_for_index(self, index: np.ndarray):
        return [self._metadata[(int(tid), int(tpos))] for tid, tpos in index]

    def market_data(self, tid: int):
        return self._market_data[int(tid)]

    def ticker(self, tid: int) -> str:
        return self.tickers_all[int(tid)]


def _market_data_from_times(times: list[str], *, opens: list[float], highs: list[float], lows: list[float], closes: list[float]):
    return {
        "timestamp": np.asarray([np.datetime64(t) for t in times]),
        "open": np.asarray(opens, dtype=np.float32),
        "high": np.asarray(highs, dtype=np.float32),
        "low": np.asarray(lows, dtype=np.float32),
        "close": np.asarray(closes, dtype=np.float32),
        "volume": np.ones(len(times), dtype=np.float32),
    }


def test_evaluator_logs_semantic_metric_groups_for_binary_directional_runs() -> None:
    dataset = _PredictionDataset(y_true=[0, 1, 1], tids=[0, 0, 0], tpos=[0, 1, 2])
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = _IndexLogitModel(
        logits=[
            [3.0, 0.1],   # down
            [0.1, 3.0],   # up
            [0.0, 0.0],   # abstain after threshold
        ]
    )
    store = _FakeStore(
        tickers_all=["AAA"],
        metadata={
            (0, 0): {"label": 0, "bar_open_time": "2024-01-01T00:00:00+00:00", "bar_close_time": "2024-01-02T00:00:00+00:00", "pnl_fraction": 0.05},
            (0, 1): {"label": 2, "bar_open_time": "2024-01-02T00:00:00+00:00", "bar_close_time": "2024-01-03T00:00:00+00:00", "pnl_fraction": 0.05},
            (0, 2): {"label": 2, "bar_open_time": "2024-01-03T00:00:00+00:00", "bar_close_time": "2024-01-04T00:00:00+00:00", "pnl_fraction": 0.05},
        },
        market_data={
            0: _market_data_from_times(
                ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
                opens=[100.0, 100.0, 100.0],
                highs=[100.0, 106.0, 100.0],
                lows=[94.0, 100.0, 100.0],
                closes=[95.0, 105.0, 100.0],
            )
        },
    )
    evaluator = ExperimentEvaluator(
        store=store,
        device=torch.device("cpu"),
        cfg=EvalConfig(
            trade_action_threshold=0.6,
            trade_direction_threshold=0.6,
            backtest_width_minutes=1439,
            backtest_barrier_height=0.05,
            labels=(0, 1),
            label_semantics=label_semantics_payload(drop_time_exit_label=True),
        ),
    )

    metrics, rows, cm, profit_curve = evaluator.evaluate_split("test", model, loader, step=1)

    assert metrics["test/accuracy"] == metrics["test/cls/accuracy"]
    assert metrics["test/f1_macro"] == metrics["test/cls/f1_macro"]
    assert metrics["test/decision/trade_action_probability_informative"] == 0
    assert metrics["test/decision/abstained_prediction_rate_pct"] == pytest.approx(100.0 / 3.0)
    assert metrics["test/decision/acted_prediction_accuracy"] == 1.0
    assert metrics["test/decision/directional_acted_accuracy"] == 1.0
    assert metrics["test/execution/n_trade_signals_raw"] == 2
    assert metrics["test/execution/n_executed_trades"] == 2
    assert metrics["test/paper/n_executed_trades"] == 2
    assert metrics["test/paper/executed_trade_gross_return_avg_pct"] == 5.0
    assert metrics["test/paper/transaction_cost_total_pct"] == 0.4
    assert cm.shape == (2, 2)
    assert isinstance(rows, list)
    assert profit_curve is not None


def test_evaluator_logs_semantic_metric_groups_for_three_class_runs() -> None:
    dataset = _PredictionDataset(y_true=[0, 1, 2], tids=[0, 0, 0], tpos=[0, 1, 2])
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = _IndexLogitModel(
        logits=[
            [3.0, 0.1, 0.0],   # down
            [0.1, 3.0, 0.0],   # exit
            [0.0, 0.1, 3.0],   # up
        ]
    )
    store = _FakeStore(
        tickers_all=["AAA"],
        metadata={
            (0, 0): {"label": 0, "bar_open_time": "2024-01-01T00:00:00+00:00", "bar_close_time": "2024-01-02T00:00:00+00:00", "pnl_fraction": 0.05},
            (0, 1): {"label": 1, "bar_open_time": "2024-01-02T00:00:00+00:00", "bar_close_time": "2024-01-03T00:00:00+00:00", "pnl_fraction": 0.00},
            (0, 2): {"label": 2, "bar_open_time": "2024-01-03T00:00:00+00:00", "bar_close_time": "2024-01-04T00:00:00+00:00", "pnl_fraction": 0.05},
        },
        market_data={
            0: _market_data_from_times(
                ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
                opens=[100.0, 100.0, 100.0],
                highs=[100.0, 100.0, 106.0],
                lows=[94.0, 100.0, 100.0],
                closes=[95.0, 100.0, 105.0],
            )
        },
    )
    evaluator = ExperimentEvaluator(
        store=store,
        device=torch.device("cpu"),
        cfg=EvalConfig(
            trade_action_threshold=0.6,
            trade_direction_threshold=0.6,
            backtest_width_minutes=1439,
            backtest_barrier_height=0.05,
            labels=(0, 1, 2),
            label_semantics=label_semantics_payload(drop_time_exit_label=False),
        ),
    )

    metrics, rows, cm, profit_curve = evaluator.evaluate_split("test", model, loader, step=1)

    assert metrics["test/accuracy"] == metrics["test/cls/accuracy"] == 1.0
    assert metrics["test/decision/trade_action_probability_informative"] == 1
    assert metrics["test/decision/abstained_prediction_rate_pct"] == pytest.approx(100.0 / 3.0)
    assert metrics["test/decision/acted_on_exit_truth_pct"] == 0.0
    assert metrics["test/execution/n_trade_signals_raw"] == 2
    assert metrics["test/paper/n_executed_trades"] == 2
    assert cm.shape == (3, 3)
    assert isinstance(rows, list)
    assert profit_curve is not None


class _FakeConfig(dict):
    def update(self, payload, allow_val_change=True):
        super().update(payload)


class _FakeRun:
    def __init__(self, project: str = "proj", name: str = "run") -> None:
        self.project = project
        self.name = name
        self.config = _FakeConfig()
        self.finished = False

    def finish(self) -> None:
        self.finished = True


class _FakeTable:
    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = []

    def add_data(self, *args):
        self.rows.append(tuple(args))


class _FakeWandb:
    def __init__(self):
        self.logged = []
        self.Table = _FakeTable
        self.Image = lambda fig: {"image": fig}
        self.plot = SimpleNamespace(
            line=lambda *args, **kwargs: {"plot": "line", "args": args, "kwargs": kwargs},
            line_series=lambda **kwargs: {"plot": "line_series", "kwargs": kwargs},
        )
        self.config = _FakeConfig()

    def init(self, **kwargs):
        return _FakeRun(project=kwargs.get("project", "proj"), name=kwargs.get("name", "run"))

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, payload, step=None):
        self.logged.append((payload, step))


class _FailingOptionalWandb(_FakeWandb):
    def log(self, payload, step=None):
        if any("charts/per_ticker/" in key for key in payload):
            raise RuntimeError("optional chart upload failed")
        super().log(payload, step=step)


def test_wandb_logger_does_not_require_api_client(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(wandb_logger_module, "wandb", fake_wandb)

    logger = WandbLogger(project="proj", name="run", api_timeout=1)

    assert logger.run.project == "proj"


class _FakeSummaryDataset:
    def __init__(self, *, class_ids: tuple[int, ...], ticker: str = "AAA") -> None:
        self._class_ids = class_ids
        self._ticker = ticker

    def summary(self, display=False):
        y_counts = {label: 2 for label in self._class_ids}
        return {
            "overall": {"n": sum(y_counts.values()), "first_ts": "2024-01-01T00:00:00+00:00", "last_ts": "2024-01-02T00:00:00+00:00", "y_counts": y_counts},
            "per_ticker": {
                self._ticker: {
                    "tid": 0,
                    "n": sum(y_counts.values()),
                    "first_ts": "2024-01-01T00:00:00+00:00",
                    "last_ts": "2024-01-02T00:00:00+00:00",
                    "y_counts": y_counts,
                }
            },
        }


def test_wandb_logger_respects_label_regime_for_inventory_and_confusion(monkeypatch, tmp_path: Path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(wandb_logger_module, "wandb", fake_wandb)

    logger = WandbLogger(project="proj", name="run", api_timeout=1)
    exp = SimpleNamespace(
        exp_dir=tmp_path / "exp",
        store=SimpleNamespace(
            label_ids=(0, 1),
            class_names=["Down barrier hit (y=0)", "Up barrier hit (y=1)"],
            label_meanings={0: "down barrier hit", 1: "up barrier hit"},
            tickers_all=["AAA"],
        ),
    )
    loaders = {
        "train": SimpleNamespace(dataset=_FakeSummaryDataset(class_ids=(0, 1))),
        "val": SimpleNamespace(dataset=_FakeSummaryDataset(class_ids=(0, 1))),
        "test": SimpleNamespace(dataset=_FakeSummaryDataset(class_ids=(0, 1))),
    }

    logger.setup(exp=exp, loaders=loaders)
    logger.log({"epoch": 1, "_confusion_counts": {"test": np.asarray([[2, 1], [0, 3]], dtype=np.int64)}}, step=1)

    inventory_tables = [payload["data/metric_inventory"] for payload, _ in fake_wandb.logged if "data/metric_inventory" in payload]
    assert inventory_tables
    assert len(inventory_tables[0].rows) > 10

    label_tables = [payload["data/label_meanings"] for payload, _ in fake_wandb.logged if "data/label_meanings" in payload]
    assert label_tables
    assert len(label_tables[0].rows) == 2

    cm_tables = [
        payload["perf/confusion_matrix_normalized/test"]
        for payload, _ in fake_wandb.logged
        if "perf/confusion_matrix_normalized/test" in payload
    ]
    assert cm_tables
    assert len(cm_tables[0].rows) == 4


def test_wandb_logger_skips_optional_per_ticker_media_by_default(monkeypatch, tmp_path: Path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(wandb_logger_module, "wandb", fake_wandb)

    logger = WandbLogger(project="proj", name="run", api_timeout=1, enable_optional_media=False, per_ticker_chart_limit=5)
    logger.log(
        {
            "epoch": 1,
            "_per_ticker_rows": [
                {
                    "epoch": 1,
                    "split": "test",
                    "tid": 0,
                    "ticker": "AAA",
                    "acc": 0.5,
                    "n": 2,
                    "buy_n_trades": 1,
                    "buy_profit_avg_per_trade_pct": 1.0,
                    "buy_profit_total_pct": 1.0,
                    "short_n_trades": 0,
                    "short_profit_avg_per_trade_pct": 0.0,
                    "short_profit_total_pct": 0.0,
                }
            ],
        },
        step=1,
    )

    assert not any("charts/per_ticker/" in next(iter(payload.keys())) for payload, _ in fake_wandb.logged)


def test_wandb_logger_logs_per_ticker_table_only_for_best_namespace(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(wandb_logger_module, "wandb", fake_wandb)

    logger = WandbLogger(project="proj", name="run", api_timeout=1)
    logger.log(
        {
            "epoch": 1,
            "_per_ticker_rows": [
                {
                    "epoch": 1,
                    "split": "test",
                    "tid": 0,
                    "ticker": "AAA",
                    "acc": 0.5,
                    "n": 2,
                    "buy_n_trades": 1,
                    "buy_profit_avg_per_trade_pct": 1.0,
                    "buy_profit_total_pct": 1.0,
                    "short_n_trades": 0,
                    "short_profit_avg_per_trade_pct": 0.0,
                    "short_profit_total_pct": 0.0,
                }
            ],
        },
        step=1,
    )
    assert not any("perf/per_ticker_table" in payload for payload, _ in fake_wandb.logged)

    best_logger = logger.child(namespace="best")
    best_logger.log(
        {
            "epoch": 2,
            "_per_ticker_rows": [
                {
                    "epoch": 2,
                    "split": "test",
                    "tid": 0,
                    "ticker": "AAA",
                    "acc": 0.75,
                    "n": 2,
                    "buy_n_trades": 1,
                    "buy_profit_avg_per_trade_pct": 2.0,
                    "buy_profit_total_pct": 2.0,
                    "short_n_trades": 0,
                    "short_profit_avg_per_trade_pct": 0.0,
                    "short_profit_total_pct": 0.0,
                }
            ],
        },
        step=2,
    )
    assert any("best/perf/per_ticker_table" in payload for payload, _ in fake_wandb.logged)


def test_wandb_logger_degrades_gracefully_when_optional_media_fails(monkeypatch, tmp_path: Path) -> None:
    fake_wandb = _FailingOptionalWandb()
    monkeypatch.setattr(wandb_logger_module, "wandb", fake_wandb)

    logger = WandbLogger(project="proj", name="run", api_timeout=1, enable_optional_media=True, per_ticker_chart_limit=1)
    logger._tickers_to_chart = ["AAA (tid=0)"]
    logger.log(
        {
            "epoch": 1,
            "_per_ticker_rows": [
                {
                    "epoch": 1,
                    "split": "train",
                    "tid": 0,
                    "ticker": "AAA",
                    "acc": 0.5,
                    "n": 2,
                    "buy_n_trades": 1,
                    "buy_profit_avg_per_trade_pct": 1.0,
                    "buy_profit_total_pct": 1.0,
                    "short_n_trades": 0,
                    "short_profit_avg_per_trade_pct": 0.0,
                    "short_profit_total_pct": 0.0,
                }
            ],
        },
        step=1,
    )
    logger.log({"epoch": 2}, step=2)

    assert logger._optional_logging_enabled is True
    assert not any("perf/per_ticker_table" in payload for payload, _ in fake_wandb.logged)


def test_classification_metrics_handle_empty_split() -> None:
    out = classification_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), labels=(0, 1))
    assert out == {"accuracy": 0.0}


def test_classification_metrics_handle_single_true_class() -> None:
    out = classification_metrics(
        np.asarray([0, 0, 0], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
        labels=(0, 1),
    )
    assert out["accuracy"] == 2.0 / 3.0
    assert out["support_class_0"] == 3
    assert out["support_class_1"] == 0
    assert out["recall_class_1"] == 0.0
