import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kvant.labels import LABEL_DOWN, LABEL_EXIT, LABEL_UP
from kvant.ml_framework.train.decision_policy import (
    LogisticMetaLabeler,
    kelly_bet_fraction,
    meta_targets_from_predictions,
    sized_trade_decisions,
)
from kvant.ml_framework.train.evaluator import EvalConfig, ExperimentEvaluator
from kvant.ml_prepare_data.data_loading import PreparedExperiment


def _write_ticker_fixture(exp_dir, ticker: str, *, labels: list[int], pnl: list[float] | None = None) -> None:
    tdir = exp_dir / "tickers" / ticker
    tdir.mkdir(parents=True)
    n = len(labels)
    features = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    timestamps = np.asarray(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(i, "m") for i in range(n)],
        dtype="datetime64[ns]",
    )
    if pnl is None:
        pnl = [(-0.05 if y == 0 else (0.05 if y == 2 else 0.0)) for y in labels]
    label_meta = [{"label": int(y), "pnl_fraction": float(p)} for y, p in zip(labels, pnl)]
    np.save(tdir / "features.npy", features)
    np.save(tdir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(tdir / "timestamps.npy", timestamps)
    np.save(tdir / "market_data.npy", np.ones((n, 5), dtype=np.float32))
    (tdir / "label_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in label_meta))


def _write_prepared_fixture(tmp_path, *, feature_names: list[str] | None) -> PreparedExperiment:
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "tickers").mkdir()
    config = {
        "lookback_L": 2,
        "pipeline_stage": "event_outcome",
        "label_spaces": {
            "event_outcome_labels": {"0": "down", "1": "exit", "2": "up"},
            "side_labels": {"0": "down", "1": "up"},
            "meta_labels": {"0": "pass", "1": "take"},
        },
        "feature_engineer": {
            "feature_names_": feature_names,
            "mean_": [0.0, 0.0, 0.0],
            "std_": [1.0, 1.0, 1.0],
        },
        "labeler": {"width_minutes": 60, "height": 0.01, "drop_time_exit_label": False},
        "label_semantics": {"version": 1, "labels": {"0": "down", "1": "exit", "2": "up"}},
    }
    (exp_dir / "config.json").write_text(json.dumps(config))
    (exp_dir / "tickers_all.json").write_text(json.dumps(["AAA"]))
    _write_ticker_fixture(exp_dir, "AAA", labels=[0, 1, 2, 0, 2])
    np.save(exp_dir / "index_train.npy", np.asarray([[0, 2], [0, 3]], dtype=np.int64))
    np.save(exp_dir / "index_val.npy", np.asarray([[0, 4]], dtype=np.int64))
    np.save(exp_dir / "index_test.npy", np.asarray([[0, 1]], dtype=np.int64))
    return PreparedExperiment(exp_dir)


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


class _IndexLogitFeatureModel(torch.nn.Module):
    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :1]

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        idx = features[:, 0].long()
        return self.logits[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(x))


class _FakeStore:
    def __init__(self, *, tickers_all: list[str], labels: list[int], pnl: list[float]) -> None:
        self.tickers_all = list(tickers_all)
        self._metadata = {}
        times = [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(i, "D") for i in range(len(labels))]
        highs = []
        lows = []
        closes = []
        for idx, (label, pnl_fraction) in enumerate(zip(labels, pnl)):
            if label == 2:
                highs.append(106.0)
                lows.append(100.0)
                closes.append(100.0 * (1.0 + pnl_fraction))
            elif label == 0:
                highs.append(100.0)
                lows.append(94.0)
                closes.append(100.0 * (1.0 + pnl_fraction))
            else:
                highs.append(101.0)
                lows.append(99.0)
                closes.append(100.0 * (1.0 + pnl_fraction))
            self._metadata[(0, idx)] = {
                "label": int(label),
                "bar_open_time": f"2024-01-{idx + 1:02d}T00:00:00+00:00",
                "bar_close_time": f"2024-01-{idx + 2:02d}T00:00:00+00:00",
                "pnl_fraction": float(pnl_fraction),
            }
        self._market_data = {
            0: {
                "timestamp": np.asarray(times),
                "open": np.asarray([100.0] * len(labels), dtype=np.float32),
                "high": np.asarray(highs, dtype=np.float32),
                "low": np.asarray(lows, dtype=np.float32),
                "close": np.asarray(closes, dtype=np.float32),
                "volume": np.ones(len(labels), dtype=np.float32),
            }
        }

    def metadata_for_index(self, index: np.ndarray):
        return [self._metadata[(int(tid), int(tpos))] for tid, tpos in index]

    def event_labels_for_index(self, index: np.ndarray) -> np.ndarray:
        return np.asarray([self._metadata[(int(tid), int(tpos))]["label"] for tid, tpos in index], dtype=np.int64)

    def market_data(self, tid: int):
        return self._market_data[int(tid)]

    def ticker(self, tid: int) -> str:
        return self.tickers_all[int(tid)]

    def prepared_last_feature_values(self, tids, tpos, feature_name: str) -> np.ndarray:
        return np.asarray([float(int(pos) + 1) for pos in tpos], dtype=np.float32)


def test_logistic_meta_labeler_builds_configurable_feature_matrix(tmp_path) -> None:
    exp = _write_prepared_fixture(tmp_path, feature_names=["f0", "f1", "f2"])
    pred_out = {
        "tid": np.asarray([0, 0], dtype=np.int64),
        "tpos": np.asarray([2, 3], dtype=np.int64),
        "y_pred": np.asarray([0, 1], dtype=np.int64),
        "y_pred_proba": np.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64),
        "y_logits": np.asarray([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64),
        "y_embedding": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    }
    policy = LogisticMetaLabeler(feature_tokens=("proba", "logits", "embedding", "prepared_last:f1"))

    X = policy.build_feature_matrix(pred_out=pred_out, store=exp.store)

    assert X.shape == (2, 7)
    np.testing.assert_allclose(X[:, -1], np.asarray([4.0, 7.0], dtype=np.float64))


def test_logistic_meta_labeler_requires_persisted_feature_names(tmp_path) -> None:
    exp = _write_prepared_fixture(tmp_path, feature_names=None)
    pred_out = {
        "tid": np.asarray([0], dtype=np.int64),
        "tpos": np.asarray([2], dtype=np.int64),
        "y_pred": np.asarray([0], dtype=np.int64),
        "y_pred_proba": np.asarray([[0.7, 0.3]], dtype=np.float64),
        "y_logits": np.asarray([[2.0, 1.0]], dtype=np.float64),
        "y_embedding": np.asarray([[1.0, 2.0]], dtype=np.float64),
    }
    policy = LogisticMetaLabeler(feature_tokens=("prepared_last:f1",))

    with pytest.raises(RuntimeError, match="missing persisted feature names"):
        policy.build_feature_matrix(pred_out=pred_out, store=exp.store)


def test_meta_targets_follow_realized_return_of_predicted_side() -> None:
    store = _FakeStore(tickers_all=["AAA"], labels=[0, 2, 1], pnl=[-0.05, 0.05, 0.01])
    pred_out = {
        "tid": np.asarray([0, 0, 0], dtype=np.int64),
        "tpos": np.asarray([0, 1, 2], dtype=np.int64),
        "y_pred": np.asarray([0, 1, 0], dtype=np.int64),
        "y_pred_proba": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]], dtype=np.float64),
        "y_logits": np.asarray([[3.0, 0.1], [0.1, 3.0], [2.0, 0.5]], dtype=np.float64),
        "y_embedding": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64),
    }

    out = meta_targets_from_predictions(pred_out=pred_out, store=store)

    np.testing.assert_array_equal(out, np.asarray([1, 1, 0], dtype=np.int64))


def test_kelly_bet_fraction_clips_negative_edge_and_supports_fractional_kelly() -> None:
    out = kelly_bet_fraction(
        np.asarray([0.40, 0.60, 0.80], dtype=np.float64),
        payoff_ratio=1.0,
        fraction=0.5,
    )

    np.testing.assert_allclose(out, np.asarray([0.0, 0.1, 0.3], dtype=np.float64))


def test_sized_trade_decisions_gate_on_threshold_and_kelly_edge() -> None:
    y_trade, bet_size, signed_bet_size = sized_trade_decisions(
        side_pred=np.asarray([0, 1, 1], dtype=np.int64),
        take_proba=np.asarray([0.80, 0.55, 0.49], dtype=np.float64),
        accept_threshold=0.5,
        payoff_ratio=1.0,
        fraction=1.0,
    )

    np.testing.assert_array_equal(y_trade, np.asarray([LABEL_DOWN, LABEL_UP, LABEL_EXIT], dtype=np.int64))
    np.testing.assert_allclose(bet_size, np.asarray([0.6, 0.1, 0.0], dtype=np.float64))
    np.testing.assert_allclose(signed_bet_size, np.asarray([-0.6, 0.1, 0.0], dtype=np.float64))


def test_evaluator_fits_meta_labeler_and_reports_combined_metrics() -> None:
    train_loader = DataLoader(
        _PredictionDataset(y_true=[0, 1, -1, 0], tids=[0, 0, 0, 0], tpos=[0, 1, 2, 3]),
        batch_size=4,
    )
    val_loader = DataLoader(
        _PredictionDataset(y_true=[1, -1], tids=[0, 0], tpos=[4, 5]),
        batch_size=2,
    )
    test_loader = DataLoader(
        _PredictionDataset(y_true=[0, 1, -1], tids=[0, 0, 0], tpos=[6, 7, 8]),
        batch_size=3,
    )
    model = _IndexLogitFeatureModel(
        logits=[
            [3.0, 0.1],
            [0.1, 3.0],
            [2.2, 0.2],
            [3.0, 0.1],
            [0.1, 3.0],
            [2.0, 0.1],
            [3.0, 0.1],
            [0.1, 3.0],
            [1.8, 0.2],
        ]
    )
    store = _FakeStore(
        tickers_all=["AAA"],
        labels=[0, 2, 1, 0, 2, 1, 0, 2, 1],
        pnl=[-0.05, 0.05, 0.0, -0.05, 0.05, -0.01, -0.05, 0.05, 0.0],
    )
    evaluator = ExperimentEvaluator(
        store=store,
        device=torch.device("cpu"),
        cfg=EvalConfig(
            meta_features=("logits",),
            meta_random_state=123,
            meta_accept_threshold=0.5,
            backtest_width_minutes=1439,
            backtest_barrier_height=0.05,
            labels=(0, 1),
        ),
    )

    metrics = evaluator.evaluate_all(model, {"train": train_loader, "val": val_loader, "test": test_loader}, step=1)

    assert metrics["test/cls/accuracy"] >= 0.0
    assert metrics["test/meta/f1"] >= 0.0
    assert metrics["test/meta/accept_threshold"] == 0.5
    assert metrics["test/decision/abstained_prediction_rate_pct"] >= 0.0
    assert metrics["test/execution/n_trade_signals_raw"] >= 0
    assert metrics["test/paper/n_executed_trades"] >= 0
