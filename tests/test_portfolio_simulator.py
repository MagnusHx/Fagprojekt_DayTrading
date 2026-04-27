import numpy as np
import pytest

from kvant.labels import LABEL_UP
from kvant.ml_framework.train.backtest import BacktestTradeSimulator
from kvant.ml_framework.train.portfolio_simulator import compute_portfolio_metrics


class _FakeMarketDataStore:
    def __init__(self, market_data: dict[int, dict[str, np.ndarray]]) -> None:
        self._market_data = market_data

    def market_data(self, tid: int) -> dict[str, np.ndarray] | None:
        return self._market_data.get(int(tid))

    def ticker(self, tid: int) -> str:
        return f"T{int(tid)}"


def _market(close: list[float]) -> dict[str, np.ndarray]:
    close_arr = np.asarray(close, dtype=np.float32)
    return {
        "timestamp": np.asarray(
            [
                np.datetime64("2024-01-01T00:00:00"),
                np.datetime64("2024-01-02T00:00:00"),
                np.datetime64("2024-01-03T00:00:00"),
            ]
        ),
        "open": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
        "high": np.maximum(close_arr, 100.0).astype(np.float32),
        "low": np.minimum(close_arr, 100.0).astype(np.float32),
        "close": close_arr,
        "volume": np.ones(3, dtype=np.float32),
    }


def _simulator(market_data: dict[int, dict[str, np.ndarray]], *, transaction_cost: float = 0.0) -> BacktestTradeSimulator:
    return BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(market_data),
        width_minutes=1440,
        barrier_height=0.50,
        transaction_cost=transaction_cost,
    )


def test_portfolio_balance_rises_for_profitable_long() -> None:
    result = compute_portfolio_metrics(
        y_true=np.asarray([LABEL_UP]),
        y_pred=np.asarray([LABEL_UP]),
        tids=np.asarray([0]),
        tpos=np.asarray([0]),
        simulator=_simulator({0: _market([100.0, 100.0, 110.0])}),
        initial_cash=10_000.0,
        max_position_fraction=0.10,
        max_total_exposure=1.0,
        max_positions=10,
        risk_free_rate=0.0,
    )

    assert result.metrics["portfolio/final_balance"] == pytest.approx(10_100.0)
    assert result.metrics["portfolio/total_return_pct"] == pytest.approx(1.0)
    assert result.metrics["portfolio/n_executed_trades"] == 1


def test_portfolio_balance_rises_for_profitable_short() -> None:
    result = compute_portfolio_metrics(
        y_true=np.asarray([0]),
        y_pred=np.asarray([0]),
        tids=np.asarray([0]),
        tpos=np.asarray([0]),
        simulator=_simulator({0: _market([100.0, 100.0, 90.0])}),
        initial_cash=10_000.0,
        max_position_fraction=0.10,
        max_total_exposure=1.0,
        max_positions=10,
        risk_free_rate=0.0,
    )

    assert result.metrics["portfolio/final_balance"] == pytest.approx(10_100.0)
    assert result.metrics["portfolio/total_return_pct"] == pytest.approx(1.0)
    assert result.metrics["portfolio/n_executed_trades"] == 1


def test_portfolio_applies_entry_and_exit_transaction_costs() -> None:
    result = compute_portfolio_metrics(
        y_true=np.asarray([LABEL_UP]),
        y_pred=np.asarray([LABEL_UP]),
        tids=np.asarray([0]),
        tpos=np.asarray([0]),
        simulator=_simulator({0: _market([100.0, 100.0, 100.0])}, transaction_cost=0.01),
        initial_cash=10_000.0,
        max_position_fraction=0.10,
        max_total_exposure=1.0,
        max_positions=10,
        risk_free_rate=0.0,
    )

    assert result.metrics["portfolio/final_balance"] == pytest.approx(9_980.0)
    assert result.metrics["portfolio/transaction_cost_total"] == pytest.approx(20.0)
    assert result.metrics["portfolio/total_return_pct"] == pytest.approx(-0.2)


def test_portfolio_skips_trade_when_exposure_budget_is_exhausted() -> None:
    result = compute_portfolio_metrics(
        y_true=np.asarray([LABEL_UP, LABEL_UP]),
        y_pred=np.asarray([LABEL_UP, LABEL_UP]),
        tids=np.asarray([0, 1]),
        tpos=np.asarray([0, 0]),
        simulator=_simulator({0: _market([100.0, 100.0, 110.0]), 1: _market([100.0, 100.0, 110.0])}),
        initial_cash=10_000.0,
        max_position_fraction=0.10,
        max_total_exposure=0.10,
        max_positions=10,
        risk_free_rate=0.0,
    )

    assert result.metrics["portfolio/n_candidate_trades"] == 2
    assert result.metrics["portfolio/n_executed_trades"] == 1
    assert result.metrics["portfolio/n_skipped_budget"] == 1
    assert result.metrics["portfolio/max_concurrent_positions"] == 1
