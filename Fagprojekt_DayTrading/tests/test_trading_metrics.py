import numpy as np
import pytest

from kvant.ml_framework.train.metrics import (
    BacktestTradeSimulator,
    apply_trade_confidence_threshold,
    compute_paper_trading_metrics,
    simulate_position_aware_trades,
)


class _FakeMarketDataStore:
    """Minimal market-data store for backtest unit tests."""

    def __init__(self, market_data: dict[int, dict[str, np.ndarray]]) -> None:
        self._market_data = market_data

    def market_data(self, tid: int) -> dict[str, np.ndarray] | None:
        return self._market_data.get(int(tid))

    def ticker(self, tid: int) -> str:
        return f"T{int(tid)}"


def test_compute_paper_trading_metrics() -> None:
    """Paper-style trading metrics should match a simple raw-bar backtest."""
    y_true = np.asarray([2, 2, 0, 1], dtype=np.int64)
    y_pred = np.asarray([2, 2, 0, 1], dtype=np.int64)
    tids = np.asarray([0, 0, 0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1, 2, 3], dtype=np.int64)
    simulator = BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(
            {
                0: {
                    "timestamp": np.asarray(
                        [
                            np.datetime64("2024-01-01T00:00:00"),
                            np.datetime64("2024-01-02T00:00:00"),
                            np.datetime64("2024-01-03T00:00:00"),
                            np.datetime64("2024-01-04T00:00:00"),
                        ]
                    ),
                    "open": np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
                    "high": np.asarray([106.0, 101.0, 104.0, 100.0], dtype=np.float32),
                    "low": np.asarray([99.0, 94.0, 94.0, 100.0], dtype=np.float32),
                    "close": np.asarray([105.0, 94.0, 95.0, 100.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                }
            }
        ),
        width_minutes=1439,
        barrier_height=0.05,
        transaction_cost=0.0,
    )

    out = compute_paper_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        tids=tids,
        tpos=tpos,
        simulator=simulator,
        initial_portfolio=1.0,
        risk_free_rate=0.0,
        days_per_year=4.0,
    )

    assert out["paper/n_executed_trades"] == 3
    assert out["paper/n_test_days"] == 4
    assert out["paper/tp"] == 2
    assert out["paper/tn"] == 1
    assert out["paper/fp"] == 0
    assert out["paper/fn"] == 0
    assert out["paper/accuracy_all_predictions"] == pytest.approx(1.0)

    expected_final_portfolio = 1.05 * 0.95 * 1.05
    expected_annual_net_profit_loss_pct = (expected_final_portfolio - 1.0) * 100.0
    assert out["paper/annual_net_profit_loss_pct"] == pytest.approx(expected_annual_net_profit_loss_pct)

    assert out["paper/profitable_transactions_pct"] == pytest.approx((2.0 / 3.0) * 100.0)
    assert out["paper/max_drawdown_pct"] == pytest.approx(5.0)

    daily_returns = np.asarray([0.0, -0.05, 0.05, 0.0], dtype=np.float64)
    expected_sharpe = np.sqrt(4.0) * (daily_returns.mean() / daily_returns.std(ddof=0))
    assert out["paper/sharpe_ratio_annualized"] == pytest.approx(expected_sharpe)


def test_position_aware_backtest_skips_overlapping_signals() -> None:
    """Overlapping acted signals should collapse to non-overlapping executed trades."""
    y_pred = np.asarray([2, 2, 0], dtype=np.int64)
    metas = [
        {
            "bar_open_time": "2024-01-01T00:00:00+00:00",
            "bar_close_time": "2024-01-03T00:00:00+00:00",
            "pnl_fraction": 0.10,
        },
        {
            "bar_open_time": "2024-01-02T00:00:00+00:00",
            "bar_close_time": "2024-01-02T12:00:00+00:00",
            "pnl_fraction": 0.05,
        },
        {
            "bar_open_time": "2024-01-03T00:00:00+00:00",
            "bar_close_time": "2024-01-04T00:00:00+00:00",
            "pnl_fraction": -0.05,
        },
    ]
    tids = np.asarray([0, 0, 0], dtype=np.int64)

    executed = simulate_position_aware_trades(y_pred=y_pred, metas=metas, tids=tids, transaction_cost=0.0)

    assert len(executed) == 2
    assert executed[0].signal_label == 2
    assert executed[1].signal_label == 0
    assert executed[0].gross_return == pytest.approx(0.10)
    assert executed[1].gross_return == pytest.approx(0.05)


def test_position_aware_backtest_allows_overlaps_across_tickers() -> None:
    """Overlapping signals on different tickers should both execute."""
    y_pred = np.asarray([2, 0], dtype=np.int64)
    metas = [
        {
            "bar_open_time": "2024-01-01T00:00:00+00:00",
            "bar_close_time": "2024-01-03T00:00:00+00:00",
            "pnl_fraction": 0.10,
        },
        {
            "bar_open_time": "2024-01-02T00:00:00+00:00",
            "bar_close_time": "2024-01-02T12:00:00+00:00",
            "pnl_fraction": -0.05,
        },
    ]
    tids = np.asarray([0, 1], dtype=np.int64)

    executed = simulate_position_aware_trades(y_pred=y_pred, metas=metas, tids=tids, transaction_cost=0.0)

    assert len(executed) == 2
    assert executed[0].tid == 0
    assert executed[1].tid == 1
    assert executed[0].gross_return == pytest.approx(0.10)
    assert executed[1].gross_return == pytest.approx(0.05)


def test_apply_trade_confidence_threshold_abstains_on_low_confidence_trade_signals() -> None:
    """Low-confidence acted predictions should be converted into exit/hold signals."""
    y_pred = np.asarray([2, 0, 1, 2], dtype=np.int64)
    y_pred_confidence = np.asarray([0.95, 0.59, 0.20, 0.60], dtype=np.float64)

    out = apply_trade_confidence_threshold(
        y_pred=y_pred,
        y_pred_confidence=y_pred_confidence,
        trade_confidence_threshold=0.60,
    )

    np.testing.assert_array_equal(out, np.asarray([2, 1, 1, 2], dtype=np.int64))
