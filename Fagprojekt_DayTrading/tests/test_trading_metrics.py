import numpy as np
import pytest

from kvant.ml_framework.train.metrics import compute_paper_trading_metrics


def test_compute_paper_trading_metrics() -> None:
    """Paper-style trading metrics should match a simple hand-worked example."""
    y_true = np.asarray([2, 2, 0, 1], dtype=np.int64)
    y_pred = np.asarray([2, 2, 0, 1], dtype=np.int64)
    metas = [
        {
            "bar_open_time": "2024-01-01T00:00:00+00:00",
            "bar_close_time": "2024-01-01T12:00:00+00:00",
            "pnl_fraction": 0.10,
        },
        {
            "bar_open_time": "2024-01-02T00:00:00+00:00",
            "bar_close_time": "2024-01-02T12:00:00+00:00",
            "pnl_fraction": -0.20,
        },
        {
            "bar_open_time": "2024-01-03T00:00:00+00:00",
            "bar_close_time": "2024-01-03T12:00:00+00:00",
            "pnl_fraction": -0.05,
        },
        {
            "bar_open_time": "2024-01-04T00:00:00+00:00",
            "bar_close_time": "2024-01-04T12:00:00+00:00",
            "pnl_fraction": 0.30,
        },
    ]

    out = compute_paper_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metas=metas,
        initial_portfolio=1.0,
        transaction_cost=0.0,
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

    expected_final_portfolio = 1.10 * 0.80 * 1.05
    expected_annual_net_profit_loss_pct = (expected_final_portfolio - 1.0) * 100.0
    assert out["paper/annual_net_profit_loss_pct"] == pytest.approx(expected_annual_net_profit_loss_pct)

    assert out["paper/profitable_transactions_pct"] == pytest.approx((2.0 / 3.0) * 100.0)
    assert out["paper/max_drawdown_pct"] == pytest.approx(20.0)

    daily_returns = np.asarray([0.0, -0.2, 0.05, 0.0], dtype=np.float64)
    expected_sharpe = np.sqrt(4.0) * (daily_returns.mean() / daily_returns.std(ddof=0))
    assert out["paper/sharpe_ratio_annualized"] == pytest.approx(expected_sharpe)
