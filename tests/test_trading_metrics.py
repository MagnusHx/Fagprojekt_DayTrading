import numpy as np
import pytest

from kvant.ml_framework.train.metrics import (
    BacktestTradeSimulator,
    apply_trade_decision_bands,
    apply_trade_decision_thresholds,
    compute_paper_trading_metrics,
    simulate_position_aware_trades,
    trade_decision_components,
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
                    "high": np.asarray([100.0, 106.0, 104.0, 100.0], dtype=np.float32),
                    "low": np.asarray([100.0, 99.0, 94.0, 94.0], dtype=np.float32),
                    "close": np.asarray([100.0, 105.0, 95.0, 95.0], dtype=np.float32),
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
    assert out["paper/acted_prediction_accuracy"] == pytest.approx(1.0)
    assert out["paper/directional_acted_accuracy"] == pytest.approx(1.0)
    assert out["paper/abstained_prediction_rate_pct"] == pytest.approx(25.0)
    assert out["paper/n_trade_signals_raw"] == 3
    assert out["paper/n_trade_signals_skipped_overlap"] == 0
    assert out["paper/executed_trade_hit_rate_pct"] == pytest.approx(100.0)
    assert out["paper/executed_trade_gross_return_avg_pct"] == pytest.approx((5.0 - 5.0 + 5.0) / 3.0)
    assert out["paper/executed_trade_net_return_avg_pct"] == pytest.approx((5.0 - 5.0 + 5.0) / 3.0)
    assert out["paper/long_n_executed_trades"] == 2
    assert out["paper/short_n_executed_trades"] == 1
    assert out["paper/long_hit_rate_pct"] == pytest.approx(100.0)
    assert out["paper/short_hit_rate_pct"] == pytest.approx(100.0)

    expected_final_portfolio = 1.05 * 0.95 * 1.05
    expected_annual_net_profit_loss_pct = (expected_final_portfolio - 1.0) * 100.0
    assert out["paper/annual_net_profit_loss_pct"] == pytest.approx(expected_annual_net_profit_loss_pct)

    assert out["paper/profitable_transactions_pct"] == pytest.approx((2.0 / 3.0) * 100.0)
    assert out["paper/max_drawdown_pct"] == pytest.approx(5.0)

    daily_returns = np.asarray([0.0, 0.05, -0.05, 0.05], dtype=np.float64)
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


def test_position_aware_backtest_scales_returns_by_bet_size() -> None:
    """Executed returns should scale with the requested Kelly position size."""
    y_pred = np.asarray([2, 0, 2], dtype=np.int64)
    metas = [
        {
            "bar_open_time": "2024-01-01T00:00:00+00:00",
            "bar_close_time": "2024-01-01T12:00:00+00:00",
            "pnl_fraction": 0.10,
        },
        {
            "bar_open_time": "2024-01-02T00:00:00+00:00",
            "bar_close_time": "2024-01-02T12:00:00+00:00",
            "pnl_fraction": -0.05,
        },
        {
            "bar_open_time": "2024-01-03T00:00:00+00:00",
            "bar_close_time": "2024-01-03T12:00:00+00:00",
            "pnl_fraction": 0.03,
        },
    ]
    tids = np.asarray([0, 0, 0], dtype=np.int64)
    bet_sizes = np.asarray([0.5, 0.25, 0.0], dtype=np.float64)

    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        bet_sizes=bet_sizes,
        transaction_cost=0.0,
    )

    assert len(executed) == 2
    assert executed[0].bet_size == pytest.approx(0.5)
    assert executed[1].bet_size == pytest.approx(0.25)
    assert executed[0].gross_return == pytest.approx(0.05)
    assert executed[1].gross_return == pytest.approx(0.0125)


def test_trade_decision_components_split_action_and_direction_confidence() -> None:
    """Action probability and directional confidence should be derived from up/down mass."""
    y_pred_proba = np.asarray(
        [
            [0.20, 0.30, 0.50],
            [0.45, 0.40, 0.15],
            [0.00, 1.00, 0.00],
        ],
        dtype=np.float64,
    )

    p_act, q_up = trade_decision_components(y_pred_proba=y_pred_proba)

    np.testing.assert_allclose(p_act, np.asarray([0.70, 0.60, 0.00], dtype=np.float64))
    np.testing.assert_allclose(q_up, np.asarray([0.50 / 0.70, 0.15 / 0.60, 0.50], dtype=np.float64))


def test_apply_trade_decision_thresholds_abstains_when_action_or_direction_is_weak() -> None:
    """Trading should require both enough action probability and directional conviction."""
    y_pred_proba = np.asarray(
        [
            [0.10, 0.10, 0.80],  # strong up
            [0.45, 0.30, 0.25],  # strong down
            [0.20, 0.50, 0.30],  # not actionable enough
            [0.35, 0.15, 0.50],  # actionable but direction too ambiguous
            [0.05, 0.80, 0.15],  # clearly exit
        ],
        dtype=np.float64,
    )

    out = apply_trade_decision_thresholds(
        y_pred_proba=y_pred_proba,
        trade_action_threshold=0.60,
        trade_direction_threshold=0.60,
    )

    np.testing.assert_array_equal(out, np.asarray([2, 0, 1, 1, 1], dtype=np.int64))


def test_apply_trade_decision_bands_matches_threshold_rule_components() -> None:
    y_pred_proba = np.asarray(
        [
            [0.10, 0.10, 0.80],
            [0.45, 0.30, 0.25],
            [0.20, 0.50, 0.30],
        ],
        dtype=np.float64,
    )

    p_act, q_up = trade_decision_components(y_pred_proba=y_pred_proba)
    direct = apply_trade_decision_bands(
        p_act=p_act,
        q_up=q_up,
        trade_action_threshold=0.60,
        trade_direction_threshold=0.60,
    )
    threshold = apply_trade_decision_thresholds(
        y_pred_proba=y_pred_proba,
        trade_action_threshold=0.60,
        trade_direction_threshold=0.60,
    )

    np.testing.assert_array_equal(direct, threshold)


def test_binary_trade_decision_components_use_probability_band_only() -> None:
    """Binary directional probabilities should map directly to q_up with always-actionable mass."""
    y_pred_proba = np.asarray(
        [
            [0.25, 0.75],
            [0.70, 0.30],
            [0.50, 0.50],
        ],
        dtype=np.float64,
    )

    p_act, q_up = trade_decision_components(y_pred_proba=y_pred_proba)

    np.testing.assert_allclose(p_act, np.asarray([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(q_up, np.asarray([0.75, 0.30, 0.50], dtype=np.float64))


def test_binary_apply_trade_decision_thresholds_uses_symmetric_abstention_band() -> None:
    """Binary directional runs should long, short, or abstain from q_up directly."""
    y_pred_proba = np.asarray(
        [
            [0.25, 0.75],  # long
            [0.80, 0.20],  # short
            [0.45, 0.55],  # abstain
        ],
        dtype=np.float64,
    )

    out = apply_trade_decision_thresholds(
        y_pred_proba=y_pred_proba,
        trade_action_threshold=0.60,
        trade_direction_threshold=0.60,
    )

    np.testing.assert_array_equal(out, np.asarray([2, 0, 1], dtype=np.int64))


def test_compute_paper_trading_metrics_exposes_abstention_debug_metrics_for_directional_truth() -> None:
    """Binary-directional truth with abstentions should produce meaningful debug metrics."""
    y_true = np.asarray([2, 0, 2], dtype=np.int64)
    y_pred = np.asarray([1, 0, 2], dtype=np.int64)
    tids = np.asarray([0, 0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1, 2], dtype=np.int64)
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
                    "high": np.asarray([100.0, 100.0, 100.0, 106.0], dtype=np.float32),
                    "low": np.asarray([100.0, 100.0, 94.0, 100.0], dtype=np.float32),
                    "close": np.asarray([100.0, 100.0, 95.0, 105.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                }
            }
        ),
        width_minutes=1439,
        barrier_height=0.05,
        transaction_cost=0.001,
    )

    out = compute_paper_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        tids=tids,
        tpos=tpos,
        simulator=simulator,
        initial_portfolio=1.0,
        risk_free_rate=0.0,
        days_per_year=3.0,
    )

    assert out["paper/accuracy_all_predictions"] == pytest.approx(2.0 / 3.0)
    assert out["paper/acted_prediction_accuracy"] == pytest.approx(1.0)
    assert out["paper/directional_acted_accuracy"] == pytest.approx(1.0)
    assert out["paper/actionable_truth_rate_pct"] == pytest.approx(100.0)
    assert out["paper/abstained_prediction_rate_pct"] == pytest.approx((1.0 / 3.0) * 100.0)
    assert out["paper/abstain_on_actionable_truth_pct"] == pytest.approx((1.0 / 3.0) * 100.0)
    assert out["paper/acted_on_exit_truth_pct"] == pytest.approx(0.0)
    assert out["paper/n_trade_signals_raw"] == 2
    assert out["paper/n_executed_trades"] == 2
    assert out["paper/n_trade_signals_skipped_overlap"] == 0
    assert out["paper/executed_trade_hit_rate_pct"] == pytest.approx(100.0)
    assert out["paper/executed_trade_gross_return_avg_pct"] == pytest.approx(5.0)
    assert out["paper/executed_trade_net_return_avg_pct"] == pytest.approx(4.8)
    assert out["paper/transaction_cost_total_pct"] == pytest.approx(0.4)
    assert out["paper/long_n_executed_trades"] == 1
    assert out["paper/short_n_executed_trades"] == 1


def test_compute_paper_trading_metrics_scales_portfolio_by_bet_size() -> None:
    """Fractional Kelly sizing should scale both trade PnL and compounded portfolio value."""
    y_true = np.asarray([2, 0, 2], dtype=np.int64)
    y_pred = np.asarray([2, 0, 2], dtype=np.int64)
    tids = np.asarray([0, 0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1, 2], dtype=np.int64)
    bet_sizes = np.asarray([0.5, 0.25, 0.0], dtype=np.float64)
    simulator = BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(
            {
                0: {
                    "timestamp": np.asarray(
                        [
                            np.datetime64("2024-01-01T00:00:00"),
                            np.datetime64("2024-01-02T00:00:00"),
                            np.datetime64("2024-01-03T00:00:00"),
                        ]
                    ),
                    "open": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
                    "high": np.asarray([100.0, 106.0, 100.0], dtype=np.float32),
                    "low": np.asarray([100.0, 100.0, 94.0], dtype=np.float32),
                    "close": np.asarray([100.0, 105.0, 95.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
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
        bet_sizes=bet_sizes,
        simulator=simulator,
        initial_portfolio=1.0,
        risk_free_rate=0.0,
        days_per_year=3.0,
    )

    expected_final_portfolio = 1.025 * 1.0125
    assert out["paper/n_trade_signals_raw"] == 2
    assert out["paper/n_executed_trades"] == 2
    assert out["paper/executed_trade_gross_return_avg_pct"] == pytest.approx(1.875)
    assert out["paper/executed_trade_gross_return_total_pct"] == pytest.approx(3.75)
    assert out["paper/annual_net_profit_loss_pct"] == pytest.approx((expected_final_portfolio - 1.0) * 100.0)


def test_compute_paper_trading_metrics_handles_zero_executed_trades() -> None:
    """No acted predictions should produce zeroed execution and economic metrics."""
    y_true = np.asarray([1, 1], dtype=np.int64)
    y_pred = np.asarray([1, 1], dtype=np.int64)
    tids = np.asarray([0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1], dtype=np.int64)
    simulator = BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(
            {
                0: {
                    "timestamp": np.asarray([np.datetime64("2024-01-01T00:00:00"), np.datetime64("2024-01-02T00:00:00")]),
                    "open": np.asarray([100.0, 100.0], dtype=np.float32),
                    "high": np.asarray([100.0, 100.0], dtype=np.float32),
                    "low": np.asarray([100.0, 100.0], dtype=np.float32),
                    "close": np.asarray([100.0, 100.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0], dtype=np.float32),
                }
            }
        ),
        width_minutes=1439,
        barrier_height=0.05,
        transaction_cost=0.001,
    )

    out = compute_paper_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        tids=tids,
        tpos=tpos,
        simulator=simulator,
    )

    assert out["paper/n_trade_signals_raw"] == 0
    assert out["paper/n_trade_signals_skipped_overlap"] == 0
    assert out["paper/n_executed_trades"] == 0
    assert out["paper/executed_trade_hit_rate_pct"] == pytest.approx(0.0)
    assert out["paper/executed_trade_gross_return_avg_pct"] == pytest.approx(0.0)
    assert out["paper/executed_trade_net_return_avg_pct"] == pytest.approx(0.0)
    assert out["paper/transaction_cost_total_pct"] == pytest.approx(0.0)
    assert out["paper/share_time_active_pct"] == pytest.approx(0.0)


def test_compute_paper_trading_metrics_exposes_cost_drag_when_gross_edge_is_positive() -> None:
    """Positive gross edge should become negative net edge when transaction costs are large enough."""
    y_true = np.asarray([2, 2], dtype=np.int64)
    y_pred = np.asarray([2, 2], dtype=np.int64)
    tids = np.asarray([0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1], dtype=np.int64)
    simulator = BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(
            {
                0: {
                    "timestamp": np.asarray(
                        [
                            np.datetime64("2024-01-01T00:00:00"),
                            np.datetime64("2024-01-02T00:00:00"),
                            np.datetime64("2024-01-03T00:00:00"),
                        ]
                    ),
                    "open": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
                    "high": np.asarray([100.0, 103.0, 103.0], dtype=np.float32),
                    "low": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
                    "close": np.asarray([100.0, 103.0, 103.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                }
            }
        ),
        width_minutes=1439,
        barrier_height=0.03,
        transaction_cost=0.02,
    )

    out = compute_paper_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        tids=tids,
        tpos=tpos,
        simulator=simulator,
        initial_portfolio=1.0,
        risk_free_rate=0.0,
        days_per_year=2.0,
    )

    assert out["paper/executed_trade_gross_return_avg_pct"] == pytest.approx(3.0)
    assert out["paper/executed_trade_net_return_avg_pct"] == pytest.approx(-1.0)
    assert out["paper/transaction_cost_total_pct"] == pytest.approx(8.0)


def test_compute_paper_trading_metrics_counts_acted_on_exit_truth() -> None:
    """Three-class runs should expose when the strategy trades rows whose truth is an exit."""
    y_true = np.asarray([1, 2], dtype=np.int64)
    y_pred = np.asarray([2, 2], dtype=np.int64)
    tids = np.asarray([0, 0], dtype=np.int64)
    tpos = np.asarray([0, 1], dtype=np.int64)
    simulator = BacktestTradeSimulator(
        market_data_store=_FakeMarketDataStore(
            {
                0: {
                    "timestamp": np.asarray(
                        [
                            np.datetime64("2024-01-01T00:00:00"),
                            np.datetime64("2024-01-02T00:00:00"),
                            np.datetime64("2024-01-03T00:00:00"),
                        ]
                    ),
                    "open": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
                    "high": np.asarray([100.0, 105.0, 105.0], dtype=np.float32),
                    "low": np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
                    "close": np.asarray([100.0, 105.0, 105.0], dtype=np.float32),
                    "volume": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
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
    )

    assert out["paper/acted_on_exit_truth_pct"] == pytest.approx(100.0)
    assert out["paper/abstain_on_actionable_truth_pct"] == pytest.approx(0.0)
