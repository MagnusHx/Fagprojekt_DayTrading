from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from kvant.labels import ACTED_LABELS, LABEL_DOWN, LABEL_UP


@dataclass(frozen=True)
class BacktestTrade:
    """Executed trade produced by the backtest simulator."""

    tid: int
    ticker: str
    signal_label: int
    true_label: Optional[int]
    entry_tpos: int
    exit_tpos: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    exit_reason: str


class MarketDataStore(Protocol):
    """Protocol for reading sampled raw market data during backtests."""

    def market_data(self, tid: int) -> Optional[Dict[str, np.ndarray]]:
        """Return sampled raw market data arrays for a ticker."""

    def ticker(self, tid: int) -> str:
        """Return the ticker name for a numeric ticker id."""


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Convert a timestamp-like value into a UTC pandas timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _total_active_duration(trades: List[BacktestTrade]) -> pd.Timedelta:
    """Return the union of active trade intervals across all executed trades."""
    if not trades:
        return pd.Timedelta(0)

    intervals = sorted((trade.entry_time, trade.exit_time) for trade in trades)
    merged_start, merged_end = intervals[0]
    total = pd.Timedelta(0)

    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        total += merged_end - merged_start
        merged_start, merged_end = start, end

    total += merged_end - merged_start
    return total


@dataclass(frozen=True)
class BacktestTradeSimulator:
    """Simulate barrier-based trades on sampled raw OHLCV bars."""

    market_data_store: MarketDataStore
    width_minutes: int
    barrier_height: float
    transaction_cost: float = 0.0

    def _candidate_trade(
        self,
        *,
        signal_label: int,
        tid: int,
        tpos: int,
        true_label: Optional[int],
    ) -> Optional[BacktestTrade]:
        if signal_label not in ACTED_LABELS:
            return None

        market_data = self.market_data_store.market_data(tid)
        if market_data is None:
            ticker = self.market_data_store.ticker(tid)
            raise RuntimeError(
                f"Prepared market data is missing for ticker {ticker}. "
                "Regenerate the prepared experiment so backtests can use raw sampled OHLCV bars."
            )

        timestamps = market_data["timestamp"]
        opens = np.asarray(market_data["open"], dtype=np.float64)
        highs = np.asarray(market_data["high"], dtype=np.float64)
        lows = np.asarray(market_data["low"], dtype=np.float64)
        closes = np.asarray(market_data["close"], dtype=np.float64)

        entry_pos = int(tpos)
        if entry_pos < 0 or entry_pos >= len(timestamps):
            return None

        entry_time = _to_utc_timestamp(timestamps[entry_pos])
        entry_price = float(opens[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            return None

        end_target = entry_time + pd.Timedelta(minutes=int(self.width_minutes))
        timestamp_index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        end_pos = int(timestamp_index.searchsorted(end_target, side="right") - 1)
        if end_pos < entry_pos:
            return None

        upper = entry_price * (1.0 + float(self.barrier_height))
        lower = entry_price * (1.0 - float(self.barrier_height))

        exit_pos = end_pos
        exit_price = float(closes[end_pos])
        exit_reason = "time_exit"

        for bar_pos in range(entry_pos, end_pos + 1):
            hit_up = bool(highs[bar_pos] >= upper)
            hit_down = bool(lows[bar_pos] <= lower)
            if not hit_up and not hit_down:
                continue

            exit_pos = bar_pos
            if signal_label == LABEL_UP:
                if hit_down:
                    exit_price = lower
                    exit_reason = "stop_loss"
                else:
                    exit_price = upper
                    exit_reason = "take_profit"
            else:
                if hit_up:
                    exit_price = upper
                    exit_reason = "stop_loss"
                else:
                    exit_price = lower
                    exit_reason = "take_profit"
            break

        exit_time = _to_utc_timestamp(timestamps[exit_pos])
        if signal_label == LABEL_UP:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
        net_return = gross_return - (2.0 * float(self.transaction_cost))

        return BacktestTrade(
            tid=int(tid),
            ticker=self.market_data_store.ticker(tid),
            signal_label=int(signal_label),
            true_label=true_label,
            entry_tpos=entry_pos,
            exit_tpos=exit_pos,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            gross_return=float(gross_return),
            net_return=float(net_return),
            exit_reason=exit_reason,
        )

    def simulate(
        self,
        *,
        y_pred: np.ndarray,
        tids: np.ndarray,
        tpos: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> List[BacktestTrade]:
        """Simulate non-overlapping per-ticker trades on sampled raw bars."""
        if not (len(y_pred) == len(tids) == len(tpos)):
            raise ValueError(
                "BacktestTradeSimulator.simulate expected y_pred, tids, and tpos to have the same length, "
                f"got {len(y_pred)}, {len(tids)}, and {len(tpos)}."
            )
        if y_true is not None:
            if len(y_true) != len(y_pred):
                raise ValueError(
                    "BacktestTradeSimulator.simulate expected y_true to match y_pred length, "
                    f"got {len(y_true)} and {len(y_pred)}."
                )

        candidates: List[BacktestTrade] = []
        for idx, signal_label in enumerate(np.asarray(y_pred, dtype=np.int64)):
            true_label = None if y_true is None else int(y_true[idx])
            trade = self._candidate_trade(
                signal_label=int(signal_label),
                tid=int(tids[idx]),
                tpos=int(tpos[idx]),
                true_label=true_label,
            )
            if trade is not None:
                candidates.append(trade)

        candidates.sort(key=lambda trade: (trade.entry_time, trade.exit_time, trade.tid))

        executed: List[BacktestTrade] = []
        active_until_by_tid: Dict[int, pd.Timestamp] = {}
        for trade in candidates:
            active_until = active_until_by_tid.get(trade.tid)
            if active_until is not None and trade.entry_time < active_until:
                continue
            executed.append(trade)
            active_until_by_tid[trade.tid] = trade.exit_time

        return executed


def compute_paper_trading_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tids: np.ndarray,
    tpos: np.ndarray,
    simulator: BacktestTradeSimulator,
    initial_portfolio: float = 1.0,
    risk_free_rate: float = 0.0314,
    days_per_year: float = 365.0,
) -> Dict[str, Any]:
    """Compute paper-trading metrics from simulated executed trades."""
    if not (len(y_true) == len(y_pred) == len(tids) == len(tpos)):
        raise ValueError(
            "compute_paper_trading_metrics expected y_true, y_pred, tids, and tpos to have the same length, "
            f"got {len(y_true)}, {len(y_pred)}, {len(tids)}, and {len(tpos)}."
        )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tids = np.asarray(tids, dtype=np.int64)
    tpos = np.asarray(tpos, dtype=np.int64)

    tp = int(np.sum((y_true == LABEL_UP) & (y_pred == LABEL_UP)))
    tn = int(np.sum((y_true == LABEL_DOWN) & (y_pred == LABEL_DOWN)))
    fp = int(np.sum((y_true == LABEL_DOWN) & (y_pred == LABEL_UP)))
    fn = int(np.sum((y_true == LABEL_UP) & (y_pred == LABEL_DOWN)))
    # Legacy compatibility metric only. On directional-binary runs, abstentions are
    # encoded as EXIT in y_pred while truth is still directional in canonical trade
    # space, so this can look artificially poor. Prefer acted/directional_acted metrics.
    accuracy_all_predictions = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    acted_mask = np.isin(y_pred, ACTED_LABELS)
    actionable_truth_mask = np.isin(y_true, ACTED_LABELS)
    directional_acted_mask = acted_mask & actionable_truth_mask
    abstained_mask = y_pred == 1

    acted_prediction_accuracy = (
        float(np.mean(y_true[acted_mask] == y_pred[acted_mask])) if np.any(acted_mask) else 0.0
    )
    directional_acted_accuracy = (
        float(np.mean(y_true[directional_acted_mask] == y_pred[directional_acted_mask]))
        if np.any(directional_acted_mask)
        else 0.0
    )
    abstain_on_actionable_truth_pct = (
        float(np.mean(abstained_mask[actionable_truth_mask]) * 100.0) if np.any(actionable_truth_mask) else 0.0
    )
    acted_on_exit_truth_pct = (
        float(np.mean(acted_mask[y_true == 1]) * 100.0) if np.any(y_true == 1) else 0.0
    )

    executed = simulator.simulate(y_pred=y_pred, tids=tids, tpos=tpos, y_true=y_true)
    gross_returns = np.asarray([trade.gross_return for trade in executed], dtype=np.float64)
    net_returns = np.asarray([trade.net_return for trade in executed], dtype=np.float64)
    executed_trade_hit_flags = np.asarray(
        [
            int(trade.true_label) == int(trade.signal_label)
            for trade in executed
            if trade.true_label in ACTED_LABELS
        ],
        dtype=np.float64,
    )
    long_net_returns = np.asarray([trade.net_return for trade in executed if trade.signal_label == LABEL_UP], dtype=np.float64)
    short_net_returns = np.asarray(
        [trade.net_return for trade in executed if trade.signal_label == LABEL_DOWN], dtype=np.float64
    )
    long_hit_flags = np.asarray(
        [int(trade.true_label) == LABEL_UP for trade in executed if trade.signal_label == LABEL_UP and trade.true_label in ACTED_LABELS],
        dtype=np.float64,
    )
    short_hit_flags = np.asarray(
        [int(trade.true_label) == LABEL_DOWN for trade in executed if trade.signal_label == LABEL_DOWN and trade.true_label in ACTED_LABELS],
        dtype=np.float64,
    )
    n_trade_signals_raw = int(np.sum(acted_mask))
    n_trade_signals_skipped_overlap = int(max(n_trade_signals_raw - len(executed), 0))
    transaction_cost_total_pct = float(len(executed) * 2.0 * float(simulator.transaction_cost) * 100.0)

    sample_times = []
    for tid, pos in zip(tids, tpos):
        market_data = simulator.market_data_store.market_data(int(tid))
        if market_data is None:
            ticker = simulator.market_data_store.ticker(int(tid))
            raise RuntimeError(
                f"Prepared market data is missing for ticker {ticker}. "
                "Regenerate the prepared experiment so backtests can use raw sampled OHLCV bars."
            )
        sample_times.append(_to_utc_timestamp(market_data["timestamp"][int(pos)]))
    if sample_times:
        period_start = min(sample_times).normalize()
        period_end = max(sample_times).normalize()
        if executed:
            period_end = max(period_end, max(trade.exit_time for trade in executed).normalize())
    elif executed:
        period_start = executed[0].entry_time.normalize()
        period_end = executed[-1].exit_time.normalize()
    else:
        now = pd.Timestamp.now(tz="UTC").normalize()
        period_start = now
        period_end = now

    daily_index = pd.date_range(period_start, period_end, freq="D", tz="UTC")
    if len(daily_index) == 0:
        daily_index = pd.DatetimeIndex([period_start])

    portfolio_value = float(initial_portfolio)
    trade_records: List[dict[str, Any]] = []
    for trade in executed:
        portfolio_value *= max(0.0, 1.0 + float(trade.net_return))
        trade_records.append(
            {
                "exit_time": trade.exit_time,
                "portfolio_value": portfolio_value,
                "net_return": float(trade.net_return),
            }
        )

    if trade_records:
        trade_df = pd.DataFrame.from_records(trade_records)
        trade_df["date"] = pd.to_datetime(trade_df["exit_time"], utc=True).dt.normalize()
        daily_portfolio = trade_df.groupby("date")["portfolio_value"].last().reindex(daily_index).ffill()
        daily_portfolio = daily_portfolio.fillna(float(initial_portfolio))
    else:
        daily_portfolio = pd.Series(float(initial_portfolio), index=daily_index, dtype=np.float64)

    daily_returns = daily_portfolio.pct_change().fillna(0.0)
    final_portfolio = float(daily_portfolio.iloc[-1])
    n_days = max(int(len(daily_portfolio)), 1)

    annual_net_profit_loss_pct = (
        ((final_portfolio / float(initial_portfolio)) ** (float(days_per_year) / float(n_days)) - 1.0) * 100.0
        if initial_portfolio > 0.0
        else 0.0
    )

    profitable_transactions_pct = (
        float(np.mean([trade.net_return > 0.0 for trade in executed]) * 100.0) if executed else 0.0
    )
    risk_free_daily = (1.0 + float(risk_free_rate)) ** (1.0 / float(days_per_year)) - 1.0
    daily_std = float(daily_returns.std(ddof=0))
    sharpe_ratio_annualized = (
        float(np.sqrt(float(days_per_year)) * ((daily_returns.mean() - risk_free_daily) / daily_std))
        if daily_std > 0.0
        else 0.0
    )

    running_peak = daily_portfolio.cummax()
    max_drawdown_pct = float(((running_peak - daily_portfolio) / running_peak.clip(lower=1e-12)).max() * 100.0)

    total_duration = max((period_end - period_start).total_seconds(), 1.0)
    active_duration = _total_active_duration(executed)
    share_time_active_pct = float(active_duration.total_seconds() / total_duration * 100.0) if executed else 0.0

    return {
        "paper/annual_net_profit_loss_pct": float(annual_net_profit_loss_pct),
        "paper/profitable_transactions_pct": float(profitable_transactions_pct),
        "paper/accuracy_all_predictions": float(accuracy_all_predictions),
        "paper/acted_prediction_accuracy": float(acted_prediction_accuracy),
        "paper/directional_acted_accuracy": float(directional_acted_accuracy),
        "paper/actionable_truth_rate_pct": float(np.mean(actionable_truth_mask) * 100.0) if len(y_true) else 0.0,
        "paper/abstained_prediction_rate_pct": float(np.mean(abstained_mask) * 100.0) if len(y_pred) else 0.0,
        "paper/abstain_on_actionable_truth_pct": float(abstain_on_actionable_truth_pct),
        "paper/acted_on_exit_truth_pct": float(acted_on_exit_truth_pct),
        "paper/sharpe_ratio_annualized": float(sharpe_ratio_annualized),
        "paper/max_drawdown_pct": float(max_drawdown_pct),
        "paper/share_time_active_pct": float(share_time_active_pct),
        "paper/n_trade_signals_raw": int(n_trade_signals_raw),
        "paper/n_trade_signals_skipped_overlap": int(n_trade_signals_skipped_overlap),
        "paper/n_executed_trades": int(len(executed)),
        "paper/executed_trade_hit_rate_pct": float(np.mean(executed_trade_hit_flags) * 100.0)
        if len(executed_trade_hit_flags)
        else 0.0,
        "paper/executed_trade_gross_return_avg_pct": float(np.mean(gross_returns) * 100.0) if len(gross_returns) else 0.0,
        "paper/executed_trade_net_return_avg_pct": float(np.mean(net_returns) * 100.0) if len(net_returns) else 0.0,
        "paper/executed_trade_gross_return_median_pct": float(np.median(gross_returns) * 100.0)
        if len(gross_returns)
        else 0.0,
        "paper/executed_trade_net_return_median_pct": float(np.median(net_returns) * 100.0) if len(net_returns) else 0.0,
        "paper/executed_trade_gross_return_total_pct": float(np.sum(gross_returns) * 100.0) if len(gross_returns) else 0.0,
        "paper/executed_trade_net_return_total_pct": float(np.sum(net_returns) * 100.0) if len(net_returns) else 0.0,
        "paper/transaction_cost_total_pct": float(transaction_cost_total_pct),
        "paper/long_n_executed_trades": int(np.sum([trade.signal_label == LABEL_UP for trade in executed])),
        "paper/short_n_executed_trades": int(np.sum([trade.signal_label == LABEL_DOWN for trade in executed])),
        "paper/long_hit_rate_pct": float(np.mean(long_hit_flags) * 100.0) if len(long_hit_flags) else 0.0,
        "paper/short_hit_rate_pct": float(np.mean(short_hit_flags) * 100.0) if len(short_hit_flags) else 0.0,
        "paper/long_net_return_avg_pct": float(np.mean(long_net_returns) * 100.0) if len(long_net_returns) else 0.0,
        "paper/short_net_return_avg_pct": float(np.mean(short_net_returns) * 100.0) if len(short_net_returns) else 0.0,
        "paper/n_test_days": int(n_days),
        "paper/tp": int(tp),
        "paper/tn": int(tn),
        "paper/fp": int(fp),
        "paper/fn": int(fn),
    }
