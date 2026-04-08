from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from kvant.labels import ACTED_LABELS, LABEL_DOWN, LABEL_UP


@dataclass(frozen=True)
class ExecutedTrade:
    tid: int | None
    signal_label: int
    true_label: int | None
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    gross_return: float
    net_return: float


def _parse_meta_timestamp(meta: dict, key: str) -> Optional[pd.Timestamp]:
    """Parse an ISO timestamp from a metadata record."""
    value = meta.get(key)
    if not isinstance(value, str) or not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _candidate_trade(
    *,
    y_pred_value: int,
    meta: Optional[dict],
    tid: int | None,
    transaction_cost: float,
) -> Optional[ExecutedTrade]:
    if meta is None or y_pred_value not in ACTED_LABELS:
        return None

    pnl_frac = meta.get("pnl_fraction")
    if not isinstance(pnl_frac, (int, float)):
        return None

    open_time = _parse_meta_timestamp(meta, "bar_open_time")
    close_time = _parse_meta_timestamp(meta, "bar_close_time")
    if open_time is None or close_time is None or close_time < open_time:
        return None

    gross_return = float(pnl_frac) if y_pred_value == LABEL_UP else -float(pnl_frac)
    net_return = gross_return - (2.0 * float(transaction_cost))

    return ExecutedTrade(
        tid=tid,
        signal_label=int(y_pred_value),
        true_label=int(meta["label"]) if isinstance(meta.get("label"), (int, float)) else None,
        open_time=open_time,
        close_time=close_time,
        gross_return=float(gross_return),
        net_return=float(net_return),
    )


def simulate_position_aware_trades(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> List[ExecutedTrade]:
    """Simulate non-overlapping trades from prediction signals."""
    assert len(y_pred) == len(metas)
    if tids is not None:
        assert len(tids) == len(y_pred)

    candidates: List[ExecutedTrade] = []
    for idx, (yp, meta) in enumerate(zip(y_pred, metas)):
        tid = None if tids is None else int(tids[idx])
        candidate = _candidate_trade(
            y_pred_value=int(yp),
            meta=meta,
            tid=tid,
            transaction_cost=transaction_cost,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda trade: (trade.open_time, trade.close_time))

    executed: List[ExecutedTrade] = []
    active_until: Optional[pd.Timestamp] = None
    for trade in candidates:
        if active_until is not None and trade.open_time < active_until:
            continue
        executed.append(trade)
        active_until = trade.close_time

    return executed


def per_ticker_trade_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
    transaction_cost: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker trade stats on non-overlapping executed trades.
    """
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        transaction_cost=transaction_cost,
    )
    by_tid: Dict[int, Dict[str, list]] = defaultdict(lambda: {"pct_change": [], "acc": []})

    for trade in executed:
        if trade.tid is None:
            continue

        by_tid[int(trade.tid)]["pct_change"].append(trade.net_return)
        if trade.true_label is not None:
            by_tid[int(trade.tid)]["acc"].append(int(trade.true_label) == int(trade.signal_label))

    out: Dict[int, Dict[str, Any]] = {}
    for tid, values in by_tid.items():
        pct_change = values["pct_change"]
        acc = values["acc"]
        out[tid] = {
            "n_trades": int(len(pct_change)),
            "bruto_profit_pct/avg": float(np.mean(pct_change) * 100.0) if pct_change else 0.0,
            "accuracy_call_put/avg": float(np.mean(acc)) if acc else 0.0,
        }
    return out


def compute_return_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[str, Any]:
    """Compute overall executed-trade return statistics."""
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        transaction_cost=transaction_cost,
    )

    out: Dict[str, Any] = {
        "n": int(len(metas)),
        "n_with_metadata": int(sum(meta is not None for meta in metas)),
    }
    trade_returns = [trade.net_return for trade in executed]
    trade_acc = [int(trade.true_label) == int(trade.signal_label) for trade in executed if trade.true_label is not None]
    out["accuracy_call_put/avg"] = float(np.mean(trade_acc)) if trade_acc else 0.0
    out["bruto_profit_pct/avg"] = float(np.mean(trade_returns) * 100.0) if trade_returns else 0.0
    return out


def compute_action_profit_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
    transaction_cost: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker profit stats split by action on executed trades.
    """
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        transaction_cost=transaction_cost,
    )

    buy_pnls: dict[int, list[float]] = defaultdict(list)
    short_pnls: dict[int, list[float]] = defaultdict(list)

    for trade in executed:
        if trade.tid is None:
            continue

        if trade.signal_label == LABEL_UP:
            buy_pnls[int(trade.tid)].append(trade.net_return)
        elif trade.signal_label == LABEL_DOWN:
            short_pnls[int(trade.tid)].append(trade.net_return)

    out: Dict[int, Dict[str, Any]] = {}
    all_tids = set(buy_pnls.keys()) | set(short_pnls.keys())
    for tid in all_tids:
        buys = buy_pnls.get(tid, [])
        shorts = short_pnls.get(tid, [])
        out[tid] = {
            "buy/n_trades": int(len(buys)),
            "buy/profit_pct/avg_per_trade": float(np.mean(buys) * 100.0) if buys else float("nan"),
            "buy/profit_pct/total": float(np.sum(buys) * 100.0) if buys else 0.0,
            "short/n_trades": int(len(shorts)),
            "short/profit_pct/avg_per_trade": float(np.mean(shorts) * 100.0) if shorts else float("nan"),
            "short/profit_pct/total": float(np.sum(shorts) * 100.0) if shorts else 0.0,
        }
    return out


def compute_profit_curve_over_trades(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[str, List[float]]:
    """Compute the cumulative profit curve over executed trades."""
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        transaction_cost=transaction_cost,
    )
    trade_profit_pct = [trade.net_return * 100.0 for trade in executed]
    cum_profit_pct = np.cumsum(np.asarray(trade_profit_pct, dtype=np.float64)).tolist()
    trade_number = list(range(1, len(trade_profit_pct) + 1))
    return {
        "trade_number": trade_number,
        "trade_profit_pct": trade_profit_pct,
        "cum_profit_pct": cum_profit_pct,
    }


def compute_paper_trading_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
    initial_portfolio: float = 1.0,
    transaction_cost: float = 0.0,
    risk_free_rate: float = 0.0314,
    days_per_year: float = 365.0,
) -> Dict[str, Any]:
    """
    Compute portfolio metrics aligned with the paper's evaluation setup.

    The implementation assumes that:
    - only predictions in classes ``0`` and ``2`` are acted upon,
    - ``meta["pnl_fraction"]`` is the realized long return for that sample,
    - class ``2`` is long/up and class ``0`` is short/down,
    - transaction costs are charged on both entry and exit.
    """
    assert len(y_true) == len(y_pred) == len(metas)
    if tids is not None:
        assert len(tids) == len(y_pred)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == LABEL_UP) & (y_pred == LABEL_UP)))
    tn = int(np.sum((y_true == LABEL_DOWN) & (y_pred == LABEL_DOWN)))
    fp = int(np.sum((y_true == LABEL_DOWN) & (y_pred == LABEL_UP)))
    fn = int(np.sum((y_true == LABEL_UP) & (y_pred == LABEL_DOWN)))
    accuracy_all_predictions = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0

    timeline_points: List[pd.Timestamp] = []
    for meta in metas:
        if meta is None:
            continue
        open_ts = _parse_meta_timestamp(meta, "bar_open_time")
        close_ts = _parse_meta_timestamp(meta, "bar_close_time")
        if open_ts is not None:
            timeline_points.append(open_ts)
        if close_ts is not None:
            timeline_points.append(close_ts)

    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        transaction_cost=transaction_cost,
    )

    if timeline_points:
        period_start = min(timeline_points).normalize()
        period_end = max(timeline_points).normalize()
    elif executed:
        period_start = executed[0].open_time.normalize()
        period_end = executed[-1].close_time.normalize()
    else:
        now = pd.Timestamp.now(tz="UTC").normalize()
        period_start = now
        period_end = now

    daily_index = pd.date_range(period_start, period_end, freq="D", tz="UTC")
    if len(daily_index) == 0:
        daily_index = pd.DatetimeIndex([period_start])

    portfolio_value = float(initial_portfolio)
    trade_records: List[dict[str, Any]] = []
    active_duration = pd.Timedelta(0)
    for trade in executed:
        portfolio_value *= max(0.0, 1.0 + float(trade.net_return))
        active_duration += trade.close_time - trade.open_time
        trade_records.append(
            {
                "close_time": trade.close_time,
                "portfolio_value": portfolio_value,
                "net_return": float(trade.net_return),
            }
        )

    if trade_records:
        trade_df = pd.DataFrame.from_records(trade_records)
        trade_df["date"] = pd.to_datetime(trade_df["close_time"], utc=True).dt.normalize()
        daily_portfolio = trade_df.groupby("date")["portfolio_value"].last().reindex(daily_index).ffill()
        daily_portfolio = daily_portfolio.fillna(float(initial_portfolio))
    else:
        daily_portfolio = pd.Series(float(initial_portfolio), index=daily_index, dtype=np.float64)

    daily_returns = daily_portfolio.pct_change().fillna(0.0)
    final_portfolio = float(daily_portfolio.iloc[-1])
    n_days = max(int(len(daily_portfolio)), 1)

    annual_net_profit_loss_pct = (
        ((final_portfolio / float(initial_portfolio)) ** (float(days_per_year) / float(n_days)) - 1.0) * 100.0
        if initial_portfolio > 0
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
    share_time_active_pct = float(active_duration.total_seconds() / total_duration * 100.0) if executed else 0.0

    return {
        "paper/annual_net_profit_loss_pct": float(annual_net_profit_loss_pct),
        "paper/profitable_transactions_pct": float(profitable_transactions_pct),
        "paper/accuracy_all_predictions": float(accuracy_all_predictions),
        "paper/sharpe_ratio_annualized": float(sharpe_ratio_annualized),
        "paper/max_drawdown_pct": float(max_drawdown_pct),
        "paper/share_time_active_pct": float(share_time_active_pct),
        "paper/n_executed_trades": int(len(executed)),
        "paper/n_test_days": int(n_days),
        "paper/tp": int(tp),
        "paper/tn": int(tn),
        "paper/fp": int(fp),
        "paper/fn": int(fn),
    }
