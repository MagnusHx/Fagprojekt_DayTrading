from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute aggregate, per-class, and class-distribution metrics."""
    if len(y_true) == 0:
        return {"accuracy": 0.0}

    labels = np.asarray([0, 1, 2], dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    y_true_counts = np.bincount(y_true.astype(np.int64, copy=False), minlength=len(labels))
    y_pred_counts = np.bincount(y_pred.astype(np.int64, copy=False), minlength=len(labels))
    n_total = max(int(len(y_true)), 1)

    for idx, label in enumerate(labels):
        out[f"precision_class_{label}"] = float(precision[idx])
        out[f"recall_class_{label}"] = float(recall[idx])
        out[f"f1_class_{label}"] = float(f1[idx])
        out[f"support_class_{label}"] = int(support[idx])
        out[f"y_true_count_class_{label}"] = int(y_true_counts[idx])
        out[f"y_pred_count_class_{label}"] = int(y_pred_counts[idx])
        out[f"y_true_pct_class_{label}"] = float(y_true_counts[idx] / n_total)
        out[f"y_pred_pct_class_{label}"] = float(y_pred_counts[idx] / n_total)

    return out


def per_ticker_trade_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Computes per-ticker trade stats based on metadata.

    A "trade" is only counted when prediction is call/put (0 or 2) AND metadata has pnl_fraction.

    Returns:
      { tid: {
          "n_trades": int,
          "bruto_profit_pct/avg": float,          # avg profit per trade in percent
          "accuracy_call_put/avg": float,         # accuracy restricted to call/put trades with metadata
        }, ... }
    """
    assert len(y_pred) == len(metas) == len(tids)

    by_tid: Dict[int, Dict[str, list]] = defaultdict(lambda: {"pct_change": [], "acc": []})

    for i in range(len(y_pred)):
        m = metas[i]
        if m is None:
            continue

        yp = int(y_pred[i])
        if yp not in (0, 2):
            continue

        pnl_frac = m.get("pnl_fraction", None)
        true_label = m.get("label", None)
        if not isinstance(pnl_frac, (int, float)):
            continue

        tid = int(tids[i])
        signed = (-1.0 if yp == 0 else 1.0) * float(pnl_frac)
        by_tid[tid]["pct_change"].append(signed)
        by_tid[tid]["acc"].append(true_label == yp)

    out: Dict[int, Dict[str, Any]] = {}
    for tid, d in by_tid.items():
        pct_change = d["pct_change"]
        acc = d["acc"]
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
) -> Dict[str, Any]:
    """
    Overall (split-level) metadata-based statistics.
    """
    assert len(y_pred) == len(metas)
    if tids is not None:
        assert len(tids) == len(y_pred)

    out: Dict[str, Any] = {}
    n_total = len(metas)
    n_with_meta = sum(m is not None for m in metas)

    out["n"] = int(n_total)
    out["n_with_metadata"] = int(n_with_meta)

    pct_change = []
    acc_call_put = []

    for i, yp in enumerate(y_pred):
        m = metas[i]
        if m is None:
            continue

        yp = int(yp)
        if yp in (0, 2):  # call / put
            pnl_frac = m.get("pnl_fraction", None)
            true_label = m.get("label", None)
            if not isinstance(pnl_frac, (int, float)):
                continue

            pct_change.append((-1 if yp == 0 else 1) * float(pnl_frac))
            acc_call_put.append(true_label == yp)

    out["accuracy_call_put/avg"] = float(np.mean(acc_call_put)) if acc_call_put else 0.0
    out["bruto_profit_pct/avg"] = float(np.mean(pct_change) * 100.0) if pct_change else 0.0
    return out


def compute_action_profit_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker profit stats split by action:

      Action BUY:   y_pred == 2  -> signed profit = +pnl_fraction
      Action SHORT: y_pred == 0  -> signed profit = -pnl_fraction

    All values are in percent (%), derived from pnl_fraction * 100.

    Returns:
      { tid: {
          "buy/n_trades": int,
          "buy/profit_pct/avg_per_trade": float (nan if no trades),
          "buy/profit_pct/total": float (0 if no trades),

          "short/n_trades": int,
          "short/profit_pct/avg_per_trade": float (nan if no trades),
          "short/profit_pct/total": float (0 if no trades),
      }}
    """
    assert len(y_pred) == len(metas) == len(tids)

    # accumulate signed pnl fractions
    buy_pnls = defaultdict(list)  # tid -> list[pnl_frac]
    short_pnls = defaultdict(list)  # tid -> list[-pnl_frac]

    for i in range(len(y_pred)):
        m = metas[i]
        if m is None:
            continue

        pnl_frac = m.get("pnl_fraction", None)
        if not isinstance(pnl_frac, (int, float)):
            continue

        tid = int(tids[i])
        yp = int(y_pred[i])

        if yp == 2:
            buy_pnls[tid].append(float(pnl_frac))
        elif yp == 0:
            short_pnls[tid].append(-float(pnl_frac))

    out: Dict[int, Dict[str, Any]] = {}
    all_tids = set(buy_pnls.keys()) | set(short_pnls.keys())

    for tid in all_tids:
        b = buy_pnls.get(tid, [])
        s = short_pnls.get(tid, [])

        out[tid] = {
            "buy/n_trades": int(len(b)),
            "buy/profit_pct/avg_per_trade": float(np.mean(b) * 100.0) if len(b) else float("nan"),
            "buy/profit_pct/total": float(np.sum(b) * 100.0) if len(b) else 0.0,
            "short/n_trades": int(len(s)),
            "short/profit_pct/avg_per_trade": float(np.mean(s) * 100.0) if len(s) else float("nan"),
            "short/profit_pct/total": float(np.sum(s) * 100.0) if len(s) else 0.0,
        }

    return out


def compute_profit_curve_over_trades(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
) -> Dict[str, List[float]]:
    """
    Compute the cumulative profit curve over executed trades.

    A trade is counted only when the prediction is buy/short (2 or 0) and the
    sample metadata contains a numeric ``pnl_fraction`` value.

    Returns:
      Dict containing:
        trade_number:
          One-based trade indices.
        trade_profit_pct:
          Signed profit for each trade in percent.
        cum_profit_pct:
          Cumulative signed profit in percent.
    """
    assert len(y_pred) == len(metas)

    trade_profit_pct: List[float] = []

    for yp, meta in zip(y_pred, metas):
        if meta is None:
            continue

        yp = int(yp)
        if yp not in (0, 2):
            continue

        pnl_frac = meta.get("pnl_fraction", None)
        if not isinstance(pnl_frac, (int, float)):
            continue

        signed_profit_pct = (float(pnl_frac) if yp == 2 else -float(pnl_frac)) * 100.0
        trade_profit_pct.append(signed_profit_pct)

    cum_profit_pct = np.cumsum(np.asarray(trade_profit_pct, dtype=np.float64)).tolist()
    trade_number = list(range(1, len(trade_profit_pct) + 1))

    return {
        "trade_number": trade_number,
        "trade_profit_pct": trade_profit_pct,
        "cum_profit_pct": cum_profit_pct,
    }


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


def compute_paper_trading_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    initial_portfolio: float = 1.0,
    transaction_cost: float = 0.0004,
    risk_free_rate: float = 0.0314,
    days_per_year: float = 365.0,
) -> Dict[str, Any]:
    """
    Compute portfolio metrics aligned with the paper's evaluation setup.

    The implementation assumes that:
    - only predictions in classes ``0`` and ``2`` are acted upon,
    - ``meta["pnl_fraction"]`` is the realized long return for that sample,
    - short trades invert the sign of ``pnl_fraction``,
    - transaction costs are charged on both entry and exit.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        metas: Label metadata aligned to the predictions.
        initial_portfolio: Initial portfolio value ``P0``.
        transaction_cost: Per-side transaction cost expressed as a return fraction.
        risk_free_rate: Annualized risk-free rate as a decimal.
        days_per_year: Trading days per year used for annualization.

    Returns:
        Dictionary with annualized profit, trade hit rate, directional accuracy,
        Sharpe ratio, drawdown, and a few supporting counters.
    """
    assert len(y_true) == len(y_pred) == len(metas)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 2) & (y_pred == 2)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 2)))
    fn = int(np.sum((y_true == 2) & (y_pred == 0)))
    directional_total = tp + tn + fp + fn
    directional_accuracy = float((tp + tn) / directional_total) if directional_total else 0.0

    timeline_points: List[pd.Timestamp] = []
    trades: List[dict[str, Any]] = []

    for yp, meta in zip(y_pred, metas):
        if meta is None:
            continue

        open_ts = _parse_meta_timestamp(meta, "bar_open_time")
        close_ts = _parse_meta_timestamp(meta, "bar_close_time")
        if open_ts is not None:
            timeline_points.append(open_ts)
        if close_ts is not None:
            timeline_points.append(close_ts)

        if int(yp) not in (0, 2):
            continue

        pnl_frac = meta.get("pnl_fraction")
        if not isinstance(pnl_frac, (int, float)):
            continue

        signed_gross_return = float(pnl_frac) if int(yp) == 2 else -float(pnl_frac)
        net_return = signed_gross_return - (2.0 * float(transaction_cost))
        trades.append(
            {
                "close_time": close_ts or open_ts,
                "gross_return": signed_gross_return,
                "net_return": net_return,
            }
        )

    trades = [trade for trade in trades if trade["close_time"] is not None]
    trades.sort(key=lambda trade: trade["close_time"])

    if timeline_points:
        period_start = min(timeline_points).normalize()
        period_end = max(timeline_points).normalize()
    elif trades:
        period_start = trades[0]["close_time"].normalize()
        period_end = trades[-1]["close_time"].normalize()
    else:
        now = pd.Timestamp.now(tz="UTC").normalize()
        period_start = now
        period_end = now

    daily_index = pd.date_range(period_start, period_end, freq="D", tz="UTC")
    if len(daily_index) == 0:
        daily_index = pd.DatetimeIndex([period_start])

    portfolio_value = float(initial_portfolio)
    trade_records: List[dict[str, Any]] = []
    for trade in trades:
        portfolio_value *= max(0.0, 1.0 + float(trade["net_return"]))
        trade_records.append(
            {
                "close_time": trade["close_time"],
                "portfolio_value": portfolio_value,
                "net_return": float(trade["net_return"]),
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

    n_executed_trades = len(trades)
    profitable_transactions_pct = (
        float(np.mean([trade["net_return"] > 0.0 for trade in trades]) * 100.0) if trades else 0.0
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

    return {
        "paper/annual_net_profit_loss_pct": float(annual_net_profit_loss_pct),
        "paper/profitable_transactions_pct": float(profitable_transactions_pct),
        "paper/directional_accuracy": float(directional_accuracy),
        "paper/sharpe_ratio_annualized": float(sharpe_ratio_annualized),
        "paper/max_drawdown_pct": float(max_drawdown_pct),
        "paper/n_executed_trades": int(n_executed_trades),
        "paper/n_test_days": int(n_days),
        "paper/tp": int(tp),
        "paper/tn": int(tn),
        "paper/fp": int(fp),
        "paper/fn": int(fn),
    }
