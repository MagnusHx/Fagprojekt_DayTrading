from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from kvant.labels import ACTED_LABELS, LABEL_DOWN, LABEL_EXIT, LABEL_UP


@dataclass(frozen=True)
class ExecutedTrade:
    tid: int | None
    signal_label: int
    true_label: int | None
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    bet_size: float
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
    bet_size: float,
    transaction_cost: float,
) -> Optional[ExecutedTrade]:
    if meta is None or y_pred_value not in ACTED_LABELS:
        return None
    if not np.isfinite(float(bet_size)) or float(bet_size) <= 0.0:
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
    gross_return *= float(bet_size)
    net_return *= float(bet_size)

    return ExecutedTrade(
        tid=tid,
        signal_label=int(y_pred_value),
        true_label=int(meta["label"]) if isinstance(meta.get("label"), (int, float)) else None,
        open_time=open_time,
        close_time=close_time,
        bet_size=float(bet_size),
        gross_return=float(gross_return),
        net_return=float(net_return),
    )


def trade_decision_components(
    *,
    y_pred_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return action probability and directional confidence from class probabilities."""
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] not in (2, 3):
        raise ValueError("y_pred_proba must have shape (n_samples, 2) or (n_samples, 3)")

    if y_pred_proba.shape[1] == 2:
        p_down = y_pred_proba[:, 0]
        q_up = y_pred_proba[:, 1]
        p_act = np.ones(len(q_up), dtype=np.float64)
        return p_act, q_up

    p_down = y_pred_proba[:, LABEL_DOWN]
    p_up = y_pred_proba[:, LABEL_UP]
    p_act = p_down + p_up
    denom = np.clip(p_act, 1e-12, None)
    q_up = p_up / denom
    q_up = np.where(p_act > 0.0, q_up, 0.5)
    return p_act, q_up


def apply_trade_decision_thresholds(
    *,
    y_pred_proba: np.ndarray,
    trade_action_threshold: float,
    trade_direction_threshold: float,
) -> np.ndarray:
    """Map class probabilities to down/exit/up using action and directional thresholds."""
    p_act, q_up = trade_decision_components(y_pred_proba=y_pred_proba)
    return apply_trade_decision_bands(
        p_act=p_act,
        q_up=q_up,
        trade_action_threshold=trade_action_threshold,
        trade_direction_threshold=trade_direction_threshold,
    )


def apply_trade_decision_bands(
    *,
    p_act: np.ndarray,
    q_up: np.ndarray,
    trade_action_threshold: float,
    trade_direction_threshold: float,
) -> np.ndarray:
    """Map action mass and directional probability to down/exit/up."""
    if not (0.0 <= float(trade_action_threshold) <= 1.0):
        raise ValueError("trade_action_threshold must be between 0 and 1")
    if not (0.5 <= float(trade_direction_threshold) <= 1.0):
        raise ValueError("trade_direction_threshold must be between 0.5 and 1.0")
    p_act = np.asarray(p_act, dtype=np.float64)
    q_up = np.asarray(q_up, dtype=np.float64)
    if len(p_act) != len(q_up):
        raise ValueError(f"p_act and q_up must have the same length, got {len(p_act)} and {len(q_up)}.")

    out = np.full(len(p_act), LABEL_EXIT, dtype=np.int64)
    actionable = p_act >= float(trade_action_threshold)
    out[actionable & (q_up >= float(trade_direction_threshold))] = LABEL_UP
    out[actionable & (q_up <= 1.0 - float(trade_direction_threshold))] = LABEL_DOWN
    return out


def apply_trade_confidence_threshold(
    *,
    y_pred: np.ndarray,
    y_pred_confidence: np.ndarray,
    trade_confidence_threshold: float,
) -> np.ndarray:
    """Legacy argmax-confidence gating kept for backwards compatibility."""
    assert len(y_pred) == len(y_pred_confidence)

    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_pred_confidence = np.asarray(y_pred_confidence, dtype=np.float64)
    out = y_pred.copy()

    if trade_confidence_threshold <= 0.0:
        return out

    low_confidence_trade_mask = np.isin(out, ACTED_LABELS) & (y_pred_confidence < float(trade_confidence_threshold))
    out[low_confidence_trade_mask] = LABEL_EXIT
    return out


def simulate_position_aware_trades(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
    bet_sizes: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> List[ExecutedTrade]:
    """Simulate non-overlapping trades from prediction signals.

    Trades may overlap across different tickers, but overlapping trades for the
    same ticker are collapsed so only one position per ticker can be active at a time.
    """
    assert len(y_pred) == len(metas)
    if tids is not None:
        assert len(tids) == len(y_pred)
    if bet_sizes is not None:
        assert len(bet_sizes) == len(y_pred)

    candidates: List[ExecutedTrade] = []
    for idx, (yp, meta) in enumerate(zip(y_pred, metas)):
        tid = None if tids is None else int(tids[idx])
        bet_size = 1.0 if bet_sizes is None else float(bet_sizes[idx])
        candidate = _candidate_trade(
            y_pred_value=int(yp),
            meta=meta,
            tid=tid,
            bet_size=bet_size,
            transaction_cost=transaction_cost,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda trade: (trade.open_time, trade.close_time))

    executed: List[ExecutedTrade] = []
    active_until_by_tid: dict[int | None, pd.Timestamp] = {}
    for trade in candidates:
        active_until = active_until_by_tid.get(trade.tid)
        if active_until is not None and trade.open_time < active_until:
            continue
        executed.append(trade)
        active_until_by_tid[trade.tid] = trade.close_time

    return executed


def per_ticker_trade_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
    bet_sizes: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker trade stats on non-overlapping executed trades.
    """
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        bet_sizes=bet_sizes,
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
    bet_sizes: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[str, Any]:
    """Compute overall executed-trade return statistics."""
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        bet_sizes=bet_sizes,
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
    bet_sizes: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker profit stats split by action on executed trades.
    """
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        bet_sizes=bet_sizes,
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
    bet_sizes: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
) -> Dict[str, List[float]]:
    """Compute the cumulative profit curve over executed trades."""
    executed = simulate_position_aware_trades(
        y_pred=y_pred,
        metas=metas,
        tids=tids,
        bet_sizes=bet_sizes,
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
