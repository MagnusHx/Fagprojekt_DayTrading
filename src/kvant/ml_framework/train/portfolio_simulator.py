from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from kvant.labels import LABEL_DOWN, LABEL_UP

from .backtest import BacktestTrade, BacktestTradeSimulator


@dataclass
class PortfolioPosition:
    trade: BacktestTrade
    side: int
    shares: float
    entry_notional: float
    entry_cost: float
    mark_price: float


@dataclass
class PortfolioSimulationResult:
    metrics: Dict[str, Any]
    equity_curve: Dict[str, list[Any]]


def _mark_price(
    simulator: BacktestTradeSimulator,
    *,
    tid: int,
    timestamp: pd.Timestamp,
    fallback: float,
) -> float:
    market_data = simulator.market_data_store.market_data(int(tid))
    if market_data is None:
        return float(fallback)
    timestamps = pd.DatetimeIndex(pd.to_datetime(market_data["timestamp"], utc=True))
    closes = np.asarray(market_data["close"], dtype=np.float64)
    pos = int(timestamps.searchsorted(timestamp, side="right") - 1)
    if pos < 0 or pos >= len(closes):
        return float(fallback)
    price = float(closes[pos])
    return price if np.isfinite(price) and price > 0.0 else float(fallback)


def _position_value(position: PortfolioPosition) -> float:
    value = float(position.shares) * float(position.mark_price)
    return value if int(position.side) == LABEL_UP else -value


def _position_exposure(position: PortfolioPosition) -> float:
    return abs(float(position.shares) * float(position.mark_price))


def _portfolio_equity(cash: float, positions: list[PortfolioPosition]) -> float:
    return float(cash + sum(_position_value(position) for position in positions))


def _portfolio_exposure(positions: list[PortfolioPosition]) -> float:
    return float(sum(_position_exposure(position) for position in positions))


def _empty_curve(initial_cash: float) -> Dict[str, list[Any]]:
    return {
        "step": [0],
        "timestamp": [],
        "equity": [float(initial_cash)],
        "cash": [float(initial_cash)],
        "exposure_pct": [0.0],
        "open_positions": [0],
    }


def _curve_metrics(
    *,
    initial_cash: float,
    equity_curve: Dict[str, list[Any]],
    realized_trade_pnl: list[float],
    realized_trade_return: list[float],
    transaction_cost_total: float,
    n_skipped_budget: int,
    days_per_year: float,
    risk_free_rate: float,
) -> Dict[str, Any]:
    equity = np.asarray(equity_curve["equity"], dtype=np.float64)
    exposure_pct = np.asarray(equity_curve["exposure_pct"], dtype=np.float64)
    final_balance = float(equity[-1]) if len(equity) else float(initial_cash)
    total_return_pct = float((final_balance / float(initial_cash) - 1.0) * 100.0) if float(initial_cash) > 0.0 else 0.0

    timestamps = equity_curve.get("timestamp", [])
    if timestamps:
        series = pd.Series(equity[1:], index=pd.to_datetime(timestamps, utc=True))
        daily = series.groupby(series.index.normalize()).last()
        if daily.empty:
            daily = pd.Series([float(initial_cash)], index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").normalize()]))
        daily = pd.concat([pd.Series([float(initial_cash)], index=[daily.index.min() - pd.Timedelta(days=1)]), daily])
        daily_returns = daily.pct_change().fillna(0.0)
        n_days = max(int(len(daily)), 1)
        year_end_equity = daily.groupby(daily.index.year).last()
        year_start_equity = pd.Series(
            [float(initial_cash), *year_end_equity.iloc[:-1].tolist()],
            index=year_end_equity.index,
            dtype=np.float64,
        )
        annual_profit_pct = (year_end_equity / year_start_equity - 1.0) * 100.0
        cumulative_annual_profit_pct = float(annual_profit_pct.sum())
    else:
        daily_returns = pd.Series([0.0], dtype=np.float64)
        n_days = 1
        cumulative_annual_profit_pct = 0.0

    annual_return_pct = (
        ((final_balance / float(initial_cash)) ** (float(days_per_year) / float(n_days)) - 1.0) * 100.0
        if float(initial_cash) > 0.0 and final_balance > 0.0
        else -100.0
        if final_balance <= 0.0
        else 0.0
    )
    risk_free_daily = (1.0 + float(risk_free_rate)) ** (1.0 / float(days_per_year)) - 1.0
    daily_std = float(daily_returns.std(ddof=0))
    sharpe = (
        float(np.sqrt(float(days_per_year)) * ((daily_returns.mean() - risk_free_daily) / daily_std))
        if daily_std > 0.0
        else 0.0
    )
    running_peak = np.maximum.accumulate(np.maximum(equity, 1e-12))
    max_drawdown_pct = float(np.max((running_peak - equity) / running_peak) * 100.0) if len(equity) else 0.0

    pnl = np.asarray(realized_trade_pnl, dtype=np.float64)
    returns = np.asarray(realized_trade_return, dtype=np.float64)

    return {
        "portfolio/total_return_pct": float(total_return_pct),
        "portfolio/cumulative_annual_profit_pct": float(cumulative_annual_profit_pct),
        "portfolio/annualized_return_pct": float(annual_return_pct),
        "portfolio/max_drawdown_pct": float(max_drawdown_pct),
        "portfolio/sharpe_ratio_annualized": float(sharpe),
        "portfolio/average_trade_return_pct": float(np.mean(returns) * 100.0) if len(returns) else 0.0,
        "portfolio/average_exposure_pct": float(np.mean(exposure_pct)) if len(exposure_pct) else 0.0,
        "portfolio/transaction_cost_total": float(transaction_cost_total),
        "portfolio/n_executed_trades": int(len(pnl)),
        "portfolio/n_skipped_budget": int(n_skipped_budget),
    }


def compute_portfolio_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tids: np.ndarray,
    tpos: np.ndarray,
    simulator: BacktestTradeSimulator,
    bet_sizes: Optional[np.ndarray] = None,
    initial_cash: float = 10_000.0,
    max_position_fraction: float = 0.05,
    max_total_exposure: float = 1.0,
    max_positions: int = 10,
    risk_free_rate: float = 0.0314,
    days_per_year: float = 365.0,
) -> PortfolioSimulationResult:
    """Simulate a cash/exposure constrained portfolio and return balance metrics."""
    if float(initial_cash) <= 0.0:
        raise ValueError("initial_cash must be positive.")
    if not (0.0 < float(max_position_fraction) <= 1.0):
        raise ValueError("max_position_fraction must be in (0, 1].")
    if float(max_total_exposure) <= 0.0:
        raise ValueError("max_total_exposure must be positive.")
    if int(max_positions) <= 0:
        raise ValueError("max_positions must be positive.")
    if not (len(y_true) == len(y_pred) == len(tids) == len(tpos)):
        raise ValueError("y_true, y_pred, tids, and tpos must have the same length.")

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tids = np.asarray(tids, dtype=np.int64)
    tpos = np.asarray(tpos, dtype=np.int64)
    bet_sizes = (
        np.asarray(bet_sizes, dtype=np.float64) if bet_sizes is not None else np.ones(len(y_pred), dtype=np.float64)
    )
    if len(bet_sizes) != len(y_pred):
        raise ValueError("bet_sizes must match y_pred length.")

    candidate_trades = simulator.simulate(y_pred=y_pred, tids=tids, tpos=tpos, y_true=y_true, bet_sizes=bet_sizes)
    entries_by_time: dict[pd.Timestamp, list[BacktestTrade]] = {}
    exits_by_time: dict[pd.Timestamp, list[BacktestTrade]] = {}
    for trade in candidate_trades:
        entries_by_time.setdefault(trade.entry_time, []).append(trade)
        exits_by_time.setdefault(trade.exit_time, []).append(trade)

    event_times = sorted(set(entries_by_time) | set(exits_by_time))
    cash = float(initial_cash)
    positions: list[PortfolioPosition] = []
    open_by_key: dict[tuple[int, int], PortfolioPosition] = {}
    realized_trade_pnl: list[float] = []
    realized_trade_return: list[float] = []
    transaction_cost_total = 0.0
    n_skipped_budget = 0
    equity_curve = _empty_curve(float(initial_cash))

    def mark_positions(timestamp: pd.Timestamp) -> None:
        for position in positions:
            position.mark_price = _mark_price(
                simulator,
                tid=position.trade.tid,
                timestamp=timestamp,
                fallback=position.mark_price,
            )

    def append_curve(timestamp: pd.Timestamp) -> None:
        equity = _portfolio_equity(cash, positions)
        exposure = _portfolio_exposure(positions)
        exposure_pct = float(exposure / max(equity, 1e-12) * 100.0) if equity > 0.0 else 0.0
        equity_curve["step"].append(len(equity_curve["step"]))
        equity_curve["timestamp"].append(timestamp.isoformat())
        equity_curve["equity"].append(float(equity))
        equity_curve["cash"].append(float(cash))
        equity_curve["exposure_pct"].append(float(exposure_pct))
        equity_curve["open_positions"].append(int(len(positions)))

    def close_position(trade: BacktestTrade) -> bool:
        nonlocal cash, transaction_cost_total

        key = (int(trade.tid), int(trade.signal_tpos))
        position = open_by_key.pop(key, None)
        if position is None:
            return False

        exit_notional = float(position.shares) * float(trade.exit_price)
        exit_cost = exit_notional * float(simulator.transaction_cost)
        transaction_cost_total += exit_cost
        if int(position.side) == LABEL_UP:
            cash += exit_notional - exit_cost
            pnl = exit_notional - position.entry_notional - position.entry_cost - exit_cost
        else:
            cash -= exit_notional + exit_cost
            pnl = position.entry_notional - exit_notional - position.entry_cost - exit_cost
        positions.remove(position)
        realized_trade_pnl.append(float(pnl))
        realized_trade_return.append(float(pnl / max(position.entry_notional, 1e-12)))
        return True

    for timestamp in event_times:
        mark_positions(timestamp)

        for trade in exits_by_time.get(timestamp, []):
            close_position(trade)

        mark_positions(timestamp)
        for trade in entries_by_time.get(timestamp, []):
            if len(positions) >= int(max_positions):
                n_skipped_budget += 1
                continue
            equity = _portfolio_equity(cash, positions)
            if equity <= 0.0:
                n_skipped_budget += 1
                continue
            exposure = _portfolio_exposure(positions)
            remaining_exposure = max(float(max_total_exposure) * equity - exposure, 0.0)
            desired_notional = equity * float(max_position_fraction) * min(max(float(trade.bet_size), 0.0), 1.0)
            notional = min(desired_notional, remaining_exposure)
            if notional <= 1e-9:
                n_skipped_budget += 1
                continue

            entry_cost = notional * float(simulator.transaction_cost)
            if int(trade.signal_label) == LABEL_UP:
                if cash < notional + entry_cost:
                    notional = max((cash / (1.0 + float(simulator.transaction_cost))), 0.0)
                    entry_cost = notional * float(simulator.transaction_cost)
                if notional <= 1e-9:
                    n_skipped_budget += 1
                    continue
                cash -= notional + entry_cost
            elif int(trade.signal_label) == LABEL_DOWN:
                cash += notional - entry_cost
            else:
                continue

            transaction_cost_total += entry_cost
            shares = notional / float(trade.entry_price)
            position = PortfolioPosition(
                trade=trade,
                side=int(trade.signal_label),
                shares=float(shares),
                entry_notional=float(notional),
                entry_cost=float(entry_cost),
                mark_price=float(trade.entry_price),
            )
            positions.append(position)
            open_by_key[(int(trade.tid), int(trade.signal_tpos))] = position

        for trade in exits_by_time.get(timestamp, []):
            close_position(trade)

        append_curve(timestamp)

    metrics = _curve_metrics(
        initial_cash=float(initial_cash),
        equity_curve=equity_curve,
        realized_trade_pnl=realized_trade_pnl,
        realized_trade_return=realized_trade_return,
        transaction_cost_total=float(transaction_cost_total),
        n_skipped_budget=n_skipped_budget,
        days_per_year=float(days_per_year),
        risk_free_rate=float(risk_free_rate),
    )
    return PortfolioSimulationResult(metrics=metrics, equity_curve=equity_curve)
