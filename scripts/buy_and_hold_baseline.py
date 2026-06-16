#!/usr/bin/env python3
"""Run an equal-weight buy-and-hold baseline on prepared walk-forward folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from kvant.ml_framework.utils.statistical_tests import calculate_ci
from kvant.ml_framework.wandb_defaults import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, wandb_init_kwargs
from kvant.ml_prepare_data.data_loading import PreparedExperiment


def load_cv_manifest(path: Path) -> dict:
    """Load a prepared CV manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def _to_utc_index(values: np.ndarray) -> pd.DatetimeIndex:
    """Convert persisted timestamps to a UTC DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Compute maximum drawdown in percent from an equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max.replace(0.0, np.nan)
    return float(drawdown.max(skipna=True) * 100.0)


def _annualized_return_pct(total_return: float, first_ts: pd.Timestamp, last_ts: pd.Timestamp, days_per_year: float) -> float:
    """Annualize total return over a timestamp interval."""
    days = max((last_ts - first_ts).total_seconds() / 86400.0, 1e-9)
    return float(((1.0 + total_return) ** (float(days_per_year) / days) - 1.0) * 100.0)


def _sharpe_ratio_annualized(equity: pd.Series, risk_free_rate: float, days_per_year: float) -> float:
    """Compute annualized Sharpe ratio from daily equity values."""
    if equity.empty:
        return 0.0
    daily = equity.resample("1D").last().ffill().pct_change().dropna()
    if len(daily) < 2:
        return 0.0
    excess = daily.to_numpy(dtype=float) - float(risk_free_rate) / float(days_per_year)
    std = float(np.std(excess, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(float(days_per_year)))


def _fold_buy_and_hold(
    experiment: PreparedExperiment,
    *,
    transaction_cost: float,
    risk_free_rate: float,
    days_per_year: float,
) -> dict[str, float | int]:
    """Compute equal-weight buy-and-hold metrics for one prepared fold."""
    test_index = np.asarray(experiment.index_test, dtype=np.int64)
    if len(test_index) == 0:
        return {
            "test_portfolio_total_return_pct": 0.0,
            "test_portfolio_annualized_return_pct": 0.0,
            "test_portfolio_sharpe_ratio_annualized": 0.0,
            "test_portfolio_max_drawdown_pct": 0.0,
            "test_portfolio_average_trade_return_pct": 0.0,
            "test_portfolio_n_executed_trades": 0,
        }

    ticker_curves: list[pd.Series] = []
    ticker_returns: list[float] = []

    for tid in np.unique(test_index[:, 0]):
        positions = np.sort(test_index[test_index[:, 0] == tid, 1])
        if len(positions) < 2:
            continue

        market_data = experiment.store.require_market_data(int(tid))
        timestamps = _to_utc_index(market_data["timestamp"][positions])
        closes = np.asarray(market_data["close"], dtype=np.float64)[positions]
        valid = np.isfinite(closes) & (closes > 0.0)
        if valid.sum() < 2:
            continue

        timestamps = timestamps[valid]
        closes = closes[valid]
        first_close = float(closes[0])
        last_close = float(closes[-1])
        net_return = (last_close * (1.0 - transaction_cost)) / (first_close * (1.0 + transaction_cost)) - 1.0
        ticker_returns.append(float(net_return))

        curve = pd.Series(closes / (first_close * (1.0 + transaction_cost)), index=timestamps)
        curve.iloc[-1] = curve.iloc[-1] * (1.0 - transaction_cost)
        ticker_curves.append(curve[~curve.index.duplicated(keep="last")])

    if not ticker_curves:
        raise RuntimeError(f"No usable buy-and-hold ticker curves found for {experiment.exp_dir}.")

    aligned = pd.concat(ticker_curves, axis=1).sort_index().ffill().dropna(how="all")
    equity = aligned.mean(axis=1).dropna()
    total_return = float(equity.iloc[-1] - 1.0)
    average_trade_return = float(np.mean(ticker_returns)) if ticker_returns else total_return

    return {
        "test_portfolio_total_return_pct": total_return * 100.0,
        "test_portfolio_annualized_return_pct": _annualized_return_pct(
            total_return, equity.index[0], equity.index[-1], days_per_year
        ),
        "test_portfolio_sharpe_ratio_annualized": _sharpe_ratio_annualized(equity, risk_free_rate, days_per_year),
        "test_portfolio_max_drawdown_pct": _max_drawdown_pct(equity),
        "test_portfolio_average_trade_return_pct": average_trade_return * 100.0,
        "test_portfolio_n_executed_trades": int(len(ticker_returns)),
    }


def run_buy_and_hold(
    manifest: dict,
    *,
    transaction_cost: float,
    risk_free_rate: float,
    days_per_year: float,
) -> list[dict[str, float | int | str]]:
    """Run buy-and-hold across all manifest folds."""
    rows: list[dict[str, float | int | str]] = []
    for fold in manifest["folds"]:
        fold_idx = int(fold["fold_idx"])
        experiment = PreparedExperiment(Path(fold["exp_dir"]))
        metrics = _fold_buy_and_hold(
            experiment,
            transaction_cost=float(transaction_cost),
            risk_free_rate=float(risk_free_rate),
            days_per_year=float(days_per_year),
        )
        row = {"fold": fold_idx, "exp_dir": str(fold["exp_dir"]), **metrics}
        rows.append(row)
        wandb.log({f"fold{fold_idx:02d}/test/portfolio_total_return_pct": metrics["test_portfolio_total_return_pct"]})
        print(
            f"fold={fold_idx:02d} total_return={metrics['test_portfolio_total_return_pct']:.4f}% "
            f"sharpe={metrics['test_portfolio_sharpe_ratio_annualized']:.4f}"
        )
    return rows


def _log_summary(rows: list[dict[str, float | int | str]]) -> None:
    """Log fold-level summary statistics to W&B and stdout."""
    metrics = [
        "test_portfolio_total_return_pct",
        "test_portfolio_annualized_return_pct",
        "test_portfolio_sharpe_ratio_annualized",
        "test_portfolio_max_drawdown_pct",
        "test_portfolio_average_trade_return_pct",
        "test_portfolio_n_executed_trades",
    ]
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        ci_lower, ci_upper = calculate_ci(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        wandb.log({
            f"{metric}/mean": mean,
            f"{metric}/std": std,
            f"{metric}/ci_lower": ci_lower,
            f"{metric}/ci_upper": ci_upper,
        })
        print(f"{metric}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")


def main() -> None:
    """Run the buy-and-hold baseline CLI."""
    parser = argparse.ArgumentParser(description="Run equal-weight buy-and-hold on prepared CV folds.")
    parser.add_argument("--cv-manifest", type=Path, required=True)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-free-rate", type=float, default=0.0314)
    parser.add_argument("--days-per-year", type=float, default=365.0)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-name", type=str, default="E0-buy-and-hold")
    parser.add_argument("--output", type=Path, default=Path("results/baselines/E0_buy_and_hold.csv"))
    args = parser.parse_args()

    wandb.init(
        **wandb_init_kwargs(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "cv_manifest": str(args.cv_manifest),
                "transaction_cost": float(args.transaction_cost),
                "risk_free_rate": float(args.risk_free_rate),
                "days_per_year": float(args.days_per_year),
            },
        ),
    )

    manifest = load_cv_manifest(args.cv_manifest)
    rows = run_buy_and_hold(
        manifest,
        transaction_cost=float(args.transaction_cost),
        risk_free_rate=float(args.risk_free_rate),
        days_per_year=float(args.days_per_year),
    )
    _log_summary(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    wandb.log({"status": "complete"})
    wandb.finish()


if __name__ == "__main__":
    main()
