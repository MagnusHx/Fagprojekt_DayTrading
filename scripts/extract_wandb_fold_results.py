#!/usr/bin/env python3
"""Extract per-fold result CSVs from a local W&B summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_MAP = {
    "test_accuracy": "classification/accuracy",
    "test_f1_macro": "classification/f1_macro",
    "test_meta_f1": "meta/f1",
    "test_take_rate": "meta/take_rate",
    "test_trade_signal_rate": "decision/trade_signal_rate",
    "test_directional_acted_accuracy": "decision/directional_acted_accuracy",
    "test_portfolio_total_return_pct": "portfolio/total_return_pct",
    "test_portfolio_sharpe_ratio_annualized": "portfolio/sharpe_ratio_annualized",
    "test_portfolio_max_drawdown_pct": "portfolio/max_drawdown_pct",
    "test_portfolio_annualized_return_pct": "portfolio/annualized_return_pct",
    "test_portfolio_average_trade_return_pct": "portfolio/average_trade_return_pct",
    "test_portfolio_n_executed_trades": "portfolio/n_executed_trades",
    "test_paper_net_return_total_pct": "paper/executed_trade_net_return_total_pct",
    "test_paper_sharpe_ratio_annualized": "paper/sharpe_ratio_annualized",
}


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def extract_rows(summary: dict[str, Any], *, folds: int) -> list[dict[str, float | int | None]]:
    """Return one result row per fold from a local W&B summary."""
    rows: list[dict[str, float | int | None]] = []
    for fold in range(int(folds)):
        fold_tag = f"fold{fold:02d}"
        row: dict[str, float | int | None] = {"fold": fold}
        for out_name, metric_name in METRIC_MAP.items():
            key = f"{fold_tag}/best/test/{metric_name}"
            row[out_name] = _numeric_or_none(summary.get(key))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-fold CSV results from local W&B summary JSON.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to wandb-summary.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to extract.")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    rows = extract_rows(summary, folds=args.folds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
