from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from kvant.ml_prepare_data.data_loading import PreparedStore
from .backtest import BacktestTradeSimulator, compute_paper_trading_metrics
from .classification_metrics import classification_metrics
from .predict import predict
from .trading_metrics import (
    apply_trade_confidence_threshold,
    compute_action_profit_stats,
    compute_profit_curve_over_trades,
)


@dataclass(frozen=True)
class EvalConfig:
    compute_per_ticker_accuracy: bool = True
    compute_profit_stats: bool = True
    compute_paper_trading_metrics: bool = True
    initial_portfolio: float = 1.0
    # Realistic per-side default: fee 0.0004 + half-spread 0.0003 + slippage 0.0003 = 0.001.
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.0314
    days_per_year: float = 365.0
    trade_confidence_threshold: float = 0.0
    backtest_width_minutes: int = 0
    backtest_barrier_height: float = 0.0
    labels: tuple[int, ...] = (0, 1, 2)


class ExperimentEvaluator:
    def __init__(
        self,
        *,
        store: PreparedStore,
        device: torch.device,
        cfg: EvalConfig = EvalConfig(),
        logger: Optional[Any] = None,
    ):
        self.store = store
        self.device = device
        self.cfg = cfg
        self.logger = logger
        if self.cfg.compute_paper_trading_metrics:
            if int(self.cfg.backtest_width_minutes) <= 0:
                raise RuntimeError("Paper trading metrics require a positive backtest_width_minutes setting.")
            if float(self.cfg.backtest_barrier_height) <= 0.0:
                raise RuntimeError("Paper trading metrics require a positive backtest_barrier_height setting.")
        self.paper_trade_simulator = BacktestTradeSimulator(
            market_data_store=self.store,
            width_minutes=int(self.cfg.backtest_width_minutes),
            barrier_height=float(self.cfg.backtest_barrier_height),
            transaction_cost=float(self.cfg.transaction_cost),
        )

    def evaluate_split(
        self,
        split: str,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        step: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Optional[Dict[str, Any]]]:
        pred_out = predict(model, loader, self.device)
        y_true = pred_out["y_true"]
        y_pred = pred_out["y_pred"]
        y_pred_confidence = pred_out["y_pred_confidence"].astype(np.float64, copy=False)
        tid = pred_out["tid"].astype(np.int64, copy=False)
        tpos = pred_out["tpos"].astype(np.int64, copy=False)
        y_trade = apply_trade_confidence_threshold(
            y_pred=y_pred,
            y_pred_confidence=y_pred_confidence,
            trade_confidence_threshold=self.cfg.trade_confidence_threshold,
        )

        metrics: Dict[str, Any] = {}
        per_ticker_rows: List[Dict[str, Any]] = []

        # split-level scalars
        cls = classification_metrics(y_true, y_pred)
        for k, v in cls.items():
            metrics[f"{split}/{k}"] = v
        metrics[f"{split}/prediction_confidence_mean"] = float(np.mean(y_pred_confidence)) if len(y_pred_confidence) else 0.0
        metrics[f"{split}/prediction_confidence_median"] = (
            float(np.median(y_pred_confidence)) if len(y_pred_confidence) else 0.0
        )
        trade_signal_mask = np.isin(y_trade, (0, 2))
        metrics[f"{split}/trade_confidence_threshold"] = float(self.cfg.trade_confidence_threshold)
        metrics[f"{split}/trade_signal_count"] = int(np.sum(trade_signal_mask))
        metrics[f"{split}/trade_signal_rate"] = float(np.mean(trade_signal_mask)) if len(y_trade) else 0.0
        metrics[f"{split}/high_confidence_trade_signal_count"] = int(np.sum(trade_signal_mask))

        # confusion counts for heatmap
        cm = confusion_matrix(y_true, y_pred, labels=list(self.cfg.labels)).astype(np.int64, copy=False)

        # profit stats need metadata
        per_tid_profit: Dict[int, Dict[str, Any]] = {}
        profit_curve: Optional[Dict[str, Any]] = None
        if self.cfg.compute_profit_stats:
            index = np.stack([tid, tpos], axis=1).astype(np.int32, copy=False)
            metas = self.store.metadata_for_index(index)
            per_tid_profit = compute_action_profit_stats(
                y_pred=y_trade,
                metas=metas,
                tids=tid,
                transaction_cost=self.cfg.transaction_cost,
            )
            profit_curve = {
                "split": split,
                "epoch": int(step) if step is not None else None,
            } | compute_profit_curve_over_trades(
                y_pred=y_trade,
                metas=metas,
                tids=tid,
                transaction_cost=self.cfg.transaction_cost,
            )
            if self.cfg.compute_paper_trading_metrics:
                paper_metrics = compute_paper_trading_metrics(
                    y_true=y_true,
                    y_pred=y_trade,
                    tids=tid,
                    tpos=tpos,
                    simulator=self.paper_trade_simulator,
                    initial_portfolio=self.cfg.initial_portfolio,
                    risk_free_rate=self.cfg.risk_free_rate,
                    days_per_year=self.cfg.days_per_year,
                )
                for k, v in paper_metrics.items():
                    metrics[f"{split}/{k}"] = v

        # per-ticker accuracy (+ profit stats columns)
        if self.cfg.compute_per_ticker_accuracy:
            for t in np.unique(tid):
                mask = tid == t
                n_t = int(mask.sum())
                acc_t = float((y_true[mask] == y_pred[mask]).mean()) if n_t > 0 else 0.0

                ticker = self.store.tickers_all[int(t)]
                p = per_tid_profit.get(int(t), {})

                per_ticker_rows.append(
                    {
                        "epoch": int(step) if step is not None else None,
                        "split": split,
                        "tid": int(t),
                        "ticker": str(ticker),
                        "acc": acc_t,
                        "n": n_t,
                        # buy-only
                        "buy_n_trades": int(p.get("buy/n_trades", 0)),
                        "buy_profit_avg_per_trade_pct": float(p.get("buy/profit_pct/avg_per_trade", float("nan"))),
                        "buy_profit_total_pct": float(p.get("buy/profit_pct/total", 0.0)),
                        # short-only
                        "short_n_trades": int(p.get("short/n_trades", 0)),
                        "short_profit_avg_per_trade_pct": float(p.get("short/profit_pct/avg_per_trade", float("nan"))),
                        "short_profit_total_pct": float(p.get("short/profit_pct/total", 0.0)),
                    }
                )

        return metrics, per_ticker_rows, cm, profit_curve

    def evaluate_all(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, Optional[DataLoader]],
        *,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        all_metrics: Dict[str, Any] = {}
        confusion_counts: Dict[str, np.ndarray] = {}
        profit_curves: List[Dict[str, Any]] = []

        rows_out: List[Dict[str, Any]] = []

        for split, loader in loaders.items():
            if loader is None or len(loader.dataset) == 0:
                continue

            m, rows, cm, profit_curve = self.evaluate_split(split, model, loader, step=step)
            all_metrics.update(m)
            rows_out.extend(rows)
            confusion_counts[split] = cm
            if profit_curve is not None:
                profit_curves.append(profit_curve)

        # special payloads for logger
        all_metrics["_per_ticker_rows"] = rows_out
        all_metrics["_confusion_counts"] = confusion_counts
        all_metrics["_profit_curves"] = profit_curves

        return all_metrics
