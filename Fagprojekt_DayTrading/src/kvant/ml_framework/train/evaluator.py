from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from kvant.labels import (
    ACTED_LABELS,
    LABEL_EXIT,
    META_LABEL_TAKE,
)
from kvant.ml_prepare_data.data_loading import PreparedStore

from .backtest import BacktestTradeSimulator, compute_paper_trading_metrics
from .classification_metrics import classification_metrics
from .decision_policy import (
    LogisticMetaLabeler,
    meta_targets_from_predictions,
    normalize_meta_features,
    sized_trade_decisions,
)
from .predict import predict
from .trading_metrics import compute_action_profit_stats, compute_profit_curve_over_trades


@dataclass(frozen=True)
class EvalConfig:
    compute_per_ticker_accuracy: bool = True
    compute_profit_stats: bool = True
    compute_paper_trading_metrics: bool = True
    initial_portfolio: float = 1.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.0314
    days_per_year: float = 365.0
    backtest_width_minutes: int = 0
    backtest_barrier_height: float = 0.0
    labels: tuple[int, ...] = (0, 1)
    meta_model: str = "logreg"
    meta_features: tuple[str, ...] = ("proba", "embedding")
    meta_random_state: int = 1337
    meta_accept_threshold: float = 0.5
    kelly_fraction: float = 1.0
    kelly_payoff_ratio: float = 1.0


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
        self.cfg = EvalConfig(
            **{
                **cfg.__dict__,
                "meta_features": normalize_meta_features(cfg.meta_features),
            }
        )
        self.logger = logger
        if self.cfg.meta_model != "logreg":
            raise RuntimeError(f"Unsupported meta_model={self.cfg.meta_model!r}.")
        if not (0.0 <= float(self.cfg.meta_accept_threshold) <= 1.0):
            raise RuntimeError("meta_accept_threshold must be between 0 and 1.")
        if float(self.cfg.kelly_fraction) < 0.0:
            raise RuntimeError("kelly_fraction must be non-negative.")
        if float(self.cfg.kelly_payoff_ratio) <= 0.0:
            raise RuntimeError("kelly_payoff_ratio must be positive.")
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

    def _fit_meta_model(self, pred_out: Dict[str, Any]) -> LogisticMetaLabeler:
        meta_model = LogisticMetaLabeler(
            feature_tokens=self.cfg.meta_features,
            random_state=int(self.cfg.meta_random_state),
        )
        meta_model.fit(pred_out=pred_out, store=self.store)
        return meta_model

    def _event_labels_for_pred_out(self, pred_out: Dict[str, Any]) -> np.ndarray:
        index = np.stack(
            [
                np.asarray(pred_out["tid"], dtype=np.int64),
                np.asarray(pred_out["tpos"], dtype=np.int64),
            ],
            axis=1,
        ).astype(np.int64, copy=False)
        return self.store.event_labels_for_index(index)

    def _evaluate_pred_out(
        self,
        split: str,
        pred_out: Dict[str, Any],
        *,
        step: Optional[int] = None,
        take_proba: np.ndarray,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Optional[Dict[str, Any]]]:
        side_true = np.asarray(pred_out["y_true"], dtype=np.int64)
        side_pred = np.asarray(pred_out["y_pred"], dtype=np.int64)
        y_pred_confidence = np.asarray(pred_out["y_pred_confidence"], dtype=np.float64)
        tid = np.asarray(pred_out["tid"], dtype=np.int64)
        tpos = np.asarray(pred_out["tpos"], dtype=np.int64)
        valid_side_mask = side_true >= 0

        event_true = self._event_labels_for_pred_out(pred_out)
        meta_true = meta_targets_from_predictions(pred_out=pred_out, store=self.store)
        meta_pred = (np.asarray(take_proba, dtype=np.float64) >= float(self.cfg.meta_accept_threshold)).astype(np.int64)
        y_trade, bet_size, _signed_bet_size = sized_trade_decisions(
            side_pred=side_pred,
            take_proba=take_proba,
            accept_threshold=float(self.cfg.meta_accept_threshold),
            payoff_ratio=float(self.cfg.kelly_payoff_ratio),
            fraction=float(self.cfg.kelly_fraction),
        )

        acted_mask = np.isin(y_trade, ACTED_LABELS) & (bet_size > 0.0)
        actionable_truth_mask = np.isin(event_true, ACTED_LABELS)
        directional_acted_mask = acted_mask & actionable_truth_mask
        abstained_mask = y_trade == LABEL_EXIT

        metrics: Dict[str, Any] = {}
        per_ticker_rows: List[Dict[str, Any]] = []

        cls = classification_metrics(
            side_true[valid_side_mask],
            side_pred[valid_side_mask],
            labels=self.cfg.labels,
        )
        for k, v in cls.items():
            metrics[f"{split}/{k}"] = v
            metrics[f"{split}/cls/{k}"] = v

        metrics[f"{split}/prediction_confidence_mean"] = (
            float(np.mean(y_pred_confidence[valid_side_mask])) if np.any(valid_side_mask) else 0.0
        )
        metrics[f"{split}/prediction_confidence_median"] = (
            float(np.median(y_pred_confidence[valid_side_mask])) if np.any(valid_side_mask) else 0.0
        )

        meta_precision, meta_recall, meta_f1, _ = precision_recall_fscore_support(
            meta_true,
            meta_pred,
            average="binary",
            pos_label=META_LABEL_TAKE,
            zero_division=0,
        )
        meta_metrics = {
            "accept_threshold": float(self.cfg.meta_accept_threshold),
            "precision": float(meta_precision),
            "recall": float(meta_recall),
            "f1": float(meta_f1),
            "accuracy": float(np.mean(meta_true == meta_pred)) if len(meta_true) else 0.0,
            "take_probability_mean": float(np.mean(take_proba)) if len(take_proba) else 0.0,
            "take_probability_median": float(np.median(take_proba)) if len(take_proba) else 0.0,
            "take_rate": float(np.mean(meta_pred == META_LABEL_TAKE)) if len(meta_pred) else 0.0,
            "truth_take_rate": float(np.mean(meta_true == META_LABEL_TAKE)) if len(meta_true) else 0.0,
        }
        for key, value in meta_metrics.items():
            metrics[f"{split}/meta/{key}"] = value

        decision_metrics = {
            "meta_accept_threshold": float(self.cfg.meta_accept_threshold),
            "trade_signal_count": int(np.sum(acted_mask)),
            "trade_signal_rate": float(np.mean(acted_mask)) if len(y_trade) else 0.0,
            "abstained_prediction_rate_pct": float(np.mean(abstained_mask) * 100.0) if len(y_trade) else 0.0,
            "acted_prediction_accuracy": float(np.mean(event_true[acted_mask] == y_trade[acted_mask]))
            if np.any(acted_mask)
            else 0.0,
            "directional_acted_accuracy": float(
                np.mean(event_true[directional_acted_mask] == y_trade[directional_acted_mask])
            )
            if np.any(directional_acted_mask)
            else 0.0,
            "actionable_truth_rate_pct": float(np.mean(actionable_truth_mask) * 100.0) if len(event_true) else 0.0,
            "abstain_on_actionable_truth_pct": float(np.mean(abstained_mask[actionable_truth_mask]) * 100.0)
            if np.any(actionable_truth_mask)
            else 0.0,
            "acted_on_exit_truth_pct": float(np.mean(acted_mask[event_true == LABEL_EXIT]) * 100.0)
            if np.any(event_true == LABEL_EXIT)
            else 0.0,
        }
        for key, value in decision_metrics.items():
            metrics[f"{split}/{key}"] = value
            metrics[f"{split}/decision/{key}"] = value

        execution_metrics = {
            "n_trade_signals_raw": int(np.sum(acted_mask)),
        }
        for key, value in execution_metrics.items():
            metrics[f"{split}/execution/{key}"] = value

        if np.any(valid_side_mask):
            cm = confusion_matrix(
                side_true[valid_side_mask],
                side_pred[valid_side_mask],
                labels=list(self.cfg.labels),
            ).astype(np.int64, copy=False)
        else:
            cm = np.zeros((len(self.cfg.labels), len(self.cfg.labels)), dtype=np.int64)

        per_tid_profit: Dict[int, Dict[str, Any]] = {}
        profit_curve: Optional[Dict[str, Any]] = None
        if self.cfg.compute_profit_stats:
            index = np.stack([tid, tpos], axis=1).astype(np.int32, copy=False)
            metas = self.store.metadata_for_index(index)
            per_tid_profit = compute_action_profit_stats(
                y_pred=y_trade,
                metas=metas,
                tids=tid,
                bet_sizes=bet_size,
                transaction_cost=self.cfg.transaction_cost,
            )
            profit_curve = {
                "split": split,
                "epoch": int(step) if step is not None else None,
            } | compute_profit_curve_over_trades(
                y_pred=y_trade,
                metas=metas,
                tids=tid,
                bet_sizes=bet_size,
                transaction_cost=self.cfg.transaction_cost,
            )
            if self.cfg.compute_paper_trading_metrics:
                paper_metrics = compute_paper_trading_metrics(
                    y_true=event_true,
                    y_pred=y_trade,
                    tids=tid,
                    tpos=tpos,
                    bet_sizes=bet_size,
                    simulator=self.paper_trade_simulator,
                    initial_portfolio=self.cfg.initial_portfolio,
                    risk_free_rate=self.cfg.risk_free_rate,
                    days_per_year=self.cfg.days_per_year,
                )
                for k, v in paper_metrics.items():
                    metrics[f"{split}/{k}"] = v
                    suffix = str(k).split("/", 1)[1] if "/" in str(k) else str(k)
                    if suffix in {
                        "acted_prediction_accuracy",
                        "directional_acted_accuracy",
                        "actionable_truth_rate_pct",
                        "abstained_prediction_rate_pct",
                        "abstain_on_actionable_truth_pct",
                        "acted_on_exit_truth_pct",
                    }:
                        metrics[f"{split}/decision/{suffix}"] = v
                    if suffix in {
                        "n_trade_signals_raw",
                        "n_trade_signals_skipped_overlap",
                        "n_executed_trades",
                        "share_time_active_pct",
                        "executed_trade_hit_rate_pct",
                    }:
                        metrics[f"{split}/execution/{suffix}"] = v

        if self.cfg.compute_per_ticker_accuracy:
            for t in np.unique(tid):
                mask = tid == t
                n_t = int(mask.sum())
                valid_t_mask = mask & valid_side_mask
                acc_t = float(np.mean(side_true[valid_t_mask] == side_pred[valid_t_mask])) if np.any(valid_t_mask) else 0.0

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
                        "buy_n_trades": int(p.get("buy/n_trades", 0)),
                        "buy_profit_avg_per_trade_pct": float(p.get("buy/profit_pct/avg_per_trade", float("nan"))),
                        "buy_profit_total_pct": float(p.get("buy/profit_pct/total", 0.0)),
                        "short_n_trades": int(p.get("short/n_trades", 0)),
                        "short_profit_avg_per_trade_pct": float(p.get("short/profit_pct/avg_per_trade", float("nan"))),
                        "short_profit_total_pct": float(p.get("short/profit_pct/total", 0.0)),
                    }
                )

        return metrics, per_ticker_rows, cm, profit_curve

    def evaluate_split(
        self,
        split: str,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        step: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Optional[Dict[str, Any]]]:
        pred_out = predict(model, loader, self.device)
        meta_model = self._fit_meta_model(pred_out)
        take_proba = meta_model.predict_take_proba(pred_out=pred_out, store=self.store)
        return self._evaluate_pred_out(split, pred_out, step=step, take_proba=take_proba)

    def evaluate_all(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, Optional[DataLoader]],
        *,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        if "train" not in loaders or loaders["train"] is None or len(loaders["train"].dataset) == 0:
            raise RuntimeError("Meta-label evaluation requires a non-empty train loader for fitting the meta model.")

        pred_out_by_split: Dict[str, Dict[str, Any]] = {}
        for split, loader in loaders.items():
            if loader is None or len(loader.dataset) == 0:
                continue
            pred_out_by_split[split] = predict(model, loader, self.device)

        meta_model = self._fit_meta_model(pred_out_by_split["train"])

        all_metrics: Dict[str, Any] = {}
        confusion_counts: Dict[str, np.ndarray] = {}
        profit_curves: List[Dict[str, Any]] = []
        rows_out: List[Dict[str, Any]] = []

        for split, pred_out in pred_out_by_split.items():
            take_proba = meta_model.predict_take_proba(pred_out=pred_out, store=self.store)
            metrics, rows, cm, profit_curve = self._evaluate_pred_out(split, pred_out, step=step, take_proba=take_proba)
            all_metrics.update(metrics)
            rows_out.extend(rows)
            confusion_counts[split] = cm
            if profit_curve is not None:
                profit_curves.append(profit_curve)

        all_metrics["_per_ticker_rows"] = rows_out
        all_metrics["_confusion_counts"] = confusion_counts
        all_metrics["_profit_curves"] = profit_curves
        all_metrics["_confusion_class_names"] = ["Down barrier hit (y=0)", "Up barrier hit (y=1)"]
        return all_metrics
