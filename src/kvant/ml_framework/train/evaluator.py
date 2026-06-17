from __future__ import annotations

import time
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
    DEFAULT_META_FEATURES,
    LogisticMetaLabeler,
    fixed_size_trade_decisions,
    meta_targets_from_predictions,
    normalize_meta_features,
    sized_trade_decisions,
)
from .predict import predict
from .portfolio_simulator import compute_portfolio_metrics
from .trading_metrics import compute_action_profit_stats, compute_profit_curve_over_trades


_PAPER_METRICS_TO_KEEP = {
    "paper/executed_trade_net_return_avg_pct",
    "paper/executed_trade_net_return_total_pct",
    "paper/transaction_cost_total_pct",
    "paper/sharpe_ratio_annualized",
    "paper/max_drawdown_pct",
    "paper/n_executed_trades",
}


def _class_distribution_metrics(
    values: np.ndarray,
    *,
    labels: tuple[int, ...],
    prefix: str,
) -> Dict[str, float | int]:
    """Return count and percentage metrics for class labels."""
    arr = np.asarray(values, dtype=np.int64)
    n = int(len(arr))
    metrics: Dict[str, float | int] = {f"{prefix}/n": n}
    for label in labels:
        count = int(np.sum(arr == int(label)))
        metrics[f"{prefix}/class_{int(label)}_count"] = count
        metrics[f"{prefix}/class_{int(label)}_pct"] = float(count / n) if n else 0.0
    return metrics


@dataclass(frozen=True)
class EvalConfig:
    compute_per_ticker_accuracy: bool = True
    compute_profit_stats: bool = True
    compute_paper_trading_metrics: bool = True
    compute_portfolio_metrics: bool = True
    initial_portfolio: float = 1.0
    portfolio_initial_cash: float = 10_000.0
    portfolio_max_position_fraction: float = 0.02
    portfolio_max_total_exposure: float = 1.0
    portfolio_max_positions: int = 10
    transaction_cost: float = 0.0
    risk_free_rate: float = 0.0314
    days_per_year: float = 365.0
    backtest_width_minutes: int = 0
    backtest_barrier_height: float = 0.0
    labels: tuple[int, ...] = (0, 1)
    meta_model: str = "logreg"
    meta_features: tuple[str, ...] = DEFAULT_META_FEATURES
    meta_random_state: int = 1337
    meta_accept_threshold: float = 0.5
    use_meta_selection: bool = True
    bet_sizing: str = "fixed"
    fixed_bet_size: float = 1.0
    primary_confidence_threshold: float = 0.0
    kelly_fraction: float = 0.25
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
        if self.cfg.bet_sizing not in {"fixed", "kelly"}:
            raise RuntimeError(f"Unsupported bet_sizing={self.cfg.bet_sizing!r}.")
        if not (0.0 <= float(self.cfg.meta_accept_threshold) <= 1.0):
            raise RuntimeError("meta_accept_threshold must be between 0 and 1.")
        if not (0.0 <= float(self.cfg.fixed_bet_size) <= 1.0):
            raise RuntimeError("fixed_bet_size must be between 0 and 1.")
        if not (0.0 <= float(self.cfg.primary_confidence_threshold) <= 1.0):
            raise RuntimeError("primary_confidence_threshold must be between 0 and 1.")
        if float(self.cfg.kelly_fraction) < 0.0:
            raise RuntimeError("kelly_fraction must be non-negative.")
        if float(self.cfg.kelly_payoff_ratio) <= 0.0:
            raise RuntimeError("kelly_payoff_ratio must be positive.")
        if float(self.cfg.portfolio_initial_cash) <= 0.0:
            raise RuntimeError("portfolio_initial_cash must be positive.")
        if not (0.0 < float(self.cfg.portfolio_max_position_fraction) <= 1.0):
            raise RuntimeError("portfolio_max_position_fraction must be in (0, 1].")
        if float(self.cfg.portfolio_max_total_exposure) <= 0.0:
            raise RuntimeError("portfolio_max_total_exposure must be positive.")
        if int(self.cfg.portfolio_max_positions) <= 0:
            raise RuntimeError("portfolio_max_positions must be positive.")
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
        started_at = time.time()
        print(f"eval: fitting meta model on {len(pred_out['tid'])} train predictions...", flush=True)
        meta_model = LogisticMetaLabeler(
            feature_tokens=self.cfg.meta_features,
            random_state=int(self.cfg.meta_random_state),
        )
        meta_model.fit(pred_out=pred_out, store=self.store)
        print(f"eval: fitted meta model in {time.time() - started_at:.1f}s", flush=True)
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
        detailed: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Optional[Dict[str, Any]]]:
        side_true = np.asarray(pred_out["y_true"], dtype=np.int64)
        side_pred = np.asarray(pred_out["y_pred"], dtype=np.int64)
        tid = np.asarray(pred_out["tid"], dtype=np.int64)
        tpos = np.asarray(pred_out["tpos"], dtype=np.int64)
        valid_side_mask = side_true >= 0

        event_true = self._event_labels_for_pred_out(pred_out)
        meta_true = meta_targets_from_predictions(pred_out=pred_out, store=self.store)
        meta_pred = (np.asarray(take_proba, dtype=np.float64) >= float(self.cfg.meta_accept_threshold)).astype(np.int64)
        accept_threshold = float(self.cfg.meta_accept_threshold) if self.cfg.use_meta_selection else 0.0
        if self.cfg.bet_sizing == "kelly":
            y_trade, bet_size, _signed_bet_size = sized_trade_decisions(
                side_pred=side_pred,
                take_proba=take_proba,
                accept_threshold=accept_threshold,
                payoff_ratio=float(self.cfg.kelly_payoff_ratio),
                fraction=float(self.cfg.kelly_fraction),
            )
        else:
            y_trade, bet_size, _signed_bet_size = fixed_size_trade_decisions(
                side_pred=side_pred,
                take_proba=take_proba,
                accept_threshold=accept_threshold,
                bet_size=float(self.cfg.fixed_bet_size),
            )
        if float(self.cfg.primary_confidence_threshold) > 0.0:
            confidence = np.asarray(pred_out["y_pred_confidence"], dtype=np.float64)
            if len(confidence) != len(y_trade):
                raise RuntimeError(
                    "Prediction confidence length does not match trade decisions: "
                    f"{len(confidence)} vs {len(y_trade)}."
                )
            low_confidence_mask = confidence < float(self.cfg.primary_confidence_threshold)
            y_trade[low_confidence_mask] = LABEL_EXIT
            bet_size[low_confidence_mask] = 0.0

        acted_mask = np.isin(y_trade, ACTED_LABELS) & (bet_size > 0.0)
        actionable_truth_mask = np.isin(event_true, ACTED_LABELS)
        directional_acted_mask = acted_mask & actionable_truth_mask
        metrics: Dict[str, Any] = {}
        per_ticker_rows: List[Dict[str, Any]] = []

        cls = classification_metrics(
            side_true[valid_side_mask],
            side_pred[valid_side_mask],
            labels=self.cfg.labels,
        )
        metrics[f"{split}/classification/accuracy"] = float(cls["accuracy"])
        metrics[f"{split}/classification/f1_macro"] = float(cls.get("f1_macro", 0.0))
        metrics[f"{split}/decision/primary_confidence_threshold"] = float(self.cfg.primary_confidence_threshold)
        metrics.update(
            _class_distribution_metrics(
                side_true[valid_side_mask],
                labels=self.cfg.labels,
                prefix=f"{split}/distribution/true_side",
            )
        )
        metrics.update(
            _class_distribution_metrics(
                side_pred[valid_side_mask],
                labels=self.cfg.labels,
                prefix=f"{split}/distribution/pred_side",
            )
        )
        metrics.update(
            _class_distribution_metrics(
                y_trade,
                labels=tuple(sorted(set(int(label) for label in ACTED_LABELS) | {int(LABEL_EXIT)})),
                prefix=f"{split}/distribution/trade_signal",
            )
        )

        meta_precision, meta_recall, meta_f1, _ = precision_recall_fscore_support(
            meta_true,
            meta_pred,
            average="binary",
            pos_label=META_LABEL_TAKE,
            zero_division=0,
        )
        meta_metrics = {
            "precision": float(meta_precision),
            "recall": float(meta_recall),
            "f1": float(meta_f1),
            "take_rate": float(np.mean(meta_pred == META_LABEL_TAKE)) if len(meta_pred) else 0.0,
        }
        for key, value in meta_metrics.items():
            metrics[f"{split}/meta/{key}"] = value

        decision_metrics = {
            "trade_signal_rate": float(np.mean(acted_mask)) if len(y_trade) else 0.0,
            "directional_acted_accuracy": float(
                np.mean(event_true[directional_acted_mask] == y_trade[directional_acted_mask])
            )
            if np.any(directional_acted_mask)
            else 0.0,
            "acted_on_exit_truth_pct": float(np.mean(acted_mask[event_true == LABEL_EXIT]) * 100.0)
            if np.any(event_true == LABEL_EXIT)
            else 0.0,
        }
        for key, value in decision_metrics.items():
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
        for true_idx, true_label in enumerate(self.cfg.labels):
            row_total = int(np.sum(cm[true_idx, :]))
            for pred_idx, pred_label in enumerate(self.cfg.labels):
                count = int(cm[true_idx, pred_idx])
                metrics[f"{split}/confusion/true{int(true_label)}_pred{int(pred_label)}_count"] = count
                metrics[f"{split}/confusion/true{int(true_label)}_pred{int(pred_label)}_row_pct"] = (
                    float(count / row_total) if row_total else 0.0
                )

        per_tid_profit: Dict[int, Dict[str, Any]] = {}
        profit_curve: Optional[Dict[str, Any]] = None
        portfolio_curve: Optional[Dict[str, Any]] = None
        if detailed and self.cfg.compute_profit_stats:
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
                    if k in _PAPER_METRICS_TO_KEEP:
                        metrics[f"{split}/{k}"] = v

        if self.cfg.compute_portfolio_metrics:
            portfolio_result = compute_portfolio_metrics(
                y_true=event_true,
                y_pred=y_trade,
                tids=tid,
                tpos=tpos,
                bet_sizes=bet_size,
                simulator=self.paper_trade_simulator,
                initial_cash=float(self.cfg.portfolio_initial_cash),
                max_position_fraction=float(self.cfg.portfolio_max_position_fraction),
                max_total_exposure=float(self.cfg.portfolio_max_total_exposure),
                max_positions=int(self.cfg.portfolio_max_positions),
                risk_free_rate=float(self.cfg.risk_free_rate),
                days_per_year=float(self.cfg.days_per_year),
            )
            for k, v in portfolio_result.metrics.items():
                metrics[f"{split}/{k}"] = v
            if detailed:
                portfolio_curve = {
                    "split": split,
                    "epoch": int(step) if step is not None else None,
                    **portfolio_result.equity_curve,
                }

        if detailed and self.cfg.compute_per_ticker_accuracy:
            for t in np.unique(tid):
                mask = tid == t
                n_t = int(mask.sum())
                valid_t_mask = mask & valid_side_mask
                acc_t = (
                    float(np.mean(side_true[valid_t_mask] == side_pred[valid_t_mask])) if np.any(valid_t_mask) else 0.0
                )

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

        if portfolio_curve is not None:
            metrics["_portfolio_curve"] = portfolio_curve
        return metrics, per_ticker_rows, cm, profit_curve

    def evaluate_split(
        self,
        split: str,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        step: Optional[int] = None,
        detailed: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Optional[Dict[str, Any]]]:
        pred_out = predict(model, loader, self.device)
        if self.cfg.use_meta_selection:
            meta_model = self._fit_meta_model(pred_out)
            take_proba = meta_model.predict_take_proba(pred_out=pred_out, store=self.store)
        else:
            take_proba = np.ones(len(pred_out["tid"]), dtype=np.float64)
        return self._evaluate_pred_out(split, pred_out, step=step, take_proba=take_proba, detailed=detailed)

    def evaluate_all(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, Optional[DataLoader]],
        *,
        step: Optional[int] = None,
        metric_splits: tuple[str, ...] = ("val",),
        detailed: bool = False,
    ) -> Dict[str, Any]:
        if self.cfg.use_meta_selection and (
            "train" not in loaders or loaders["train"] is None or len(loaders["train"].dataset) == 0
        ):
            raise RuntimeError("Meta-label evaluation requires a non-empty train loader for fitting the meta model.")

        pred_out_by_split: Dict[str, Dict[str, Any]] = {}
        required_splits = {"train", *metric_splits}
        for split, loader in loaders.items():
            if split not in required_splits:
                continue
            if loader is None or len(loader.dataset) == 0:
                continue
            started_at = time.time()
            print(f"eval: predicting {split} split ({len(loader.dataset)} rows)...", flush=True)
            pred_out_by_split[split] = predict(model, loader, self.device)
            print(f"eval: predicted {split} split in {time.time() - started_at:.1f}s", flush=True)

        meta_model = self._fit_meta_model(pred_out_by_split["train"]) if self.cfg.use_meta_selection else None

        all_metrics: Dict[str, Any] = {}
        confusion_counts: Dict[str, np.ndarray] = {}
        profit_curves: List[Dict[str, Any]] = []
        portfolio_curves: List[Dict[str, Any]] = []
        rows_out: List[Dict[str, Any]] = []

        for split in metric_splits:
            pred_out = pred_out_by_split.get(split)
            if pred_out is None:
                continue
            started_at = time.time()
            print(f"eval: scoring {split} split ({len(pred_out['tid'])} rows)...", flush=True)
            take_proba = (
                meta_model.predict_take_proba(pred_out=pred_out, store=self.store)
                if meta_model is not None
                else np.ones(len(pred_out["tid"]), dtype=np.float64)
            )
            metrics, rows, cm, profit_curve = self._evaluate_pred_out(
                split,
                pred_out,
                step=step,
                take_proba=take_proba,
                detailed=detailed,
            )
            portfolio_curve = metrics.pop("_portfolio_curve", None)
            all_metrics.update(metrics)
            rows_out.extend(rows)
            if detailed:
                confusion_counts[split] = cm
            if profit_curve is not None:
                profit_curves.append(profit_curve)
            if portfolio_curve is not None:
                portfolio_curves.append(portfolio_curve)
            print(f"eval: scored {split} split in {time.time() - started_at:.1f}s", flush=True)

        if detailed:
            all_metrics["_per_ticker_rows"] = rows_out
            all_metrics["_confusion_counts"] = confusion_counts
            all_metrics["_profit_curves"] = profit_curves
            all_metrics["_portfolio_curves"] = portfolio_curves
            all_metrics["_confusion_class_names"] = ["Down barrier hit (y=0)", "Up barrier hit (y=1)"]
        return all_metrics
