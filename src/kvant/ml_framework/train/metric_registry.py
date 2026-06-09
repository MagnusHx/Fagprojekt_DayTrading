from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class MetricSpec:
    """Describe one metric in the compact evaluation contract."""

    name: str
    location: str
    interpretation: str
    applicability: str
    layer: str
    label_space: str
    legacy: bool = False
    primary_debug_metric: bool = True
    notes: str = ""


def _spec(name: str, layer: str, interpretation: str, label_space: str = "run") -> MetricSpec:
    return MetricSpec(
        name=name,
        location="evaluator",
        interpretation=interpretation,
        applicability="both" if layer in {"training", "classification"} else "event_outcome",
        layer=layer,
        label_space=label_space,
    )


METRIC_SPECS: tuple[MetricSpec, ...] = (
    _spec("train/training/loss", "training", "Mean optimization loss over training minibatches.", "side"),
    _spec("val/training/loss", "training", "Mean validation loss.", "side"),
    _spec("{split}/classification/accuracy", "classification", "Primary side-model accuracy.", "side"),
    _spec("{split}/classification/f1_macro", "classification", "Macro F1 for balanced side-model quality.", "side"),
    _spec("{split}/meta/precision", "meta", "Precision of accepted TAKE decisions.", "meta"),
    _spec("{split}/meta/recall", "meta", "Recall of profitable TAKE opportunities.", "meta"),
    _spec("{split}/meta/f1", "meta", "Balanced TAKE/PASS quality and checkpoint metric.", "meta"),
    _spec("{split}/meta/take_rate", "meta", "Share of proposals accepted by the meta-model.", "meta"),
    _spec(
        "{split}/decision/trade_signal_rate", "decision", "Share of predictions converted into trade signals.", "trade"
    ),
    _spec(
        "{split}/decision/directional_acted_accuracy",
        "decision",
        "Directional accuracy for acted trades with actionable truth.",
        "trade",
    ),
    _spec(
        "{split}/decision/acted_on_exit_truth_pct",
        "decision",
        "Share of EXIT truths incorrectly converted into trades.",
        "trade",
    ),
    _spec(
        "{split}/execution/n_trade_signals_raw",
        "execution",
        "Raw trade signals before simulation constraints.",
        "trade",
    ),
    _spec("{split}/paper/n_executed_trades", "paper", "Executed trades in the detailed paper simulation.", "trade"),
    _spec(
        "{split}/paper/executed_trade_net_return_avg_pct",
        "paper",
        "Average net return per detailed paper trade.",
        "trade",
    ),
    _spec(
        "{split}/paper/executed_trade_net_return_total_pct",
        "paper",
        "Total net return across detailed paper trades.",
        "trade",
    ),
    _spec(
        "{split}/paper/transaction_cost_total_pct", "paper", "Total paper-simulation transaction-cost drag.", "trade"
    ),
    _spec("{split}/paper/sharpe_ratio_annualized", "paper", "Annualized paper-simulation Sharpe ratio.", "run"),
    _spec("{split}/paper/max_drawdown_pct", "paper", "Worst paper-simulation drawdown.", "run"),
    _spec("{split}/portfolio/total_return_pct", "portfolio", "Compounded budget-constrained account return."),
    _spec(
        "{split}/portfolio/cumulative_annual_profit_pct",
        "portfolio",
        "Sum of each calendar year's budget-constrained account return.",
    ),
    _spec("{split}/portfolio/annualized_return_pct", "portfolio", "Annualized budget-constrained account return."),
    _spec("{split}/portfolio/sharpe_ratio_annualized", "portfolio", "Annualized budget-constrained Sharpe ratio."),
    _spec("{split}/portfolio/max_drawdown_pct", "portfolio", "Worst budget-constrained account drawdown."),
    _spec(
        "{split}/portfolio/average_trade_return_pct",
        "portfolio",
        "Average return per executed portfolio trade.",
        "trade",
    ),
    _spec("{split}/portfolio/average_exposure_pct", "portfolio", "Average account exposure."),
    _spec(
        "{split}/portfolio/transaction_cost_total", "portfolio", "Total portfolio transaction cost in account currency."
    ),
    _spec("{split}/portfolio/n_executed_trades", "portfolio", "Trades executed after account constraints.", "trade"),
    _spec("{split}/portfolio/n_skipped_budget", "portfolio", "Trades skipped by account constraints.", "trade"),
    _spec(
        "perf/confusion_matrix_normalized/{split}",
        "artifact",
        "Final best-model normalized confusion matrix.",
        "artifact",
    ),
    _spec(
        "perf/profit_curve_over_trades/{split}",
        "artifact",
        "Final best-model cumulative paper-profit curve.",
        "artifact",
    ),
    _spec("perf/portfolio_equity_curve/{split}", "artifact", "Final best-model account equity curve.", "artifact"),
    _spec("perf/per_ticker_table", "artifact", "Final best-model per-ticker diagnostics.", "artifact"),
)


DEBUG_DASHBOARD_CONTRACT: tuple[tuple[str, str, str], ...] = tuple(
    (spec.layer.replace("_", " ").title(), spec.name, spec.interpretation) for spec in METRIC_SPECS
)


def metric_inventory_rows() -> List[dict]:
    """Return the compact metric registry as table rows."""
    return [
        {
            "name": spec.name,
            "location": spec.location,
            "interpretation": spec.interpretation,
            "applicability": spec.applicability,
            "layer": spec.layer,
            "label_space": spec.label_space,
            "legacy": int(spec.legacy),
            "primary_debug_metric": int(spec.primary_debug_metric),
            "notes": spec.notes,
        }
        for spec in METRIC_SPECS
    ]


def dashboard_contract_rows() -> List[dict]:
    """Return the compact dashboard contract as table rows."""
    return [
        {"section": section, "metric": metric, "notes": notes} for section, metric, notes in DEBUG_DASHBOARD_CONTRACT
    ]


def metric_names_for_layer(layer: str) -> Iterable[str]:
    """Return registered metric names for one category."""
    return (spec.name for spec in METRIC_SPECS if spec.layer == layer)
