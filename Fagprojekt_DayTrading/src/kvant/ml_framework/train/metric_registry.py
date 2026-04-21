from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class MetricSpec:
    name: str
    location: str
    interpretation: str
    applicability: str  # three_class | binary | both
    layer: str  # learning | decision | execution | economics
    legacy: bool = False
    primary_debug_metric: bool = True
    notes: str = ""


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="{split}/accuracy",
        location="classification_metrics -> evaluator",
        interpretation="Overall classifier accuracy in model label space before abstention.",
        applicability="both",
        layer="learning",
        notes="Primary learning metric; use together with per-class metrics.",
    ),
    MetricSpec(
        name="{split}/f1_macro",
        location="classification_metrics -> evaluator",
        interpretation="Macro F1 in model label space before abstention.",
        applicability="both",
        layer="learning",
        notes="Primary learning metric when class balance differs by regime or fold.",
    ),
    MetricSpec(
        name="{split}/precision_class_{label}",
        location="classification_metrics -> evaluator",
        interpretation="Per-class precision in model label space before abstention.",
        applicability="both",
        layer="learning",
    ),
    MetricSpec(
        name="{split}/recall_class_{label}",
        location="classification_metrics -> evaluator",
        interpretation="Per-class recall in model label space before abstention.",
        applicability="both",
        layer="learning",
    ),
    MetricSpec(
        name="{split}/cls/*",
        location="evaluator aliases",
        interpretation="Stable debug namespace for all pre-abstention classifier metrics.",
        applicability="both",
        layer="learning",
        notes="Preferred namespace for debug dashboards.",
    ),
    MetricSpec(
        name="{split}/trade_direction_threshold",
        location="evaluator",
        interpretation="Directional abstention-band threshold used to convert probabilities into trade labels.",
        applicability="both",
        layer="decision",
    ),
    MetricSpec(
        name="{split}/trade_signal_rate",
        location="evaluator",
        interpretation="Fraction of samples converted into acted trade labels after thresholding.",
        applicability="both",
        layer="decision",
    ),
    MetricSpec(
        name="{split}/decision/abstained_prediction_rate_pct",
        location="evaluator + paper aliases",
        interpretation="Percentage of predictions converted into EXIT/no-trade after thresholding.",
        applicability="both",
        layer="decision",
    ),
    MetricSpec(
        name="{split}/decision/acted_prediction_accuracy",
        location="evaluator + paper aliases",
        interpretation="Accuracy restricted to rows where the strategy chose to act.",
        applicability="both",
        layer="decision",
    ),
    MetricSpec(
        name="{split}/decision/directional_acted_accuracy",
        location="evaluator + paper aliases",
        interpretation="Accuracy on acted rows whose true label is directional in canonical trade space.",
        applicability="both",
        layer="decision",
        notes="Preferred acted-only learning signal across label regimes.",
    ),
    MetricSpec(
        name="{split}/trade_action_probability_mean",
        location="evaluator",
        interpretation="Mean p(down)+p(up) before thresholding.",
        applicability="both",
        layer="decision",
        legacy=True,
        primary_debug_metric=False,
        notes="Structurally constant at 1.0 in binary directional runs; do not use in binary debug dashboards.",
    ),
    MetricSpec(
        name="{split}/decision/trade_action_probability_informative",
        location="evaluator",
        interpretation="Boolean flag indicating whether action-probability metrics are informative in the current label regime.",
        applicability="both",
        layer="decision",
        notes="0 for binary directional runs, 1 for three-class runs.",
    ),
    MetricSpec(
        name="{split}/execution/n_trade_signals_raw",
        location="evaluator + paper aliases",
        interpretation="Number of raw acted signals before overlap suppression.",
        applicability="both",
        layer="execution",
    ),
    MetricSpec(
        name="{split}/execution/n_trade_signals_skipped_overlap",
        location="backtest -> evaluator aliases",
        interpretation="Count of raw acted signals dropped because another trade on the same ticker was already open.",
        applicability="both",
        layer="execution",
    ),
    MetricSpec(
        name="{split}/paper/n_executed_trades",
        location="backtest",
        interpretation="Number of actually executed non-overlapping trades.",
        applicability="both",
        layer="execution",
    ),
    MetricSpec(
        name="{split}/paper/share_time_active_pct",
        location="backtest",
        interpretation="Share of backtest time during which at least one per-ticker trade was active.",
        applicability="both",
        layer="execution",
    ),
    MetricSpec(
        name="{split}/paper/executed_trade_hit_rate_pct",
        location="backtest",
        interpretation="Directional hit rate across executed trades only.",
        applicability="both",
        layer="execution",
    ),
    MetricSpec(
        name="{split}/paper/executed_trade_gross_return_avg_pct",
        location="backtest",
        interpretation="Mean gross return per executed trade, before transaction costs.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/executed_trade_net_return_avg_pct",
        location="backtest",
        interpretation="Mean net return per executed trade, after transaction costs.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/transaction_cost_total_pct",
        location="backtest",
        interpretation="Total transaction-cost drag across executed trades in percentage points.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/profitable_transactions_pct",
        location="backtest",
        interpretation="Percentage of executed trades with positive net return.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/annual_net_profit_loss_pct",
        location="backtest",
        interpretation="Annualized net portfolio return from executed trades.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/sharpe_ratio_annualized",
        location="backtest",
        interpretation="Annualized Sharpe ratio of the daily portfolio value series.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/max_drawdown_pct",
        location="backtest",
        interpretation="Maximum drawdown of the daily portfolio value series.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/long_hit_rate_pct",
        location="backtest",
        interpretation="Directional hit rate across executed long trades.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/short_hit_rate_pct",
        location="backtest",
        interpretation="Directional hit rate across executed short trades.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/long_net_return_avg_pct",
        location="backtest",
        interpretation="Mean net return of executed long trades.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/short_net_return_avg_pct",
        location="backtest",
        interpretation="Mean net return of executed short trades.",
        applicability="both",
        layer="economics",
    ),
    MetricSpec(
        name="{split}/paper/accuracy_all_predictions",
        location="backtest",
        interpretation="Legacy canonical-trade-label accuracy over all rows after abstention.",
        applicability="both",
        layer="economics",
        legacy=True,
        primary_debug_metric=False,
        notes="Not a primary learning metric; abstentions can make it look artificially poor, especially in binary directional runs.",
    ),
)


DEBUG_DASHBOARD_CONTRACT: tuple[tuple[str, str, str], ...] = (
    ("Learning", "{split}/cls/accuracy", "Primary learning metric before abstention."),
    ("Learning", "{split}/cls/f1_macro", "Primary balanced learning metric before abstention."),
    ("Learning", "{split}/perf/confusion_matrix_normalized/{split}", "Normalized confusion matrix by split."),
    ("Decision", "{split}/trade_direction_threshold", "Abstention-band threshold."),
    ("Decision", "{split}/decision/abstained_prediction_rate_pct", "How often the strategy abstains."),
    ("Decision", "{split}/decision/acted_prediction_accuracy", "Accuracy on acted rows."),
    ("Decision", "{split}/decision/directional_acted_accuracy", "Best acted-only accuracy metric across regimes."),
    ("Execution", "{split}/execution/n_trade_signals_raw", "Raw acted signals before overlap suppression."),
    ("Execution", "{split}/execution/n_trade_signals_skipped_overlap", "Signals dropped due to overlap."),
    ("Execution", "{split}/paper/n_executed_trades", "Executed trades after overlap suppression."),
    ("Execution", "{split}/paper/share_time_active_pct", "Share of backtest time active."),
    ("Economics", "{split}/paper/executed_trade_gross_return_avg_pct", "Gross edge before costs."),
    ("Economics", "{split}/paper/executed_trade_net_return_avg_pct", "Net edge after costs."),
    ("Economics", "{split}/paper/transaction_cost_total_pct", "Total cost drag."),
    ("Economics", "{split}/paper/profitable_transactions_pct", "Positive-trade percentage."),
    ("Economics", "{split}/paper/annual_net_profit_loss_pct", "Annualized portfolio return."),
    ("Economics", "{split}/paper/sharpe_ratio_annualized", "Risk-adjusted performance."),
    ("Economics", "{split}/paper/max_drawdown_pct", "Worst portfolio drawdown."),
    ("Direction Split", "{split}/paper/long_n_executed_trades", "Executed long trades."),
    ("Direction Split", "{split}/paper/short_n_executed_trades", "Executed short trades."),
    ("Direction Split", "{split}/paper/long_hit_rate_pct", "Long-side hit rate."),
    ("Direction Split", "{split}/paper/short_hit_rate_pct", "Short-side hit rate."),
    ("Direction Split", "{split}/paper/long_net_return_avg_pct", "Average long net return."),
    ("Direction Split", "{split}/paper/short_net_return_avg_pct", "Average short net return."),
)


def metric_inventory_rows() -> List[dict]:
    return [
        {
            "name": spec.name,
            "location": spec.location,
            "interpretation": spec.interpretation,
            "applicability": spec.applicability,
            "layer": spec.layer,
            "legacy": int(spec.legacy),
            "primary_debug_metric": int(spec.primary_debug_metric),
            "notes": spec.notes,
        }
        for spec in METRIC_SPECS
    ]


def dashboard_contract_rows() -> List[dict]:
    return [
        {
            "section": section,
            "metric": metric,
            "notes": notes,
        }
        for section, metric, notes in DEBUG_DASHBOARD_CONTRACT
    ]


def metric_names_for_layer(layer: str) -> Iterable[str]:
    return (spec.name for spec in METRIC_SPECS if spec.layer == layer)
