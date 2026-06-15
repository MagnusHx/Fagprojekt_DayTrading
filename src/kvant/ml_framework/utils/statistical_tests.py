"""Statistical utilities for confidence intervals and hypothesis testing across folds."""

from typing import Optional, Tuple
import numpy as np
from scipy import stats


def calculate_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a metric across folds using t-distribution.

    Args:
        values: Array of metric values (one per fold)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower, upper) bounds of confidence interval
    """
    values = np.array(values)
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float(np.mean(values))

    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of mean
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

    return float(mean - margin), float(mean + margin)


def paired_ttest(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided"
) -> dict:
    """
    Paired t-test comparing two sets of metrics across folds.

    Use this to test if model A significantly outperforms model B on the same folds.

    Args:
        values_a: Metric values for model A (one per fold)
        values_b: Metric values for model B (one per fold)
        alternative: "two-sided", "less", or "greater"

    Returns:
        dict with keys: t_statistic, p_value, mean_diff, ci_lower, ci_upper, significant
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError(f"Unequal fold counts: {len(values_a)} vs {len(values_b)}")

    differences = values_a - values_b
    t_stat, p_value = stats.ttest_1samp(differences, 0, alternative=alternative)

    mean_diff = float(np.mean(differences))
    ci_lower, ci_upper = calculate_ci(differences, confidence=0.95)

    significant = p_value < 0.05

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
    }


def format_ci(mean: float, ci_lower: float, ci_upper: float, decimals: int = 4) -> str:
    """Format a metric with confidence interval for display."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]"


def format_ttest_result(result: dict, metric_name: str = "") -> str:
    """Format paired t-test result for display."""
    sig_marker = "***" if result["significant"] else ""
    lines = [
        f"{metric_name} paired t-test{sig_marker}",
        f"  mean difference: {result['mean_diff']:.4f}",
        f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]",
        f"  t-statistic: {result['t_statistic']:.4f}",
        f"  p-value: {result['p_value']:.6f}",
    ]
    return "\n".join(lines)
