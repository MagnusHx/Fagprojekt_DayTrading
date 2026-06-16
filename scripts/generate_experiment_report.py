#!/usr/bin/env python3
"""Generate report-ready tables and figures from experiment result CSV files."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kvant.ml_framework.utils.statistical_tests import calculate_ci, paired_ttest


DEFAULT_METRICS = [
    "test_accuracy",
    "test_f1_macro",
    "test_trade_signal_rate",
    "test_directional_acted_accuracy",
    "test_portfolio_total_return_pct",
    "test_portfolio_sharpe_ratio_annualized",
    "test_portfolio_max_drawdown_pct",
    "test_portfolio_n_executed_trades",
]


def _parse_result_arg(value: str) -> tuple[str, Path]:
    """Parse NAME=PATH result arguments."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Result name cannot be empty.")
    return name, Path(path.strip())


def _expand_result_globs(patterns: list[str]) -> list[tuple[str, Path]]:
    """Expand result glob patterns into NAME=PATH pairs using file stems as names."""
    out: list[tuple[str, Path]] = []
    for pattern in patterns:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
        if not paths:
            raise FileNotFoundError(f"No result CSVs matched {pattern!r}.")
        out.extend((path.stem, path) for path in paths)
    return out


def _load_results(results: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    """Load result CSV files keyed by display name."""
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in results:
        if not path.exists():
            raise FileNotFoundError(f"Missing result CSV for {name}: {path}")
        df = pd.read_csv(path)
        if "fold" in df.columns:
            df = df.sort_values("fold").reset_index(drop=True)
        loaded[name] = df
    return loaded


def _metric_summary(df: pd.DataFrame, metric: str) -> dict[str, float] | None:
    """Summarize one metric for one run."""
    if metric not in df.columns:
        return None
    values = df[metric].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    ci_lower, ci_upper = calculate_ci(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_folds": float(len(values)),
    }


def build_summary_table(loaded: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    """Build a long summary table with mean, std, and confidence interval."""
    rows = []
    for run_name, df in loaded.items():
        for metric in metrics:
            summary = _metric_summary(df, metric)
            if summary is None:
                continue
            rows.append({"run": run_name, "metric": metric, **summary})
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    """Write a compact LaTeX table with mean and 95 percent confidence interval."""
    if summary.empty:
        return
    table = summary.copy()
    table["mean_95_ci"] = table.apply(
        lambda row: f"{row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]", axis=1
    )
    pivot = table.pivot(index="metric", columns="run", values="mean_95_ci")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pivot.to_latex(escape=True), encoding="utf-8")


def _plot_metric_bar(summary: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Plot one metric as bars with 95 percent confidence intervals."""
    rows = summary[summary["metric"] == metric].copy()
    if rows.empty:
        return
    rows = rows.sort_values("mean", ascending=False)
    yerr = np.vstack([rows["mean"] - rows["ci_lower"], rows["ci_upper"] - rows["mean"]])

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.85), 4.5), dpi=150)
    ax.bar(rows["run"], rows["mean"], yerr=yerr, capsize=4)
    ax.set_title(metric.replace("_", " "))
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = metric.replace("/", "_")
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def write_metric_plots(summary: pd.DataFrame, metrics: list[str], output_dir: Path) -> None:
    """Write metric comparison plots."""
    for metric in metrics:
        _plot_metric_bar(summary, metric, output_dir)


def _extract_grid_params(run_name: str) -> tuple[float, float] | None:
    """Extract CUSUM and triple-barrier percent values from an E2 grid run name."""
    match = re.search(r"tb(?P<tb>\d+(?:p\d+)?)-cusum(?P<cusum>\d+(?:p\d+)?)", run_name)
    if match is None:
        return None
    tb = float(match.group("tb").replace("p", "."))
    cusum = float(match.group("cusum").replace("p", "."))
    return cusum, tb


def write_grid_heatmap(summary: pd.DataFrame, *, metric: str, output_dir: Path) -> None:
    """Write a grid-search heatmap when grid run names are present."""
    rows = []
    for _, row in summary[summary["metric"] == metric].iterrows():
        params = _extract_grid_params(str(row["run"]))
        if params is None:
            continue
        cusum, tb = params
        rows.append({"cusum": cusum, "tb": tb, "mean": float(row["mean"])})
    if not rows:
        return

    heatmap_df = pd.DataFrame(rows)
    pivot = heatmap_df.pivot(index="cusum", columns="tb", values="mean").sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}%" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}%" for value in pivot.index])
    ax.set_xlabel("Triple-barrier height")
    ax.set_ylabel("CUSUM threshold")
    ax.set_title(metric.replace("_", " "))
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = pivot.iloc[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"grid_heatmap_{metric}"
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def write_pairwise_tests(
    loaded: dict[str, pd.DataFrame],
    *,
    comparisons: list[str],
    metrics: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Write paired t-test results for requested A=B comparisons."""
    rows = []
    for comparison in comparisons:
        if "=" not in comparison:
            raise ValueError(f"Expected comparison in NAME_A=NAME_B form, got {comparison!r}.")
        name_a, name_b = [part.strip() for part in comparison.split("=", 1)]
        if name_a not in loaded or name_b not in loaded:
            raise ValueError(f"Comparison references missing runs: {comparison!r}.")
        for metric in metrics:
            if metric not in loaded[name_a].columns or metric not in loaded[name_b].columns:
                continue
            values_a = loaded[name_a][metric].dropna().to_numpy(dtype=float)
            values_b = loaded[name_b][metric].dropna().to_numpy(dtype=float)
            if len(values_a) != len(values_b) or len(values_a) == 0:
                continue
            test = paired_ttest(values_a, values_b)
            rows.append({
                "comparison": f"{name_a} vs {name_b}",
                "metric": metric,
                "mean_diff": test["mean_diff"],
                "ci_lower": test["ci_lower"],
                "ci_upper": test["ci_upper"],
                "p_value": test["p_value"],
                "significant_5pct": bool(test["significant"]),
            })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """Run the report generation CLI."""
    parser = argparse.ArgumentParser(description="Generate report-ready tables and figures.")
    parser.add_argument("--result", action="append", type=_parse_result_arg, default=[])
    parser.add_argument(
        "--results-glob",
        action="append",
        default=[],
        help="Glob of result CSV files. File stems are used as run names.",
    )
    parser.add_argument("--metric", action="append", dest="metrics", default=None)
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--grid-heatmap-metric", default="val_f1_macro")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()

    metrics = args.metrics or DEFAULT_METRICS
    result_items = list(args.result) + _expand_result_globs(list(args.results_glob))
    if not result_items:
        raise SystemExit("Pass at least one --result NAME=PATH or --results-glob PATTERN.")
    loaded = _load_results(result_items)
    summary = build_summary_table(loaded, metrics)

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(table_dir / "summary_metrics.csv", index=False)
    write_latex_table(summary, table_dir / "summary_metrics.tex")
    write_metric_plots(summary, metrics, figure_dir)
    write_grid_heatmap(summary, metric=str(args.grid_heatmap_metric), output_dir=figure_dir)

    if args.comparison:
        tests = write_pairwise_tests(
            loaded,
            comparisons=list(args.comparison),
            metrics=metrics,
            output_path=table_dir / "pairwise_tests.csv",
        )
        if not tests.empty:
            (table_dir / "pairwise_tests.tex").write_text(tests.to_latex(index=False, escape=True), encoding="utf-8")

    print(f"Wrote report tables to {table_dir}")
    print(f"Wrote report figures to {figure_dir}")


if __name__ == "__main__":
    main()
