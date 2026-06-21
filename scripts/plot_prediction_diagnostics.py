from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLD_RE = re.compile(r"-mt(?P<threshold>[0-9pm]+)-fold(?P<fold>[0-9]+)_(?P<split>val|test)_predictions\.csv$")
TRADE_LABELS = {
    0: "Down",
    1: "Exit/pass",
    2: "Up",
}
TRADE_COLORS = {
    0: "#2f6fbb",
    1: "#8a8f98",
    2: "#d05a39",
}


def _threshold_token_to_float(token: str) -> float:
    """Convert a filename threshold token such as ``0p45`` to a float."""
    return float(token.replace("p", ".").replace("m", "-"))


def _load_predictions(input_dir: Path, *, split: str) -> pd.DataFrame:
    """Load and annotate per-sample prediction diagnostics for one split."""
    rows: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob(f"*_{split}_predictions.csv")):
        match = THRESHOLD_RE.search(path.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        df["threshold"] = _threshold_token_to_float(match.group("threshold"))
        df["fold"] = int(match.group("fold"))
        df["source_file"] = path.name
        rows.append(df)
    if not rows:
        raise SystemExit(f"No {split!r} prediction CSVs found in {input_dir}.")
    out = pd.concat(rows, ignore_index=True)
    for col in [
        "primary_logit_margin",
        "primary_proba_margin",
        "meta_take_proba",
        "pnl_fraction",
        "proposed_signed_return",
        "executed_signed_return",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["trade_signal_name"] = out["trade_signal"].map(TRADE_LABELS).fillna("Unknown")
    return out


def _sample_for_plot(df: pd.DataFrame, *, max_points_per_threshold: int, seed: int) -> pd.DataFrame:
    """Limit scatter density while preserving all threshold panels."""
    pieces = []
    for threshold, group in df.groupby("threshold", sort=True):
        if len(group) <= max_points_per_threshold:
            pieces.append(group)
        else:
            pieces.append(group.sample(n=max_points_per_threshold, random_state=seed + int(round(threshold * 100))))
    return pd.concat(pieces, ignore_index=True)


def _plot_logit_margin_vs_take_proba(df: pd.DataFrame, out_path: Path, *, max_points_per_threshold: int) -> None:
    """Plot primary logit margin against meta TAKE probability by threshold."""
    sampled = _sample_for_plot(df, max_points_per_threshold=max_points_per_threshold, seed=1337)
    thresholds = sorted(sampled["threshold"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, threshold in zip(axes_flat, thresholds):
        panel = sampled[sampled["threshold"] == threshold]
        for trade_signal, label in TRADE_LABELS.items():
            part = panel[panel["trade_signal"] == trade_signal]
            ax.scatter(
                part["primary_logit_margin"],
                part["meta_take_proba"],
                s=7,
                alpha=0.28,
                c=TRADE_COLORS[trade_signal],
                label=label,
                linewidths=0,
            )
        ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.9)
        ax.set_title(f"Meta threshold = {threshold:.2f}")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)

    for ax in axes[:, 0]:
        ax.set_ylabel("Meta TAKE probability")
    for ax in axes[-1, :]:
        ax.set_xlabel("Primary logit margin (up logit - down logit)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Meta-selection decisions over primary-model logit margin", y=0.98)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pnl_vs_take_proba(df: pd.DataFrame, out_path: Path, *, max_points_per_threshold: int) -> None:
    """Plot realized proposed-trade return against meta TAKE probability by predicted side."""
    sampled = _sample_for_plot(df, max_points_per_threshold=max_points_per_threshold, seed=2027)
    thresholds = sorted(sampled["threshold"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, threshold in zip(axes_flat, thresholds):
        panel = sampled[sampled["threshold"] == threshold]
        for trade_signal, label in ((0, "Predicted down"), (2, "Predicted up")):
            side_value = 0 if trade_signal == 0 else 1
            part = panel[panel["side_pred"] == side_value]
            ax.scatter(
                part["meta_take_proba"],
                part["proposed_signed_return"] * 100.0,
                s=7,
                alpha=0.25,
                c=TRADE_COLORS[trade_signal],
                label=label,
                linewidths=0,
            )
        ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
        ax.set_title(f"Meta threshold = {threshold:.2f}")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)

    for ax in axes[:, 0]:
        ax.set_ylabel("Realized proposed return (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Meta TAKE probability")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Do higher meta scores correspond to profitable proposed trades?", y=0.98)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_outlier_overlay(df: pd.DataFrame, out_path: Path, *, outlier_quantile: float) -> None:
    """Plot all test points with extreme realized-return observations highlighted."""
    threshold = 0.45 if 0.45 in set(df["threshold"]) else float(sorted(df["threshold"].unique())[0])
    panel = df[df["threshold"] == threshold].copy()
    cutoff = float(panel["proposed_signed_return"].abs().quantile(outlier_quantile))
    panel["is_outlier"] = panel["proposed_signed_return"].abs() >= cutoff

    fig, ax = plt.subplots(figsize=(9, 6))
    regular = panel[~panel["is_outlier"]]
    outliers = panel[panel["is_outlier"]]
    ax.scatter(
        regular["primary_logit_margin"],
        regular["meta_take_proba"],
        s=7,
        alpha=0.18,
        c=regular["trade_signal"].map(TRADE_COLORS),
        linewidths=0,
    )
    ax.scatter(
        outliers["primary_logit_margin"],
        outliers["meta_take_proba"],
        s=24,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.8,
        label=f"Top {(1.0 - outlier_quantile) * 100:.0f}% absolute realized returns",
    )
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    ax.set_title(f"Outlier overlay for threshold {threshold:.2f}")
    ax.set_xlabel("Primary logit margin (up logit - down logit)")
    ax.set_ylabel("Meta TAKE probability")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _summarize(df: pd.DataFrame, *, outlier_quantile: float) -> pd.DataFrame:
    """Create threshold-level diagnostics for the scatter plots."""
    rows: list[dict[str, float | int]] = []
    for threshold, group in df.groupby("threshold", sort=True):
        accepted = group[group["trade_signal"].isin([0, 2])]
        cutoff = float(group["proposed_signed_return"].abs().quantile(outlier_quantile))
        outliers = group[group["proposed_signed_return"].abs() >= cutoff]
        accepted_outliers = outliers[outliers["trade_signal"].isin([0, 2])]
        down_count = int(np.sum(accepted["trade_signal"] == 0))
        up_count = int(np.sum(accepted["trade_signal"] == 2))
        rows.append(
            {
                "threshold": float(threshold),
                "n": int(len(group)),
                "accepted_n": int(len(accepted)),
                "accepted_rate": float(len(accepted) / len(group)) if len(group) else 0.0,
                "accepted_down_pct": float(down_count / len(accepted)) if len(accepted) else 0.0,
                "accepted_up_pct": float(up_count / len(accepted)) if len(accepted) else 0.0,
                "corr_logit_margin_take_proba": float(
                    group[["primary_logit_margin", "meta_take_proba"]].corr().iloc[0, 1]
                ),
                "corr_take_proba_return": float(group[["meta_take_proba", "proposed_signed_return"]].corr().iloc[0, 1]),
                "outlier_abs_return_cutoff": cutoff,
                "outlier_n": int(len(outliers)),
                "outlier_accept_rate": float(len(accepted_outliers) / len(outliers)) if len(outliers) else 0.0,
                "outlier_executed_return_share": float(
                    accepted_outliers["executed_signed_return"].sum() / accepted["executed_signed_return"].sum()
                )
                if len(accepted) and accepted["executed_signed_return"].sum() != 0.0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot per-sample meta-selection diagnostics.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/prediction_diagnostics"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/prediction_diagnostics/plots"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-points-per-threshold", type=int, default=12000)
    parser.add_argument("--outlier-quantile", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    """Generate scatter plots and summary tables from prediction diagnostics."""
    args = parse_args()
    if not 0.0 < float(args.outlier_quantile) < 1.0:
        raise SystemExit("--outlier-quantile must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_predictions(args.input_dir, split=args.split)
    scatter_path = args.output_dir / f"{args.split}_logit_margin_vs_meta_take_proba.png"
    pnl_path = args.output_dir / f"{args.split}_meta_take_proba_vs_realized_return.png"
    outlier_path = args.output_dir / f"{args.split}_outlier_overlay.png"
    summary_path = args.output_dir / f"{args.split}_scatter_summary.csv"

    _plot_logit_margin_vs_take_proba(df, scatter_path, max_points_per_threshold=int(args.max_points_per_threshold))
    _plot_pnl_vs_take_proba(df, pnl_path, max_points_per_threshold=int(args.max_points_per_threshold))
    _plot_outlier_overlay(df, outlier_path, outlier_quantile=float(args.outlier_quantile))
    _summarize(df, outlier_quantile=float(args.outlier_quantile)).to_csv(summary_path, index=False)

    print(f"Wrote {scatter_path}")
    print(f"Wrote {pnl_path}")
    print(f"Wrote {outlier_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
