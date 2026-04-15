from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict | None]:
    out: list[dict | None] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _daily_sample_counts_from_timestamps(ts: np.ndarray) -> pd.Series:
    """
    ts: np.ndarray of datetime64[ns] (as saved by prepare_experiment)
    Returns: Series indexed by day (Timestamp, midnight) with counts per day.
    """
    if ts is None or len(ts) == 0:
        return pd.Series(dtype=np.int64)

    dt = pd.to_datetime(ts)  # tz-naive OK
    days = pd.Series(dt).dt.floor("D")
    counts = days.value_counts().sort_index()
    counts.index.name = "day"
    counts.name = "samples"
    return counts

def _save_hist_png(values: np.ndarray, out_path: Path, title: str, bins: int = 50) -> None:
    """
    Uses matplotlib if available. If not, writes histogram bins to JSON instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        # nothing to plot
        (out_path.with_suffix(".json")).write_text(json.dumps({"title": title, "empty": True}, indent=2))
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        # fallback: save histogram data
        hist, edges = np.histogram(values, bins=bins)
        payload = {
            "title": title,
            "bins": int(bins),
            "edges": edges.tolist(),
            "counts": hist.tolist(),
        }
        (out_path.with_suffix(".json")).write_text(json.dumps(payload, indent=2))
        return

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("samples per day")
    plt.ylabel("number of days")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def report_sampling_density(
    exp_dir: Path,
    *,
    tickers: Optional[List[str]] = None,
    bins: int = 50,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Reads prepared artifacts from exp_dir and produces:
      - per-ticker daily counts CSV
      - per-ticker histogram PNG of daily samples
      - global histogram PNG over all ticker-days
      - global histogram PNG over per-ticker mean samples/day

    Returns a DataFrame summary (also saved to exp_dir/sampling_report.csv).
    """
    tickers_root = exp_dir / "tickers"
    if tickers is None:
        tickers = sorted([p.name for p in tickers_root.iterdir() if p.is_dir()])

    out_report_dir = exp_dir / "reports"
    out_report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_daily_values = []
    per_ticker_means = []

    for t in tickers:
        tdir = tickers_root / t
        ts_path = tdir / "timestamps.npy"
        meta_path = tdir / "meta.json"
        if not ts_path.exists() or not meta_path.exists():
            continue

        ts = np.load(ts_path, mmap_mode="r")
        meta = _load_json(meta_path)

        n_raw = meta.get("n_rows_raw_full", None)  # requires you to include this in meta during prepare
        n_sampled = meta.get("n_rows_sampled_full", meta.get("n_rows_full", len(ts)))

        counts = _daily_sample_counts_from_timestamps(ts)
        n_days = int(counts.shape[0])
        mean_bpd = float(counts.mean()) if n_days else 0.0
        median_bpd = float(counts.median()) if n_days else 0.0

        all_daily_values.append(counts.to_numpy(dtype=float))
        per_ticker_means.append(mean_bpd)

        # Save per-day counts
        counts_csv = out_report_dir / f"{t}_samples_per_day.csv"
        counts.to_csv(counts_csv, header=True)

        # Per-ticker histogram
        _save_hist_png(
            counts.to_numpy(dtype=float),
            out_report_dir / f"{t}_samples_per_day_hist.png",
            title=f"{t}: samples/day distribution (n_days={n_days}, mean={mean_bpd:.2f})",
            bins=bins,
        )

        retention = None
        if isinstance(n_raw, (int, float)) and n_raw and n_sampled is not None:
            retention = float(n_sampled) / float(n_raw)

        rows.append({
            "ticker": t,
            "n_raw_full": n_raw,
            "n_sampled_full": int(n_sampled) if n_sampled is not None else None,
            "retention_ratio": retention,
            "n_days": n_days,
            "samples_per_day_mean": mean_bpd,
            "samples_per_day_median": median_bpd,
            "samples_per_day_min": float(counts.min()) if n_days else 0.0,
            "samples_per_day_max": float(counts.max()) if n_days else 0.0,
        })

    df = pd.DataFrame(rows).sort_values(["samples_per_day_mean", "ticker"], ascending=[False, True])
    df.to_csv(out_report_dir / "sampling_report.csv", index=False)

    # Global hist: all ticker-days combined
    if len(all_daily_values):
        all_vals = np.concatenate(all_daily_values, axis=0)
    else:
        all_vals = np.array([], dtype=float)

    _save_hist_png(
        all_vals,
        out_report_dir / "ALL_TICKERS_samples_per_day_hist.png",
        title=f"ALL TICKERS: samples/day distribution over ticker-days (n={len(all_vals)})",
        bins=bins,
    )

    # Global hist: per-ticker mean samples/day (one value per ticker)
    _save_hist_png(
        np.asarray(per_ticker_means, dtype=float),
        out_report_dir / "PER_TICKER_mean_samples_per_day_hist.png",
        title=f"PER TICKER: mean samples/day (n_tickers={len(per_ticker_means)})",
        bins=min(bins, max(10, len(per_ticker_means))),
    )

    if print_table and len(df):
        # Pretty minimal console table
        cols = ["ticker", "n_raw_full", "n_sampled_full", "retention_ratio", "n_days",
                "samples_per_day_mean", "samples_per_day_median"]
        print("\nSampling density report (top 50):")
        print(df[cols].head(50).to_string(index=False))

    return df


def report_sample_labeling(
    exp_dir: Path,
    *,
    tickers: Optional[List[str]] = None,
    max_tickers: int = 6,
    max_points_per_ticker: int = 1500,
) -> list[Path]:
    """
    Create sample labeling plots from prepared ticker artifacts.

    The plots are based on the saved sampled raw market data and corresponding labels.
    """
    tickers_root = exp_dir / "tickers"
    reports_dir = exp_dir / "reports" / "labeling_samples"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        preferred = exp_dir / "tickers_train.json"
        fallback = exp_dir / "tickers_all.json"
        tickers = json.loads((preferred if preferred.exists() else fallback).read_text())

    tickers = list(tickers)[: max(1, int(max_tickers))]
    out_paths: list[Path] = []

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        payload = {
            "tickers": tickers,
            "max_tickers": int(max_tickers),
            "max_points_per_ticker": int(max_points_per_ticker),
            "error": "matplotlib unavailable",
        }
        out_path = reports_dir / "labeling_samples.json"
        out_path.write_text(json.dumps(payload, indent=2))
        return [out_path]

    color_map = {-1: "black", 0: "red", 1: "blue", 2: "green"}
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="No label", markerfacecolor="black", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Down barrier", markerfacecolor="red", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Time exit", markerfacecolor="blue", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Up barrier", markerfacecolor="green", markersize=6),
    ]

    n_cols = min(2, len(tickers))
    n_rows = int(np.ceil(len(tickers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), dpi=140, squeeze=False)
    fig.suptitle("Sample ticker labeling", fontsize=14)

    for ax in axes.flat[len(tickers) :]:
        ax.axis("off")

    for plot_idx, ticker in enumerate(tickers):
        ax = axes.flat[plot_idx]
        tdir = tickers_root / ticker
        labels_path = tdir / "labels.npy"
        market_data_path = tdir / "market_data.npy"
        meta_path = tdir / "meta.json"
        label_metadata_path = tdir / "label_metadata.jsonl"
        timestamps_path = tdir / "timestamps.npy"

        if not (labels_path.exists() and market_data_path.exists() and meta_path.exists() and timestamps_path.exists()):
            ax.text(0.5, 0.5, f"Missing prepared artifacts for {ticker}", ha="center", va="center")
            ax.axis("off")
            continue

        labels = np.load(labels_path, mmap_mode="r")
        timestamps = pd.to_datetime(np.load(timestamps_path, mmap_mode="r"), utc=True)
        market_data = np.load(market_data_path, mmap_mode="r")
        close = np.asarray(market_data[:, 3], dtype=np.float64)
        meta = _load_json(meta_path)
        label_metadata = _load_jsonl(label_metadata_path) if label_metadata_path.exists() else []

        n_points = min(int(len(close)), int(max_points_per_ticker))
        timestamps = timestamps[:n_points]
        close = close[:n_points]
        labels = np.asarray(labels[:n_points], dtype=np.int64)
        colors = [color_map.get(int(label), "black") for label in labels]

        counts = pd.Series(labels).value_counts().to_dict()
        width_minutes = ((meta.get("sampler_ticker_meta") or {}).get("h", None))
        tb_width = ((label_metadata[0] or {}).get("bar_close_time") if label_metadata else None)

        ax.plot(timestamps, close, color="#999999", linewidth=1.0, alpha=0.7)
        ax.scatter(timestamps, close, c=colors, s=10, linewidths=0)
        ax.set_title(f"{ticker} | rows={n_points:,}")
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel("close")
        ax.grid(alpha=0.2)
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)
        ax.text(
            1.02,
            0.98,
            "\n".join(
                [
                    f"Down / Exit / Up: {counts.get(0, 0)} / {counts.get(1, 0)} / {counts.get(2, 0)}",
                    f"No label: {counts.get(-1, 0)}",
                    f"Sampled rows shown: {n_points:,}",
                    f"Tuned sampler h: {width_minutes}" if width_minutes is not None else "Tuned sampler h: n/a",
                    f"First labeled close_ts: {tb_width}" if tb_width is not None else "First labeled close_ts: n/a",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        ticker_out = reports_dir / f"{ticker}_labeling.png"
        ticker_fig, ticker_ax = plt.subplots(figsize=(16, 5), dpi=140)
        ticker_ax.plot(timestamps, close, color="#999999", linewidth=1.0, alpha=0.7)
        ticker_ax.scatter(timestamps, close, c=colors, s=10, linewidths=0)
        ticker_ax.set_title(f"Sample labeling for {ticker}")
        ticker_ax.set_xlabel("time (UTC)")
        ticker_ax.set_ylabel("close")
        ticker_ax.grid(alpha=0.2)
        ticker_ax.legend(handles=legend_handles, loc="upper left", frameon=False)
        ticker_fig.tight_layout()
        ticker_fig.savefig(ticker_out, dpi=140)
        plt.close(ticker_fig)
        out_paths.append(ticker_out)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    combined_out = reports_dir / "labeling_sample_tickers.png"
    fig.savefig(combined_out, dpi=140)
    plt.close(fig)
    out_paths.append(combined_out)
    return out_paths
