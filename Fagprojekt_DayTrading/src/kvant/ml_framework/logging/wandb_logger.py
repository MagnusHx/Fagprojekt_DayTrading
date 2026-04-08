from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from kvant.labels import CLASS_NAMES, LABEL_MEANINGS


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _plot_confusion_heatmap(cm: np.ndarray, title: str) -> plt.Figure:
    """
    cm: (C,C) int array. Produces a heatmap with:
      - cell text: "count\n(percent of row)"
    """
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig = plt.figure(figsize=(5.5, 4.8), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = 100.0 * float(cm_norm[i, j])
            ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    return fig


def _normalized_confusion_table(cm: np.ndarray) -> wandb.Table:
    """Create a row-normalized confusion-matrix table for W&B."""
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    table = wandb.Table(columns=["true_class", "pred_class", "count", "row_normalized"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            table.add_data(int(i), int(j), int(cm[i, j]), float(cm_norm[i, j]))
    return table


def _safe_pct(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def _parse_ts(x: Any) -> pd.Timestamp | None:
    if x in (None, "", "None"):
        return None
    try:
        ts = pd.Timestamp(x)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _plot_split_class_balance(split_stats: Dict[str, dict]) -> plt.Figure:
    splits = [s for s in ["train", "val", "test"] if s in split_stats]
    cls_ids = [0, 1, 2]
    cls_labels = [CLASS_NAMES[c] for c in cls_ids]
    x = np.arange(len(splits))

    counts = np.array(
        [[int((split_stats[s].get("y_counts", {}) or {}).get(c, 0)) for s in splits] for c in cls_ids], dtype=np.float64
    )
    totals = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(totals > 0, counts / totals, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)

    bottom = np.zeros(len(splits), dtype=np.float64)
    for i, cls_label in enumerate(cls_labels):
        axes[0].bar(x, counts[i], bottom=bottom, label=cls_label)
        bottom += counts[i]
    axes[0].set_title("Class counts by split")
    axes[0].set_xticks(x, splits)
    axes[0].set_ylabel("samples")

    bottom = np.zeros(len(splits), dtype=np.float64)
    for i, cls_label in enumerate(cls_labels):
        axes[1].bar(x, 100.0 * shares[i], bottom=100.0 * bottom, label=cls_label)
        bottom += shares[i]
    axes[1].set_title("Class share by split")
    axes[1].set_xticks(x, splits)
    axes[1].set_ylabel("percent")
    axes[1].set_ylim(0.0, 100.0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _plot_split_time_ranges(split_stats: Dict[str, dict]) -> plt.Figure:
    rows = []
    for split in ["train", "val", "test"]:
        info = split_stats.get(split, {}) or {}
        start = _parse_ts(info.get("first_ts"))
        end = _parse_ts(info.get("last_ts"))
        if start is None or end is None:
            continue
        rows.append((split, start, end))

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=140)
    if not rows:
        ax.text(0.5, 0.5, "No split timestamps available", ha="center", va="center")
        ax.axis("off")
        return fig

    colors = {"train": "#4C956C", "val": "#F4A259", "test": "#BC4B51"}
    y = np.arange(len(rows))
    widths = np.array([(r[2] - r[1]).total_seconds() / 86400.0 for r in rows], dtype=float)

    for i, (split, start, end) in enumerate(rows):
        ax.barh(i, widths[i], left=start.to_pydatetime(), color=colors.get(split, "#777777"), alpha=0.9)
        ax.text(end.to_pydatetime(), i, f" {start.date()} -> {end.date()}", va="center", fontsize=8)

    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_title("Split time ranges")
    ax.set_xlabel("time")
    ax.grid(axis="x", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_split_expansion(split_stats: Dict[str, dict]) -> plt.Figure:
    train = split_stats.get("train", {}) or {}
    val = split_stats.get("val", {}) or {}
    test = split_stats.get("test", {}) or {}

    train_start = _parse_ts(train.get("first_ts"))
    val_start = _parse_ts(val.get("first_ts"))
    test_start = _parse_ts(test.get("first_ts"))
    test_end = _parse_ts(test.get("last_ts"))

    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=140)
    if None in (train_start, val_start, test_start, test_end):
        ax.text(0.5, 0.5, "Insufficient timestamps for expansion plot", ha="center", va="center")
        ax.axis("off")
        return fig

    phases = [
        ("train only", train_start, val_start),
        ("train + val visible", train_start, test_start),
        ("full window incl. test", train_start, test_end),
    ]
    colors = ["#4C956C", "#F4A259", "#BC4B51"]

    for i, (label, start, end) in enumerate(phases):
        width = max((end - start).total_seconds() / 86400.0, 0.0)
        ax.barh(i, width, left=start.to_pydatetime(), color=colors[i], alpha=0.9)
        ax.text(end.to_pydatetime(), i, f" {label}", va="center", fontsize=8)

    ax.axvline(val_start.to_pydatetime(), color="#F4A259", linestyle="--", linewidth=1)
    ax.axvline(test_start.to_pydatetime(), color="#BC4B51", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(phases)), [p[0] for p in phases])
    ax.set_title("How the split window expands")
    ax.set_xlabel("time")
    ax.grid(axis="x", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_top_ticker_class_balance(per_ticker_rows: List[dict], top_n: int = 12) -> plt.Figure:
    ranked = sorted(per_ticker_rows, key=lambda r: int(r.get("n", 0)), reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=140)
    if not ranked:
        ax.text(0.5, 0.5, "No per-ticker rows available", ha="center", va="center")
        ax.axis("off")
        return fig

    tickers = [str(r["ticker"]) for r in ranked]
    cls0 = np.array([int(r.get("y_count_0", 0)) for r in ranked], dtype=float)
    cls1 = np.array([int(r.get("y_count_1", 0)) for r in ranked], dtype=float)
    cls2 = np.array([int(r.get("y_count_2", 0)) for r in ranked], dtype=float)
    total = np.maximum(cls0 + cls1 + cls2, 1.0)

    ax.bar(tickers, 100.0 * cls0 / total, label=CLASS_NAMES[0])
    ax.bar(tickers, 100.0 * cls1 / total, bottom=100.0 * cls0 / total, label=CLASS_NAMES[1])
    ax.bar(tickers, 100.0 * cls2 / total, bottom=100.0 * (cls0 + cls1) / total, label=CLASS_NAMES[2])
    ax.set_title(f"Class balance for top {len(ranked)} tickers by sample count")
    ax.set_ylabel("percent")
    ax.set_ylim(0.0, 100.0)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_density_summary(density_rows: List[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    if not density_rows:
        for ax in axes:
            ax.text(0.5, 0.5, "No density summary available", ha="center", va="center")
            ax.axis("off")
        return fig

    retention = np.array([float(r.get("retention_ratio", np.nan)) for r in density_rows], dtype=float)
    bpd = np.array([float(r.get("bars_per_day_sampled", np.nan)) for r in density_rows], dtype=float)
    hvals = np.array(
        [float((r.get("sampler_ticker_meta", {}) or {}).get("h", np.nan)) for r in density_rows], dtype=float
    )

    axes[0].hist(
        retention[np.isfinite(retention)], bins=min(20, max(5, len(density_rows))), color="#4C956C", edgecolor="black"
    )
    axes[0].set_title("Retention ratio across tickers")
    axes[0].set_xlabel("sampled/raw")
    axes[0].set_ylabel("tickers")

    axes[1].scatter(hvals[np.isfinite(hvals)], bpd[np.isfinite(hvals)], alpha=0.8, color="#2C7DA0")
    axes[1].set_title("Tuned CUSUM h vs sampled bars/day")
    axes[1].set_xlabel("tuned h")
    axes[1].set_ylabel("bars/day")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _load_density_summary(exp_dir: Path) -> List[dict]:
    path = exp_dir / "density_summary.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


class WandbLogger:
    _SETUP_STEP = 0
    _SHARED_AXIS_KEYS = {"epoch", "global_epoch"}

    def __init__(
        self,
        *,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        api_timeout: int = 29,
        run: Optional[Any] = None,
        namespace: Optional[str] = None,
        step_offset: int = 0,
        manage_run: bool = True,
        **init_kwargs,
    ):
        self.run = run or wandb.init(project=project, name=name, config=config or {}, **init_kwargs)
        self.api = wandb.Api(timeout=int(api_timeout))
        self.api_timeout = int(api_timeout)
        self.namespace = str(namespace).strip("/") if namespace else ""
        self.step_offset = int(step_offset)
        self.manage_run = bool(manage_run)

        # ticker_label -> list of epoch dicts
        self._ticker_history: Dict[str, List[Dict[str, Any]]] = {}
        self._tickers_to_chart: List[str] = []
        self._define_default_metrics()

    def _define_default_metrics(self) -> None:
        wandb.define_metric("global_epoch")
        wandb.define_metric("epoch")

    def child(self, *, namespace: str, step_offset: int = 0) -> WandbLogger:
        return WandbLogger(
            project=self.run.project,
            name=self.run.name,
            api_timeout=self.api_timeout,
            run=self.run,
            namespace=self._qualify_key(namespace),
            step_offset=self.step_offset + int(step_offset),
            manage_run=False,
        )

    def log_config(self, cfg: Any) -> None:
        if is_dataclass(cfg):
            wandb.config.update(asdict(cfg), allow_val_change=True)
        elif isinstance(cfg, dict):
            wandb.config.update(cfg, allow_val_change=True)

    def _qualify_key(self, key: str) -> str:
        if not self.namespace:
            return key
        return f"{self.namespace}/{key}"

    def _qualify_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        qualified: Dict[str, Any] = {}
        for key, value in metrics.items():
            key_str = str(key)
            if key_str in self._SHARED_AXIS_KEYS:
                qualified[key_str] = value
            else:
                qualified[self._qualify_key(key_str)] = value
        return qualified

    def _normalize_step(self, step: Optional[int]) -> Optional[int]:
        if step is None:
            return None
        return self.step_offset + int(step)

    def _log_static(self, metrics: Dict[str, Any]) -> None:
        wandb.log(self._qualify_metrics(metrics), step=self._normalize_step(self._SETUP_STEP))

    def setup(self, *, exp: Any, loaders: Dict[str, Any]) -> None:
        # label meanings
        self.run.config.update({"label_meanings": LABEL_MEANINGS}, allow_val_change=True)
        label_table = wandb.Table(columns=["y", "meaning"])
        for y, meaning in LABEL_MEANINGS.items():
            label_table.add_data(int(y), str(meaning))
        self._log_static({"data/label_meanings": label_table})

        # tickers mapping
        tickers = getattr(exp.store, "tickers_all", None) or []
        ticker_table = wandb.Table(columns=["tid", "ticker", "ticker_label"])
        for tid, tkr in enumerate(tickers):
            ticker_table.add_data(int(tid), str(tkr), f"{tkr} (tid={tid})")
        self._log_static({"data/tickers": ticker_table})

        # first 10 tickers (or fewer)
        n_chart = min(10, len(tickers))
        self._tickers_to_chart = [f"{tickers[tid]} (tid={tid})" for tid in range(n_chart)]
        self.run.config.update({"wandb_chart_first_n_tickers": n_chart}, allow_val_change=True)

        # split distribution table (static)
        # (Assumes ds.summary exists; if you truly want to use ds.display instead, tell me what it returns.)
        dist_table = wandb.Table(columns=["split", "n", "first_ts", "last_ts", "y_count_0", "y_count_1", "y_count_2"])
        split_stats: Dict[str, dict] = {}
        per_ticker_balance_rows: List[dict] = []
        for split, loader in loaders.items():
            if loader is None:
                continue
            ds = loader.dataset
            s = ds.summary(display=False)
            overall = s.get("overall", {}) or {}
            split_stats[str(split)] = overall
            yc = overall.get("y_counts", {}) or {}
            dist_table.add_data(
                str(split),
                _safe_int(overall.get("n", 0)),
                overall.get("first_ts", None),
                overall.get("last_ts", None),
                _safe_int(yc.get(0, 0)),
                _safe_int(yc.get(1, 0)),
                _safe_int(yc.get(2, 0)),
            )

            for ticker, row in sorted((s.get("per_ticker", {}) or {}).items()):
                yct = row.get("y_counts", {}) or {}
                n = _safe_int(row.get("n", 0))
                per_ticker_balance_rows.append(
                    {
                        "split": str(split),
                        "ticker": str(ticker),
                        "tid": _safe_int(row.get("tid", -1), default=-1),
                        "n": n,
                        "first_ts": row.get("first_ts"),
                        "last_ts": row.get("last_ts"),
                        "y_count_0": _safe_int(yct.get(0, 0)),
                        "y_count_1": _safe_int(yct.get(1, 0)),
                        "y_count_2": _safe_int(yct.get(2, 0)),
                        "y_pct_0": _safe_pct(_safe_int(yct.get(0, 0)), n),
                        "y_pct_1": _safe_pct(_safe_int(yct.get(1, 0)), n),
                        "y_pct_2": _safe_pct(_safe_int(yct.get(2, 0)), n),
                    }
                )
        self._log_static({"data/split_distribution": dist_table})

        balance_table = wandb.Table(
            columns=[
                "split",
                "ticker",
                "tid",
                "n",
                "first_ts",
                "last_ts",
                "y_count_0",
                "y_count_1",
                "y_count_2",
                "y_pct_0",
                "y_pct_1",
                "y_pct_2",
            ]
        )
        for row in per_ticker_balance_rows:
            balance_table.add_data(
                row["split"],
                row["ticker"],
                row["tid"],
                row["n"],
                row["first_ts"],
                row["last_ts"],
                row["y_count_0"],
                row["y_count_1"],
                row["y_count_2"],
                row["y_pct_0"],
                row["y_pct_1"],
                row["y_pct_2"],
            )
        self._log_static({"data/per_ticker_class_balance": balance_table})

        fig = _plot_split_class_balance(split_stats)
        self._log_static({"charts/data/class_balance_by_split": wandb.Image(fig)})
        plt.close(fig)

        fig = _plot_split_time_ranges(split_stats)
        self._log_static({"charts/data/split_time_ranges": wandb.Image(fig)})
        plt.close(fig)

        fig = _plot_split_expansion(split_stats)
        self._log_static({"charts/data/split_expansion": wandb.Image(fig)})
        plt.close(fig)

        fig = _plot_top_ticker_class_balance([r for r in per_ticker_balance_rows if r["split"] == "train"])
        self._log_static({"charts/data/top_ticker_train_class_balance": wandb.Image(fig)})
        plt.close(fig)

        density_rows = _load_density_summary(Path(exp.exp_dir))
        if density_rows:
            density_table = wandb.Table(
                columns=[
                    "ticker",
                    "n_raw_full",
                    "n_sampled_full",
                    "retention_ratio",
                    "bars_per_day_raw",
                    "bars_per_day_sampled",
                    "h",
                    "raw_train",
                    "raw_val",
                    "raw_test",
                    "sampled_train",
                    "sampled_val",
                    "sampled_test",
                ]
            )
            for row in density_rows:
                raw_split = row.get("raw_counts_by_split", {}) or {}
                sampled_split = row.get("sampled_counts_by_split", {}) or {}
                meta = row.get("sampler_ticker_meta", {}) or {}
                density_table.add_data(
                    row.get("ticker"),
                    _safe_int(row.get("n_raw_full", 0)),
                    _safe_int(row.get("n_sampled_full", 0)),
                    _to_float_or_nan(row.get("retention_ratio", np.nan)),
                    _to_float_or_nan(row.get("bars_per_day_raw", np.nan)),
                    _to_float_or_nan(row.get("bars_per_day_sampled", np.nan)),
                    _to_float_or_nan(meta.get("h", np.nan)),
                    _safe_int(raw_split.get("train", 0)),
                    _safe_int(raw_split.get("val", 0)),
                    _safe_int(raw_split.get("test", 0)),
                    _safe_int(sampled_split.get("train", 0)),
                    _safe_int(sampled_split.get("val", 0)),
                    _safe_int(sampled_split.get("test", 0)),
                )
            self._log_static({"data/sampling_density": density_table})

            fig = _plot_density_summary(density_rows)
            self._log_static({"charts/data/sampling_density_summary": wandb.Image(fig)})
            plt.close(fig)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        per_ticker_rows = metrics.pop("_per_ticker_rows", None)
        confusion_counts = metrics.pop("_confusion_counts", None)
        profit_curves = metrics.pop("_profit_curves", None)
        local_step = None if step is None else int(step)
        step = self._normalize_step(step)

        if local_step is not None:
            metrics.setdefault("epoch", local_step)
        if step is not None:
            metrics.setdefault("global_epoch", int(step))

        # log scalars
        wandb.log(self._qualify_metrics(metrics), step=step)

        if step is None:
            return

        # confusion matrices (graphical heatmaps)
        if isinstance(confusion_counts, dict):
            for split, cm in confusion_counts.items():
                if cm is None:
                    continue
                fig = _plot_confusion_heatmap(np.asarray(cm), title=f"Confusion matrix ({split})")
                wandb.log({self._qualify_key(f"charts/confusion_matrix/{split}"): wandb.Image(fig)}, step=step)
                wandb.log(
                    {
                        self._qualify_key(f"perf/confusion_matrix_normalized/{split}"): _normalized_confusion_table(
                            np.asarray(cm)
                        )
                    },
                    step=step,
                )
                plt.close(fig)

        if isinstance(profit_curves, list):
            for curve in profit_curves:
                split = str(curve.get("split", "unknown"))
                trade_numbers = curve.get("trade_number", [])
                trade_profit_pct = curve.get("trade_profit_pct", [])
                cum_profit_pct = curve.get("cum_profit_pct", [])
                if not trade_numbers:
                    continue

                table = wandb.Table(columns=["trade_number", "trade_profit_pct", "cum_profit_pct"])
                for trade_number, trade_profit, cum_profit in zip(trade_numbers, trade_profit_pct, cum_profit_pct):
                    table.add_data(int(trade_number), float(trade_profit), float(cum_profit))

                wandb.log({self._qualify_key(f"perf/profit_curve_over_trades/{split}"): table}, step=step)
                wandb.log(
                    {
                        self._qualify_key(f"charts/profit_over_trades/{split}"): wandb.plot.line(
                            table,
                            "trade_number",
                            "cum_profit_pct",
                            title=f"Cumulative profit over trades ({split})",
                        )
                    },
                    step=step,
                )

        if not per_ticker_rows:
            return

        # 1) long-form per-ticker table
        table = wandb.Table(
            columns=[
                "epoch",
                "split",
                "tid",
                "ticker",
                "ticker_label",
                "acc",
                "n",
                "buy_n_trades",
                "buy_profit_avg_per_trade_pct",
                "buy_profit_total_pct",
                "short_n_trades",
                "short_profit_avg_per_trade_pct",
                "short_profit_total_pct",
            ]
        )

        # per-epoch values for charting: ticker_label -> split -> dict(values)
        by_ticker = defaultdict(lambda: defaultdict(dict))

        for r in per_ticker_rows:
            epoch = int(r.get("epoch", local_step) or local_step or 0)
            split = str(r["split"])
            tid = int(r["tid"])
            ticker = str(r["ticker"])
            ticker_label = f"{ticker} (tid={tid})"

            acc = _to_float_or_nan(r.get("acc", np.nan))
            n = int(r.get("n", 0))

            buy_n_trades = int(r.get("buy_n_trades", 0))
            buy_avg = _to_float_or_nan(r.get("buy_profit_avg_per_trade_pct", np.nan))
            buy_total = _to_float_or_nan(r.get("buy_profit_total_pct", 0.0))

            short_n_trades = int(r.get("short_n_trades", 0))
            short_avg = _to_float_or_nan(r.get("short_profit_avg_per_trade_pct", np.nan))
            short_total = _to_float_or_nan(r.get("short_profit_total_pct", 0.0))

            table.add_data(
                epoch,
                split,
                tid,
                ticker,
                ticker_label,
                acc,
                n,
                buy_n_trades,
                buy_avg,
                buy_total,
                short_n_trades,
                short_avg,
                short_total,
            )

            by_ticker[ticker_label][split] = {
                "acc": acc,
                "buy_avg": buy_avg,
                "buy_total": buy_total,
                "short_avg": short_avg,
                "short_total": short_total,
            }

        wandb.log({self._qualify_key("perf/per_ticker_table"): table}, step=step)

        # 2) charts for first 10 tickers (train/val/test lines)
        for ticker_label in self._tickers_to_chart:
            d = by_ticker.get(ticker_label, {})
            if not d:
                continue

            hist = self._ticker_history.setdefault(ticker_label, [])
            hist.append(
                {
                    "epoch": int(local_step or 0),
                    "acc_train": d.get("train", {}).get("acc", np.nan),
                    "acc_val": d.get("val", {}).get("acc", np.nan),
                    "acc_test": d.get("test", {}).get("acc", np.nan),
                    "buy_avg_train": d.get("train", {}).get("buy_avg", np.nan),
                    "buy_avg_val": d.get("val", {}).get("buy_avg", np.nan),
                    "buy_avg_test": d.get("test", {}).get("buy_avg", np.nan),
                    "short_avg_train": d.get("train", {}).get("short_avg", np.nan),
                    "short_avg_val": d.get("val", {}).get("short_avg", np.nan),
                    "short_avg_test": d.get("test", {}).get("short_avg", np.nan),
                    "buy_total_train": d.get("train", {}).get("buy_total", np.nan),
                    "buy_total_val": d.get("val", {}).get("buy_total", np.nan),
                    "buy_total_test": d.get("test", {}).get("buy_total", np.nan),
                    "short_total_train": d.get("train", {}).get("short_total", np.nan),
                    "short_total_val": d.get("val", {}).get("short_total", np.nan),
                    "short_total_test": d.get("test", {}).get("short_total", np.nan),
                }
            )

            xs = [h["epoch"] for h in hist]

            def line_series(key_train: str, key_val: str, key_test: str, title: str, chart_key: str):
                chart = wandb.plot.line_series(
                    xs=xs,
                    ys=[
                        [h[key_train] for h in hist],
                        [h[key_val] for h in hist],
                        [h[key_test] for h in hist],
                    ],
                    keys=["train", "val", "test"],
                    title=title,
                    xname="epoch",
                    split_table=True,
                )
                wandb.log({self._qualify_key(chart_key): chart}, step=step)

            # Accuracy
            line_series(
                "acc_train",
                "acc_val",
                "acc_test",
                f"Per-ticker accuracy: {ticker_label}",
                f"charts/per_ticker/acc/{ticker_label}",
            )

            # Per-trade profit (buy-only)
            line_series(
                "buy_avg_train",
                "buy_avg_val",
                "buy_avg_test",
                f"Buy-only profit (% avg per trade): {ticker_label}",
                f"charts/per_ticker/buy_profit_avg_per_trade_pct/{ticker_label}",
            )

            # Per-trade profit (short-only)
            line_series(
                "short_avg_train",
                "short_avg_val",
                "short_avg_test",
                f"Short-only profit (% avg per trade): {ticker_label}",
                f"charts/per_ticker/short_profit_avg_per_trade_pct/{ticker_label}",
            )

            # Total profit (buy-only)
            line_series(
                "buy_total_train",
                "buy_total_val",
                "buy_total_test",
                f"Buy-only profit (% total): {ticker_label}",
                f"charts/per_ticker/buy_profit_total_pct/{ticker_label}",
            )

            # Total profit (short-only)
            line_series(
                "short_total_train",
                "short_total_val",
                "short_total_test",
                f"Short-only profit (% total): {ticker_label}",
                f"charts/per_ticker/short_profit_total_pct/{ticker_label}",
            )

    def stop(self) -> None:
        if self.manage_run:
            self.run.finish()
