#!/usr/bin/env python3
"""Summarize sample counts for prepared CV manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kvant.ml_prepare_data.data_loading import PreparedExperiment


def _parse_manifest_arg(value: str) -> tuple[str, Path]:
    """Parse NAME=PATH manifest arguments."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Manifest name cannot be empty.")
    return name, Path(path.strip())


def _load_manifest(path: Path) -> dict:
    """Load a prepared CV manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_manifest(name: str, path: Path) -> dict[str, float | int | str]:
    """Summarize split sample counts for one manifest."""
    manifest = _load_manifest(path)
    train = val = test = 0
    first_ts = None
    last_ts = None
    for fold in manifest["folds"]:
        experiment = PreparedExperiment(Path(fold["exp_dir"]))
        train += int(len(experiment.index_train))
        val += int(len(experiment.index_val))
        test += int(len(experiment.index_test))
        for index in (experiment.index_train, experiment.index_val, experiment.index_test):
            if len(index) == 0:
                continue
            timestamps = [experiment.store.timestamp(int(tid), int(tpos)) for tid, tpos in index]
            fold_first = np.min(timestamps)
            fold_last = np.max(timestamps)
            first_ts = fold_first if first_ts is None else min(first_ts, fold_first)
            last_ts = fold_last if last_ts is None else max(last_ts, fold_last)
    return {
        "setup": name,
        "manifest": str(path),
        "folds": int(len(manifest["folds"])),
        "train_samples": train,
        "val_samples": val,
        "test_samples": test,
        "total_samples": train + val + test,
        "first_timestamp": "" if first_ts is None else str(np.datetime_as_string(first_ts, unit="s")),
        "last_timestamp": "" if last_ts is None else str(np.datetime_as_string(last_ts, unit="s")),
    }


def write_sample_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Write a bar plot comparing total samples per setup."""
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.5, len(df) * 0.8), 4.5), dpi=150)
    ax.bar(df["setup"], df["total_samples"])
    ax.set_ylabel("Total samples")
    ax.set_title("Prepared sample counts")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_count_comparison.png")
    fig.savefig(output_dir / "sample_count_comparison.pdf")
    plt.close(fig)


def main() -> None:
    """Run the prepared-manifest summary CLI."""
    parser = argparse.ArgumentParser(description="Summarize prepared CV manifest sample counts.")
    parser.add_argument("--manifest", action="append", type=_parse_manifest_arg, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/generated/tables/dataset_summary.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("reports/generated/tables/dataset_summary.tex"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/generated/figures"))
    args = parser.parse_args()

    rows = [summarize_manifest(name, path) for name, path in args.manifest]
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    write_sample_plot(df, args.figure_dir)
    print(df.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
