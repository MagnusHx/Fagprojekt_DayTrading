#!/usr/bin/env python3
"""Create a preparation command for a time-bar baseline matched to a reference sample density."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from kvant.ml_prepare_data.data_loading import PreparedExperiment


def _load_manifest(path: Path) -> dict:
    """Load a prepared CV manifest JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selection(path: Path) -> dict:
    """Load the selected grid config JSON produced by select_best_grid_config.py."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    if not isinstance(selected, dict):
        raise ValueError(f"Invalid selection JSON: {path}")
    return selected


def _split_index(experiment: PreparedExperiment, split: str) -> np.ndarray:
    """Return the requested prepared split index."""
    if split == "train":
        return np.asarray(experiment.index_train, dtype=np.int64)
    if split == "val":
        return np.asarray(experiment.index_val, dtype=np.int64)
    if split == "test":
        return np.asarray(experiment.index_test, dtype=np.int64)
    if split == "all":
        return np.concatenate(
            [
                np.asarray(experiment.index_train, dtype=np.int64),
                np.asarray(experiment.index_val, dtype=np.int64),
                np.asarray(experiment.index_test, dtype=np.int64),
            ],
            axis=0,
        )
    raise ValueError(f"Unknown split: {split}")


def bars_per_ticker_day(manifest_path: Path, *, split: str) -> tuple[float, int, int]:
    """Compute average usable samples per ticker-day from a prepared manifest."""
    manifest = _load_manifest(manifest_path)
    counts: dict[tuple[int, str, str], int] = defaultdict(int)

    for fold in manifest["folds"]:
        fold_idx = int(fold["fold_idx"])
        experiment = PreparedExperiment(Path(fold["exp_dir"]))
        index = _split_index(experiment, split)
        for tid, tpos in index:
            timestamp = pd.Timestamp(experiment.store.timestamp(int(tid), int(tpos)))
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            ticker = experiment.store.ticker(int(tid))
            day = str(timestamp.date())
            counts[(fold_idx, ticker, day)] += 1

    if not counts:
        raise RuntimeError(f"No usable samples found in {manifest_path} for split={split!r}.")

    values = np.asarray(list(counts.values()), dtype=float)
    return float(np.mean(values)), int(values.sum()), int(len(values))


def matched_timebar_minutes(bars_per_day: float, *, session_minutes: int) -> int:
    """Return the nearest integer time-bar interval matching a target bars/day density."""
    if bars_per_day <= 0.0:
        raise ValueError("bars_per_day must be positive.")
    return max(1, int(round(float(session_minutes) / float(bars_per_day))))


def build_prepare_command(
    *,
    minutes: int,
    labeler: str,
    barrier_height_pct: float,
    barrier_width: int,
    output_manifest: Path,
) -> list[str]:
    """Build the density-matched time-bar preparation command."""
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "kvant.ml_prepare_data.prepare_experiment",
        "--sampler",
        "time_bar",
        "--time-bar-minutes",
        str(int(minutes)),
        "--labeler",
        labeler,
        "--barrier-width",
        str(int(barrier_width)),
    ]
    if labeler == "triple_barrier":
        command.extend(["--barrier-height-pct", f"{float(barrier_height_pct):g}"])
    command.extend(["--cv-manifest", str(output_manifest)])
    return command


def main() -> None:
    """Run the density-matched time-bar command generator."""
    parser = argparse.ArgumentParser(description="Create a density-matched time-bar preparation command.")
    parser.add_argument("--reference-manifest", type=Path, default=None)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--session-minutes", type=int, default=390)
    parser.add_argument("--labeler", choices=("next_bar", "triple_barrier"), default="next_bar")
    parser.add_argument("--barrier-height-pct", type=float, default=None)
    parser.add_argument("--barrier-width", type=int, default=240)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("src/kvant/ml_framework/prepared/E2_timebar_density_matched_cv_manifest.json"),
    )
    parser.add_argument("--execute", action="store_true", help="Prints only by default; use README command manually.")
    args = parser.parse_args()

    selected = _load_selection(args.selection_json) if args.selection_json is not None else {}
    reference_manifest = args.reference_manifest
    if reference_manifest is None and selected:
        reference_manifest = Path(str(selected["manifest_path"]))
    if reference_manifest is None:
        raise SystemExit("Pass --reference-manifest or --selection-json.")

    barrier_height_pct = args.barrier_height_pct
    if barrier_height_pct is None and selected:
        barrier_height_pct = float(selected["barrier_height_pct"])
    if barrier_height_pct is None:
        barrier_height_pct = 2.0

    target_bpd, n_samples, n_ticker_days = bars_per_ticker_day(reference_manifest, split=str(args.split))
    minutes = matched_timebar_minutes(target_bpd, session_minutes=int(args.session_minutes))
    approximate_bpd = float(args.session_minutes) / float(minutes)
    command = build_prepare_command(
        minutes=minutes,
        labeler=str(args.labeler),
        barrier_height_pct=float(barrier_height_pct),
        barrier_width=int(args.barrier_width),
        output_manifest=args.output_manifest,
    )

    print(f"Reference manifest: {reference_manifest}")
    print(f"Split used for matching: {args.split}")
    print(f"Reference usable samples: {n_samples} across {n_ticker_days} ticker-days")
    print(f"Reference bars/ticker-day: {target_bpd:.4f}")
    print(f"Matched time-bar interval: {minutes} minutes")
    print(f"Approximate time-bar bars/day: {approximate_bpd:.4f}")
    print("")
    print("Prepare command:")
    print(" ".join(command))

    if args.execute:
        import subprocess

        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
