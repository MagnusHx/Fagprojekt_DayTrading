#!/usr/bin/env python3
"""Select the best CUSUM/triple-barrier grid configuration from result CSVs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd


def _extract_params(run_name: str) -> tuple[int, float, float] | None:
    """Extract barrier width, TB percent, and CUSUM percent from a grid run name."""
    match = re.search(
        r"w(?P<width>\d+)-tb(?P<tb>\d+(?:p\d+)?)-cusum(?P<cusum>\d+(?:p\d+)?)",
        run_name,
    )
    if match is None:
        return None
    width = int(match.group("width"))
    tb_pct = float(match.group("tb").replace("p", "."))
    cusum_pct = float(match.group("cusum").replace("p", "."))
    return width, tb_pct, cusum_pct


def _manifest_path(*, width: int, tb_pct: float, cusum_pct: float) -> Path:
    """Return the manifest path produced by prepare_experiment for this config."""
    cusum_h = cusum_pct / 100.0
    return Path("src/kvant/ml_framework/prepared") / f"sb_L_12_w{width}_h{tb_pct:g}_fixedCUSUM{cusum_h:g}_cv_manifest.json"


def _score_csv(path: Path, *, metric: str, tie_breakers: list[str]) -> dict[str, float | str | int]:
    """Load and score one grid result CSV."""
    run_name = path.stem
    params = _extract_params(run_name)
    if params is None:
        raise ValueError(f"Could not parse grid parameters from {run_name!r}.")
    width, tb_pct, cusum_pct = params

    df = pd.read_csv(path)
    if metric not in df.columns:
        raise ValueError(f"{path} is missing primary metric {metric!r}.")

    row: dict[str, float | str | int] = {
        "run_name": run_name,
        "result_path": str(path),
        "manifest_path": str(_manifest_path(width=width, tb_pct=tb_pct, cusum_pct=cusum_pct)),
        "barrier_width": int(width),
        "barrier_height_pct": float(tb_pct),
        "barrier_height": float(tb_pct / 100.0),
        "cusum_h_pct": float(cusum_pct),
        "cusum_h": float(cusum_pct / 100.0),
        metric: float(df[metric].dropna().mean()),
    }
    for tie_breaker in tie_breakers:
        row[tie_breaker] = float(df[tie_breaker].dropna().mean()) if tie_breaker in df.columns else float("-inf")
    return row


def _write_env(path: Path, selected: dict[str, float | str | int]) -> None:
    """Write a shell-sourceable env file for downstream README commands."""
    width = int(selected["barrier_width"])
    tb_pct = float(selected["barrier_height_pct"])
    cusum_pct = float(selected["cusum_h_pct"])
    resnet_run = f"E2-grid-resnet_lstm-w{width}-tb{tb_pct:g}-cusum{cusum_pct:g}-nometa"
    confidence_glob = f"results/grid_search/E2-grid-conv1d-w{width}-tb{tb_pct:g}-cusum{cusum_pct:g}-nometa-ct*.csv"
    meta_glob = f"results/grid_search/E2-grid-conv1d-w{width}-tb{tb_pct:g}-cusum{cusum_pct:g}-mt*.csv"
    lines = [
        f"export BEST_GRID_RUN='{selected['run_name']}'",
        f"export BEST_GRID_RESULT='{selected['result_path']}'",
        f"export BEST_MANIFEST='{selected['manifest_path']}'",
        f"export BEST_RESNET_RUN='{resnet_run}'",
        f"export BEST_RESNET_RESULT='results/grid_search/{resnet_run}.csv'",
        f"export BEST_CONFIDENCE_GLOB='{confidence_glob}'",
        f"export BEST_META_GLOB='{meta_glob}'",
        f"export BEST_CUSUM_H='{selected['cusum_h']}'",
        f"export BEST_TB_HEIGHT='{selected['barrier_height']}'",
        f"export BEST_TB_HEIGHT_PCT='{selected['barrier_height_pct']}'",
        f"export BEST_BARRIER_WIDTH='{selected['barrier_width']}'",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_promising(path: Path, selected: dict[str, float | str | int]) -> None:
    """Write the selected config in the format consumed by run_experiment_grid.py."""
    payload = {
        "description": "Auto-written by scripts/select_best_grid_config.py from validation grid results.",
        "configs": [
            {
                "cusum_h": float(selected["cusum_h"]),
                "barrier_height": float(selected["barrier_height"]),
                "barrier_width": int(selected["barrier_width"]),
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run the grid-selection CLI."""
    parser = argparse.ArgumentParser(description="Select the best grid config using validation metrics.")
    parser.add_argument("--results-glob", default="results/grid_search/E2-grid-conv1d-w240-*.csv")
    parser.add_argument("--primary-metric", default="val_f1_macro")
    parser.add_argument(
        "--tie-breaker",
        action="append",
        default=["val_portfolio_sharpe_ratio_annualized", "val_portfolio_total_return_pct"],
    )
    parser.add_argument("--selection-json", type=Path, default=Path("artifacts/final_plan/selected_grid.json"))
    parser.add_argument("--env-out", type=Path, default=Path("artifacts/final_plan/selected_grid.env"))
    parser.add_argument("--promising-out", type=Path, default=Path("reports/promising_grid_configs.json"))
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(str(args.results_glob)))]
    if not paths:
        raise FileNotFoundError(f"No result CSVs matched {args.results_glob!r}.")

    rows = [_score_csv(path, metric=str(args.primary_metric), tie_breakers=list(args.tie_breaker)) for path in paths]
    sort_keys = [str(args.primary_metric), *list(args.tie_breaker)]
    selected = max(
        rows,
        key=lambda row: tuple(float(row.get(key, float("-inf"))) for key in sort_keys),
    )

    payload = {
        "selection_rule": {
            "primary_metric": str(args.primary_metric),
            "tie_breakers": list(args.tie_breaker),
            "direction": "max",
        },
        "selected": selected,
        "all_candidates": rows,
    }
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_env(args.env_out, selected)
    _write_promising(args.promising_out, selected)

    print("Selected grid configuration:")
    print(json.dumps(selected, indent=2))
    print(f"Wrote {args.selection_json}")
    print(f"Wrote {args.env_out}")
    print(f"Wrote {args.promising_out}")
    print("")
    print("Next command:")
    print(f"source {args.env_out}")


if __name__ == "__main__":
    main()
