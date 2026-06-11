from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kvant.labels import label_semantics_payload, pipeline_label_spaces_payload
from kvant.ml_prepare_data import prepared_data_root


FEATURE_NAMES = (
    "ewmstd_close_20b",
    "logret_1",
    "volume_log1p",
    "minute_sin",
    "minute_cos",
    "trend",
)
MARKET_DATA_COLUMNS = 5


def _split_index(*, ticker_count: int, start: int, stop: int) -> np.ndarray:
    """Return a split index with rows for every ticker and position in the interval."""
    rows = [(tid, pos) for tid in range(ticker_count) for pos in range(start, stop)]
    return np.asarray(rows, dtype=np.int64)


def _labels_for_rows(row_count: int) -> np.ndarray:
    """Return deterministic three-class event labels with down, exit, and up outcomes."""
    pattern = np.asarray([0, 2, 1, 2, 0, 2, 1, 0], dtype=np.int64)
    repeats = int(np.ceil(row_count / len(pattern)))
    return np.tile(pattern, repeats)[:row_count].astype(np.int64, copy=False)


def _features_for_ticker(*, row_count: int, ticker_id: int, seed: int) -> np.ndarray:
    """Return deterministic synthetic feature rows matching the training feature contract."""
    rng = np.random.default_rng(seed + ticker_id)
    t = np.arange(row_count, dtype=np.float32)
    close = 100.0 + ticker_id * 5.0 + 0.04 * t + np.sin(t / 5.0)
    logret = np.zeros(row_count, dtype=np.float32)
    logret[1:] = np.diff(np.log(close)).astype(np.float32)
    volatility = np.maximum(0.001, np.abs(np.sin(t / 9.0)) * 0.015 + 0.002)
    volume = np.log1p(1_000_000.0 + 10_000.0 * np.cos(t / 7.0) + ticker_id * 20_000.0)
    minute = (t % 390.0) / 390.0
    trend = (t - t.mean()) / max(float(t.std()), 1.0)
    noise = rng.normal(loc=0.0, scale=0.005, size=(row_count, len(FEATURE_NAMES))).astype(np.float32)
    features = np.stack(
        [
            volatility,
            logret,
            volume.astype(np.float32),
            np.sin(2.0 * np.pi * minute).astype(np.float32),
            np.cos(2.0 * np.pi * minute).astype(np.float32),
            trend.astype(np.float32),
        ],
        axis=1,
    )
    return (features + noise).astype(np.float32, copy=False)


def _market_data_for_ticker(*, row_count: int, ticker_id: int) -> np.ndarray:
    """Return synthetic sampled OHLCV rows for optional backtest validation."""
    t = np.arange(row_count, dtype=np.float32)
    close = 100.0 + ticker_id * 5.0 + 0.04 * t + np.sin(t / 5.0)
    open_ = close - 0.02
    high = close + 0.08
    low = close - 0.08
    volume = 1_000_000.0 + 10_000.0 * np.cos(t / 7.0) + ticker_id * 20_000.0
    return np.stack([open_, high, low, close, volume], axis=1).astype(np.float32, copy=False)


def _timestamps(row_count: int) -> np.ndarray:
    """Return monotonic minute timestamps for a synthetic trading sequence."""
    start = np.datetime64("2024-01-02T14:30:00", "ns")
    offsets = np.arange(row_count).astype("timedelta64[m]")
    return (start + offsets).astype("datetime64[ns]")


def _metadata_rows(labels: np.ndarray, timestamps: np.ndarray) -> list[dict[str, object]]:
    """Return label metadata rows used by meta-label and trade diagnostics."""
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        ts = np.datetime_as_string(timestamps[idx], unit="s")
        close_idx = min(idx + 3, len(timestamps) - 1)
        close_ts = np.datetime_as_string(timestamps[close_idx], unit="s")
        if int(label) == 2:
            pnl_fraction = 0.012
        elif int(label) == 0:
            pnl_fraction = -0.010
        else:
            pnl_fraction = 0.0
        rows.append(
            {
                "label": int(label),
                "signal_time": ts,
                "bar_open_time": ts,
                "bar_close_time": close_ts,
                "pnl_fraction": float(pnl_fraction),
                "pnl_absolute": float(pnl_fraction * 100.0),
            }
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_ticker(exp_dir: Path, *, ticker: str, ticker_id: int, row_count: int, seed: int) -> None:
    """Write one ticker directory in the prepared artifact format."""
    ticker_dir = exp_dir / "tickers" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels_for_rows(row_count)
    timestamps = _timestamps(row_count)
    np.save(ticker_dir / "features.npy", _features_for_ticker(row_count=row_count, ticker_id=ticker_id, seed=seed))
    np.save(ticker_dir / "labels.npy", labels)
    np.save(ticker_dir / "timestamps.npy", timestamps)
    np.save(ticker_dir / "market_data.npy", _market_data_for_ticker(row_count=row_count, ticker_id=ticker_id))
    metadata = _metadata_rows(labels, timestamps)
    ticker_dir.joinpath("label_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata) + "\n")
    _write_json(ticker_dir / "meta.json", {"ticker": ticker, "ticker_id": ticker_id, "row_count": row_count})


def create_smoke_prepared_experiment(
    *,
    out_root: Path,
    label: str,
    row_count: int,
    lookback: int,
    seed: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Create a one-fold synthetic prepared experiment and CV manifest."""
    if row_count < lookback + 30:
        raise ValueError("row_count must leave enough rows after lookback for train, val, and test splits.")

    out_root.mkdir(parents=True, exist_ok=True)
    exp_id = f"{label}_fold00"
    exp_dir = out_root / exp_id
    manifest_path = out_root / f"{label}_cv_manifest.json"
    if exp_dir.exists() and not overwrite:
        raise FileExistsError(f"{exp_dir} already exists. Pass --overwrite to replace smoke artifacts.")

    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    exp_dir.joinpath("tickers").mkdir()

    tickers = ["SMOKE_A", "SMOKE_B"]
    train_start = lookback
    train_stop = int(row_count * 0.65)
    val_stop = int(row_count * 0.82)

    config = {
        "experiment_name": "synthetic_smoke_prepared_experiment",
        "pipeline_stage": "event_outcome",
        "lookback_L": int(lookback),
        "label_spaces": pipeline_label_spaces_payload(),
        "label_semantics": label_semantics_payload(drop_time_exit_label=False),
        "feature_engineer": {
            "name": "synthetic_smoke",
            "feature_names_": list(FEATURE_NAMES),
            "mean_": [0.0] * len(FEATURE_NAMES),
            "std_": [1.0] * len(FEATURE_NAMES),
        },
        "labeler": {
            "name": label,
            "width_minutes": 30,
            "height": 0.01,
            "drop_time_exit_label": False,
        },
        "sampler": {
            "name": "synthetic_every_minute",
            "aggregate_ohlcv": True,
        },
    }
    _write_json(exp_dir / "config.json", config)
    _write_json(exp_dir / "tickers_all.json", tickers)
    _write_json(exp_dir / "tickers_train.json", tickers)
    _write_json(exp_dir / "tickers_val.json", tickers)
    _write_json(exp_dir / "tickers_test.json", tickers)

    for ticker_id, ticker in enumerate(tickers):
        _write_ticker(exp_dir, ticker=ticker, ticker_id=ticker_id, row_count=row_count, seed=seed)

    np.save(exp_dir / "index_train.npy", _split_index(ticker_count=len(tickers), start=train_start, stop=train_stop))
    np.save(exp_dir / "index_val.npy", _split_index(ticker_count=len(tickers), start=train_stop, stop=val_stop))
    np.save(exp_dir / "index_test.npy", _split_index(ticker_count=len(tickers), start=val_stop, stop=row_count))
    density_rows = [
        {
            "ticker": ticker,
            "n_raw_full": int(row_count),
            "n_sampled_full": int(row_count),
            "retention_ratio": 1.0,
            "bars_per_day_raw": float(row_count),
            "bars_per_day_sampled": float(row_count),
            "raw_counts_by_split": {
                "train": int(train_stop - train_start),
                "val": int(val_stop - train_stop),
                "test": int(row_count - val_stop),
            },
            "sampled_counts_by_split": {
                "train": int(train_stop - train_start),
                "val": int(val_stop - train_stop),
                "test": int(row_count - val_stop),
            },
            "sampler_ticker_meta": {"h": 0.0},
        }
        for ticker in tickers
    ]
    _write_json(exp_dir / "density_summary.json", density_rows)

    manifest = {
        "label": label,
        "n_folds": 1,
        "folds": [
            {
                "fold_idx": 0,
                "exp_id": exp_id,
                "exp_dir": str(exp_dir),
                "train_tickers": tickers,
                "val_tickers": tickers,
                "test_tickers": tickers,
            }
        ],
    }
    _write_json(manifest_path, manifest)
    (out_root / "last_experiment.txt").write_text(f"{exp_id}\n")
    (out_root / "last_experiment_cv_manifest.txt").write_text(f"{manifest_path}\n")
    return exp_dir, manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a tiny one-fold prepared artifact for local/HPC smoke tests.")
    parser.add_argument("--out-root", type=Path, default=prepared_data_root)
    parser.add_argument("--label", type=str, default="smoke_one_fold")
    parser.add_argument("--row-count", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create the smoke prepared experiment and print the generated paths."""
    args = parse_args()
    exp_dir, manifest_path = create_smoke_prepared_experiment(
        out_root=args.out_root,
        label=args.label,
        row_count=args.row_count,
        lookback=args.lookback,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Prepared smoke fold: {exp_dir}")
    print(f"Prepared smoke manifest: {manifest_path}")


if __name__ == "__main__":
    main()
