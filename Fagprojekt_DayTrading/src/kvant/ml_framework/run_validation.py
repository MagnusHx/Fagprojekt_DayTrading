from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kvant.ml_prepare_data.data_loading import PreparedExperiment


class RunValidationError(RuntimeError):
    """Raised when a run preflight check fails."""


@dataclass(frozen=True)
class PreparedArtifactDiagnostics:
    exp_dir: Path
    label_regime: str
    n_classes: int
    n_features: int
    has_market_data: bool
    split_summaries: dict[str, dict[str, Any]]
    feature_checks: dict[str, Any]
    dependency_checks: dict[str, bool]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "exp_dir": str(self.exp_dir),
            "label_regime": self.label_regime,
            "n_classes": int(self.n_classes),
            "n_features": int(self.n_features),
            "has_market_data": bool(self.has_market_data),
            "split_summaries": self.split_summaries,
            "feature_checks": self.feature_checks,
            "dependency_checks": self.dependency_checks,
        }


def git_commit_or_none(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _split_summary(exp: PreparedExperiment, split_name: str, index: np.ndarray) -> dict[str, Any]:
    if index.ndim != 2 or index.shape[1] != 2:
        raise RunValidationError(
            f"{exp.exp_dir}: {split_name} index must have shape (N, 2), got {tuple(index.shape)}."
        )

    ds = {
        "train": exp.get_datasets()[0],
        "val": exp.get_datasets()[1],
        "test": exp.get_datasets()[2],
    }[split_name]
    summary = ds.summary(display=False)
    per_ticker = summary.get("per_ticker", {}) or {}
    overall = summary.get("overall", {}) or {}
    return {
        "n": int(overall.get("n", 0)),
        "first_ts": overall.get("first_ts"),
        "last_ts": overall.get("last_ts"),
        "y_counts": {str(k): int(v) for k, v in (overall.get("y_counts", {}) or {}).items()},
        "ticker_count": int(len(per_ticker)),
    }


def _validate_index_bounds(exp: PreparedExperiment, split_name: str, index: np.ndarray) -> None:
    tids = index[:, 0].astype(np.int64, copy=False)
    tpos = index[:, 1].astype(np.int64, copy=False)
    n_tickers = len(exp.store.tickers_all)
    if np.any(tids < 0) or np.any(tids >= n_tickers):
        bad = index[(tids < 0) | (tids >= n_tickers)][0].tolist()
        raise RunValidationError(f"{exp.exp_dir}: {split_name} index has invalid ticker id row {bad}.")

    for tid in np.unique(tids):
        ticker_len = len(exp.store._labels[int(tid)])
        bad_mask = (tids == tid) & ((tpos < 0) | (tpos >= ticker_len))
        if np.any(bad_mask):
            bad = index[bad_mask][0].tolist()
            ticker = exp.store.ticker(int(tid))
            raise RunValidationError(
                f"{exp.exp_dir}: {split_name} index has invalid position {bad} for ticker {ticker}."
            )


def _validate_split_overlap(exp: PreparedExperiment) -> None:
    pairs = {
        "train": exp.index_train,
        "val": exp.index_val,
        "test": exp.index_test,
    }
    structured = {
        name: np.ascontiguousarray(index).view([("tid", index.dtype), ("tpos", index.dtype)]).reshape(-1)
        for name, index in pairs.items()
    }
    intersections = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    for left, right in intersections:
        overlap = np.intersect1d(structured[left], structured[right], assume_unique=False)
        if len(overlap):
            raise RunValidationError(
                f"{exp.exp_dir}: split indices overlap between {left} and {right} ({len(overlap)} rows)."
            )


def validate_prepared_experiment(
    exp_dir: Path,
    *,
    require_market_data: bool = False,
) -> PreparedArtifactDiagnostics:
    exp_dir = Path(exp_dir)
    if not (exp_dir / "config.json").exists():
        raise RunValidationError(f"{exp_dir}: missing config.json.")

    try:
        exp = PreparedExperiment(exp_dir)
    except Exception as exc:
        raise RunValidationError(f"{exp_dir}: failed to load prepared experiment: {exc}") from exc

    _validate_index_bounds(exp, "train", exp.index_train)
    _validate_index_bounds(exp, "val", exp.index_val)
    _validate_index_bounds(exp, "test", exp.index_test)
    _validate_split_overlap(exp)

    invalid_feature_values = 0
    total_rows = 0
    non_monotonic_tickers: list[str] = []
    inconsistent_market_data: list[str] = []
    label_value_violations: list[str] = []
    referenced_positions_by_tid: dict[int, np.ndarray] = {}
    n_features = None

    all_index = np.concatenate([exp.index_train, exp.index_val, exp.index_test], axis=0)
    for tid in np.unique(all_index[:, 0].astype(np.int64, copy=False)):
        mask = all_index[:, 0].astype(np.int64, copy=False) == tid
        referenced_positions_by_tid[int(tid)] = np.unique(all_index[mask, 1].astype(np.int64, copy=False))

    for tid, ticker in enumerate(exp.store.tickers_all):
        X = exp.store._features[tid]
        y = exp.store._labels[tid]
        ts = exp.store._timestamps[tid]
        label_meta = exp.store._label_metadata[tid]
        market_data = exp.store._market_data[tid]

        if len(X) != len(y) or len(y) != len(ts) or len(label_meta) != len(y):
            raise RunValidationError(
                f"{exp_dir}: ticker {ticker} has inconsistent lengths features={len(X)} labels={len(y)} "
                f"timestamps={len(ts)} label_metadata={len(label_meta)}."
            )
        if X.ndim != 2:
            raise RunValidationError(f"{exp_dir}: ticker {ticker} features must be 2D, got ndim={X.ndim}.")
        if n_features is None:
            n_features = int(X.shape[1])
        elif int(X.shape[1]) != int(n_features):
            raise RunValidationError(
                f"{exp_dir}: ticker {ticker} has {X.shape[1]} features, expected {n_features}."
            )

        total_rows += int(len(y))
        invalid_feature_values += int(np.size(X) - np.count_nonzero(np.isfinite(X)))

        ts_ns = np.asarray(ts).astype("datetime64[ns]").astype(np.int64)
        if np.any(np.diff(ts_ns) < 0):
            non_monotonic_tickers.append(str(ticker))

        if market_data is not None and len(market_data) != len(ts):
            inconsistent_market_data.append(str(ticker))
        if market_data is None and require_market_data:
            inconsistent_market_data.append(str(ticker))

        referenced_positions = referenced_positions_by_tid.get(int(tid), np.asarray([], dtype=np.int64))
        referenced_labels = np.asarray(y, dtype=np.int64)[referenced_positions] if len(referenced_positions) else np.asarray([], dtype=np.int64)
        bad_labels = np.setdiff1d(np.unique(referenced_labels), np.asarray(exp.label_ids, dtype=np.int64))
        if len(bad_labels):
            label_value_violations.append(f"{ticker}:{bad_labels.tolist()}")

    if non_monotonic_tickers:
        raise RunValidationError(
            f"{exp_dir}: timestamps are not monotonic for tickers {', '.join(non_monotonic_tickers[:5])}."
        )
    if inconsistent_market_data:
        raise RunValidationError(
            f"{exp_dir}: market_data integrity failed for tickers {', '.join(inconsistent_market_data[:5])}. "
            "Regenerate the prepared experiment with raw sampled OHLCV persistence enabled."
        )
    if label_value_violations:
        raise RunValidationError(
            f"{exp_dir}: labels outside declared semantics found for {', '.join(label_value_violations[:5])}."
        )

    feature_engineer_cfg = (exp.cfg.get("feature_engineer") or {}) if isinstance(exp.cfg, dict) else {}
    feature_names = feature_engineer_cfg.get("feature_names_")
    mean_ = feature_engineer_cfg.get("mean_")
    std_ = feature_engineer_cfg.get("std_")
    if feature_names is not None and len(feature_names) != n_features:
        raise RunValidationError(
            f"{exp_dir}: feature_engineer.feature_names_ has length {len(feature_names)}, expected {n_features}."
        )
    if mean_ is not None and len(mean_) != n_features:
        raise RunValidationError(f"{exp_dir}: feature_engineer.mean_ has length {len(mean_)}, expected {n_features}.")
    if std_ is not None and len(std_) != n_features:
        raise RunValidationError(f"{exp_dir}: feature_engineer.std_ has length {len(std_)}, expected {n_features}.")

    label_regime = "binary" if exp.n_classes == 2 else "three_class"
    split_summaries = {
        "train": _split_summary(exp, "train", exp.index_train),
        "val": _split_summary(exp, "val", exp.index_val),
        "test": _split_summary(exp, "test", exp.index_test),
    }
    dependency_checks = {
        "tabulate": _dependency_available("tabulate"),
        "wandb": _dependency_available("wandb"),
    }
    feature_checks = {
        "total_rows": int(total_rows),
        "n_features": int(n_features or 0),
        "nonfinite_feature_values": int(invalid_feature_values),
    }

    return PreparedArtifactDiagnostics(
        exp_dir=exp_dir,
        label_regime=label_regime,
        n_classes=int(exp.n_classes),
        n_features=int(n_features or 0),
        has_market_data=bool(exp.store.has_market_data),
        split_summaries=split_summaries,
        feature_checks=feature_checks,
        dependency_checks=dependency_checks,
    )


def validate_cv_manifest(
    manifest_path: Path,
    *,
    require_market_data: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RunValidationError(f"CV manifest does not exist: {manifest_path}")
    if manifest_path.suffix == ".txt":
        try:
            resolved = Path(manifest_path.read_text().strip())
        except Exception as exc:
            raise RunValidationError(f"Failed to read CV manifest pointer {manifest_path}: {exc}") from exc
        if not resolved.exists():
            raise RunValidationError(
                f"CV manifest pointer {manifest_path} points to missing file: {resolved}"
            )
        manifest_path = resolved
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise RunValidationError(f"Failed to parse CV manifest {manifest_path}: {exc}") from exc

    folds = payload.get("folds", [])
    if not folds:
        raise RunValidationError(f"No folds found in CV manifest: {manifest_path}")

    seen_fold_idxs: set[int] = set()
    fold_diagnostics = []
    for fold in folds:
        fold_idx = int(fold.get("fold_idx"))
        if fold_idx in seen_fold_idxs:
            raise RunValidationError(f"Duplicate fold_idx={fold_idx} in CV manifest {manifest_path}.")
        seen_fold_idxs.add(fold_idx)
        exp_dir = Path(fold.get("exp_dir", ""))
        diagnostics = validate_prepared_experiment(exp_dir, require_market_data=require_market_data)
        fold_diagnostics.append(
            {
                "fold_idx": fold_idx,
                "exp_id": str(fold.get("exp_id", exp_dir.name)),
                **diagnostics.to_jsonable(),
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "n_folds": int(len(folds)),
        "folds": fold_diagnostics,
    }
