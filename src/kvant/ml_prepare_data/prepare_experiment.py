# prepare_experiment.py
import argparse
from kvant.ml_prepare_data.features.feature_engineering import (
    IntradayTA10Features,
    StandardizedFeatures,
    FeatureEngineer,
)
from kvant.ml_prepare_data.features.feature_selection import (
    FeatureSelector,
    PrimarySideFScoreSelector,
)
import json
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm

from kvant.labels import (
    event_labels_to_side_labels,
    label_semantics_payload,
    pipeline_label_spaces_payload,
)
from kvant.ml_prepare_data.labelling.tripple_bar import Labeler, TripleBarrierLabeler
from kvant.ml_prepare_data.labelling.next_bar import NextBarDirectionLabeler
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler
from kvant.ml_prepare_data.samplers.time_bar import TimeBarSampler
from kvant.ml_prepare_data.reporting import report_sampling_density, report_sampling_timeline
from kvant.ml_prepare_data.samplers.sampler_cumsum import FixedThresholdCUSUMBarSampler, TunedCUSUMBarSampler
from typing import Dict, Optional, List
from kvant.kdata.hf_minute_data import (
    get_ticker_data,
    DownloadedDatasetSplit,
    get_huggingface_top_20_normal_splits,
)
from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index

MARKET_DATA_COLUMNS = ("open", "high", "low", "close", "volume")


# ============================================================
# 3) Experiment config + stable id
# ============================================================
@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    sampler: dict
    feature_engineer: dict
    labeler: dict
    lookback_L: int
    feature_selector: Optional[dict] = None
    label_semantics: dict = field(default_factory=lambda: label_semantics_payload(drop_time_exit_label=False))

    def stable_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


# ============================================================
# 4) Utilities
# ============================================================


def valid_target_positions(labels: np.ndarray, lookback_L: int) -> np.ndarray:
    pos = np.arange(len(labels))
    return pos[(labels != -1) & (pos >= lookback_L)]


def _json_default(x):
    # Fallback serializer for json.dumps
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.datetime64,)):
        return str(pd.Timestamp(x))
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


def save_label_metadata_jsonl(tdir: Path, metadata: List[Optional[dict]]) -> None:
    """
    Writes one JSON value per row: either `null` or an object.
    Aligned by position with features/labels/timestamps.
    """
    path = tdir / "label_metadata.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, default=_json_default))
            f.write("\n")


def save_ticker_artifacts(
    tdir: Path,
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    meta: dict,
    market_data: Optional[np.ndarray] = None,
    label_metadata: Optional[list[Optional[dict]]] = None,
) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "features.npy", X.astype(np.float32, copy=False))
    np.save(tdir / "labels.npy", y.astype(np.int8, copy=False))
    np.save(tdir / "timestamps.npy", ts.astype("datetime64[ns]", copy=False))
    if market_data is not None:
        np.save(tdir / "market_data.npy", market_data.astype(np.float32, copy=False))
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if label_metadata is not None:
        if len(label_metadata) != len(y):
            raise RuntimeError(f"label_metadata length {len(label_metadata)} != labels length {len(y)}")
        save_label_metadata_jsonl(tdir, label_metadata)


def _feature_selector_payload(selector: FeatureSelector | None) -> Optional[dict]:
    if selector is None:
        return None
    getter = getattr(selector, "get_meta", None)
    if callable(getter):
        return getter()
    return asdict(selector)


def _subset_feature_engineer_payload(feature_payload: dict, selector: FeatureSelector | None) -> dict:
    if selector is None:
        return feature_payload

    selected_indices = getattr(selector, "selected_indices_", None)
    selected_feature_names = getattr(selector, "selected_feature_names_", None)
    original_feature_names = getattr(selector, "feature_names_", None)
    if selected_indices is None or selected_feature_names is None or original_feature_names is None:
        return feature_payload

    chosen = np.asarray(selected_indices, dtype=np.int64)
    original_len = int(len(original_feature_names))

    out = dict(feature_payload)
    out["feature_names_"] = list(selected_feature_names)
    for key in ("mean_", "std_", "n_samples_seen_", "sum_", "sumsq_"):
        value = out.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim != 1 or len(arr) != original_len:
            continue
        out[key] = arr[chosen].tolist()
    return out


def _config_payload(
    cfg: ExperimentConfig,
    fe: FeatureEngineer | None = None,
    selector: FeatureSelector | None = None,
) -> dict:
    payload = asdict(cfg)
    payload["pipeline_stage"] = "event_outcome"
    payload["label_spaces"] = pipeline_label_spaces_payload()
    if fe is not None:
        payload["feature_engineer"] = _subset_feature_engineer_payload(asdict(fe), selector)
    if selector is not None:
        payload["feature_selector"] = _feature_selector_payload(selector)
    return payload


def _as_dt64_utc_naive(x) -> np.datetime64:
    """
    Convert x (pd.Timestamp/np.datetime64/etc.) to UTC-naive np.datetime64[ns].
    """
    if x is None:
        return None
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    if isinstance(x, pd.Timestamp):
        # convert tz-aware -> UTC, then drop tz
        if x.tz is not None:
            x = x.tz_convert("UTC").tz_localize(None)
        return x.to_datetime64().astype("datetime64[ns]")
    # last resort
    return np.datetime64(pd.Timestamp(x, tz="UTC").tz_localize(None)).astype("datetime64[ns]")


def _first_ts_utc_dt64(df: pd.DataFrame) -> np.datetime64:
    df = ensure_utc_sorted_index(df)
    return _as_dt64_utc_naive(df.index[0])


def _concat_nonempty(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts2 = [p for p in parts if p is not None and len(p) > 0]
    if not parts2:
        return pd.DataFrame()
    if len(parts2) == 1:
        return ensure_utc_sorted_index(parts2[0])
    out = pd.concat([ensure_utc_sorted_index(p) for p in parts2], axis=0)
    # timestamps assumed strictly increasing per ticker; still keep sorted for safety
    out = out.sort_index()
    return out


def _in_split(tt, split: str, val_start, test_start) -> bool:
    tt = _as_dt64_utc_naive(tt)
    val_start = _as_dt64_utc_naive(val_start) if val_start is not None else None
    test_start = _as_dt64_utc_naive(test_start) if test_start is not None else None

    if split == "train":
        cut = val_start if val_start is not None else test_start
        return True if cut is None else (tt < cut)

    if split == "val":
        if val_start is None:
            return False
        if test_start is None:
            return tt >= val_start
        return (tt >= val_start) and (tt < test_start)

    if split == "test":
        if test_start is None:
            return False
        return tt >= test_start

    raise ValueError(split)


def _labeler_embargo_delta(labeler: Labeler) -> np.timedelta64:
    width_minutes = getattr(labeler, "width_minutes", 0)
    try:
        width_minutes = max(int(width_minutes), 0)
    except (TypeError, ValueError):
        width_minutes = 0
    return np.timedelta64(width_minutes, "m")


def _label_interval_from_metadata(
    metadata: Optional[dict],
    fallback_signal_ts,
) -> Optional[tuple[np.datetime64, np.datetime64]]:
    if not isinstance(metadata, dict):
        return None

    signal_ts = metadata.get("signal_time", fallback_signal_ts)
    close_ts = metadata.get("bar_close_time")
    if signal_ts is None or close_ts is None:
        return None

    try:
        signal_dt = _as_dt64_utc_naive(signal_ts)
        close_dt = _as_dt64_utc_naive(close_ts)
    except Exception:
        return None

    if close_dt < signal_dt:
        return None
    return signal_dt, close_dt


def _label_interval_is_safe_for_split(
    signal_ts,
    close_ts,
    split: str,
    val_start,
    test_start,
    embargo: np.timedelta64 = np.timedelta64(0, "ns"),
) -> bool:
    signal_ts = _as_dt64_utc_naive(signal_ts)
    close_ts = _as_dt64_utc_naive(close_ts)
    val_start = _as_dt64_utc_naive(val_start) if val_start is not None else None
    test_start = _as_dt64_utc_naive(test_start) if test_start is not None else None

    if close_ts < signal_ts:
        return False

    if split == "train":
        cut = val_start if val_start is not None else test_start
        if cut is None:
            return True
        return bool(signal_ts < (cut - embargo) and close_ts < cut)

    if split == "val":
        if val_start is None:
            return False
        if signal_ts < val_start:
            return False
        if test_start is None:
            return True
        return bool(signal_ts < (test_start - embargo) and close_ts < test_start)

    if split == "test":
        if test_start is None:
            return False
        return bool(signal_ts >= test_start)

    raise ValueError(split)


def _fit_feature_selector_on_train(
    *,
    tickers_train: list[str],
    minute_train_chunks: Dict[str, pd.DataFrame],
    sampler: BaseBarSampler,
    fe: FeatureEngineer,
    labeler: Labeler,
    lookback_L: int,
    feature_selector: FeatureSelector | None,
) -> FeatureSelector | None:
    if feature_selector is None:
        return None

    selector_X: list[np.ndarray] = []
    selector_y: list[np.ndarray] = []
    feature_names_ref: list[str] | None = None

    for ticker in tqdm.tqdm(tickers_train, desc="Fitting feature selector", dynamic_ncols=True):
        dft = minute_train_chunks.get(ticker)
        if dft is None or len(dft) == 0:
            continue

        dft_s = sampler.transform(dft, ticker=ticker)
        if dft_s is None or len(dft_s) == 0:
            continue
        dft_s = ensure_utc_sorted_index(dft_s)

        X_full, feat_names = fe.transform(dft)
        if feature_names_ref is None:
            feature_names_ref = list(feat_names)
        elif list(feat_names) != feature_names_ref:
            raise RuntimeError("Feature names changed between train chunks during feature-selection fit.")

        feat_df_full = pd.DataFrame(X_full, index=dft.index, columns=feat_names)
        X_sampled = feat_df_full.loc[dft_s.index].to_numpy(dtype=np.float32, copy=False)
        y_sampled, _ = labeler.transform(dft_s)
        valid_pos = valid_target_positions(y_sampled, lookback_L)
        if len(valid_pos) == 0:
            continue

        side_targets = event_labels_to_side_labels(y_sampled[valid_pos])
        keep = side_targets >= 0
        if not np.any(keep):
            continue

        selector_X.append(X_sampled[valid_pos][keep])
        selector_y.append(side_targets[keep].astype(np.int64, copy=False))

    if feature_names_ref is None or not selector_X:
        raise RuntimeError(
            "No train-only primary-side samples were available to fit the feature selector."
        )

    feature_selector.fit(
        np.concatenate(selector_X, axis=0),
        np.concatenate(selector_y, axis=0),
        feature_names=feature_names_ref,
    )
    return feature_selector


# ============================================================
# 5) Preparation Orchestrator
# ============================================================
@dataclass
class PreparedExperimentManifest:
    exp_dir: Path
    tickers_all: list[str]
    tickers_train: list[str]
    tickers_val: list[str]
    tickers_test: list[str]


def prepare_experiment(
    out_root: Path,
    cfg: ExperimentConfig,
    sampler: BaseBarSampler,
    fe: FeatureEngineer,
    labeler: Labeler,
    ticker_dfs_train: Dict[str, pd.DataFrame],
    ticker_dfs_val: Dict[str, pd.DataFrame],
    ticker_dfs_test: Dict[str, pd.DataFrame],
    experiment_id: str = None,  # Provide a stable id of the experiment.
    feature_selector: FeatureSelector | None = None,
) -> PreparedExperimentManifest:
    """
    Key behavior:
      1) Splits are manual and always provided: train/val/test dicts.
      2) For each ticker, concatenate (train + val + test) first.
      3) Apply feature engineering on the concatenated minute series first, then
         sample the already-computed features using the sampler timestamps so
         val/test can use training history causally (no leakage).
      4) Apply labeling on sampled OHLCV bars, then build train/val/test indices using per-ticker boundaries inferred
         from the first timestamp in val/test.

    Additional behavior in this version:
      - sampler.fit(...) is called explicitly on TRAIN ONLY (per-ticker tuning allowed).
      - sampler provides explicit metadata (global + per-ticker) that is persisted.
      - density + label distribution diagnostics are saved per ticker and as a global summary.
    """

    # -----------------------------
    # small local helpers
    # -----------------------------

    def _counts_by_split_for_ts(
        ts: np.ndarray,
        val_start: Optional[np.datetime64],
        test_start: Optional[np.datetime64],
    ) -> dict:
        out = {"train": 0, "val": 0, "test": 0}
        for tt in ts:
            if _in_split(tt, "train", val_start, test_start):
                out["train"] += 1
            elif _in_split(tt, "val", val_start, test_start):
                out["val"] += 1
            elif _in_split(tt, "test", val_start, test_start):
                out["test"] += 1
        return out

    def _bars_per_day(ts: np.ndarray) -> float:
        if ts is None or len(ts) == 0:
            return 0.0
        s = pd.to_datetime(ts)
        if getattr(s, "tz", None) is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        days = pd.Series(s).dt.normalize()
        n_days = int(days.nunique())
        if n_days <= 0:
            return 0.0
        return float(len(ts) / n_days)

    def _label_counts(y: np.ndarray) -> dict:
        if y is None or len(y) == 0:
            return {}
        u, c = np.unique(y, return_counts=True)
        return {str(int(uu)): int(cc) for uu, cc in zip(u, c)}

    def _label_counts_by_split(
        y: np.ndarray,
        ts: np.ndarray,
        val_start: Optional[np.datetime64],
        test_start: Optional[np.datetime64],
        *,
        only_valid_positions: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Returns:
          {
            "train": {"-1":..., "0":..., ...},
            "val":   {...},
            "test":  {...}
          }
        """
        out = {"train": {}, "val": {}, "test": {}}

        if only_valid_positions is None:
            positions = range(len(y))
        else:
            positions = (int(p) for p in only_valid_positions)

        for p in positions:
            lab = int(y[p])
            tt = ts[p]
            if _in_split(tt, "train", val_start, test_start):
                bucket = "train"
            elif _in_split(tt, "val", val_start, test_start):
                bucket = "val"
            elif _in_split(tt, "test", val_start, test_start):
                bucket = "test"
            else:
                continue
            k = str(lab)
            out[bucket][k] = out[bucket].get(k, 0) + 1

        return out

    # -----------------------------
    # experiment id + dirs
    # -----------------------------
    exp_id = cfg.stable_id() if experiment_id is None else experiment_id
    exp_dir = out_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(_config_payload(cfg, selector=feature_selector), indent=2, default=str))

    tickers_train = sorted(ticker_dfs_train.keys())
    tickers_val = sorted(ticker_dfs_val.keys())
    tickers_test = sorted(ticker_dfs_test.keys())
    tickers_all = sorted(set(tickers_train) | set(tickers_val) | set(tickers_test))
    print(
        f"[{exp_id}] Starting prepare_experiment "
        f"(train={len(tickers_train)}, val={len(tickers_val)}, test={len(tickers_test)}, all={len(tickers_all)})"
    )

    (exp_dir / "tickers_all.json").write_text(json.dumps(tickers_all, indent=2))
    (exp_dir / "tickers_train.json").write_text(json.dumps(tickers_train, indent=2))
    (exp_dir / "tickers_val.json").write_text(json.dumps(tickers_val, indent=2))
    (exp_dir / "tickers_test.json").write_text(json.dumps(tickers_test, indent=2))

    ticker_id = {t: i for i, t in enumerate(tickers_all)}
    tickers_root = exp_dir / "tickers"
    tickers_root.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Infer per-ticker boundaries from provided split dicts
    # --------------------------------------------------------
    boundaries: dict[str, tuple[Optional[np.datetime64], Optional[np.datetime64]]] = {}
    for t in tickers_all:
        val_start = _first_ts_utc_dt64(ticker_dfs_val[t]) if t in ticker_dfs_val and len(ticker_dfs_val[t]) else None
        test_start = (
            _first_ts_utc_dt64(ticker_dfs_test[t]) if t in ticker_dfs_test and len(ticker_dfs_test[t]) else None
        )
        boundaries[t] = (val_start, test_start)
        # --------------------------------------------------------
        # Fit on TRAIN ONLY
        #   - sampler: explicit per-ticker tuning allowed (no-op in IdentitySampler)
        #   - feature engineer: fit on minute-resolution train data
        #   - labeler: fit on sampled train bars (currently a no-op)
        # --------------------------------------------------------

        # 1) Tune sampler on TRAIN ONLY (per-ticker tuning handled internally).
    print(f"[{exp_id}] Fitting sampler on train tickers...")
    sampler.fit(ticker_dfs_train)

    # 2) Persist sampler metadata
    sampler_global_meta = sampler.get_global_meta()
    sampler_per_ticker_meta = {t: sampler.get_ticker_meta(t) for t in tickers_all}
    (exp_dir / "sampler_global_meta.json").write_text(json.dumps(sampler_global_meta, indent=2, default=_json_default))
    (exp_dir / "sampler_per_ticker_meta.json").write_text(
        json.dumps(sampler_per_ticker_meta, indent=2, default=_json_default)
    )

    minute_train_chunks: dict[str, pd.DataFrame] = {}

    # 3) Collect full minute-resolution train chunks for feature fitting.
    print(f"[{exp_id}] Collecting minute train data for feature fitting...")
    sampled_train_chunks: list[pd.DataFrame] = []
    for ticker in tqdm.tqdm(tickers_train, desc="Collecting minute train chunks", dynamic_ncols=True):
        dft = ticker_dfs_train.get(ticker)
        if dft is None or len(dft) == 0:
            continue
        dft = ensure_utc_sorted_index(dft)
        minute_train_chunks[ticker] = dft

    if not minute_train_chunks:
        raise RuntimeError("No minute-resolution training rows available to fit feature engineer.")

    # 4) Fit FE on minute train data, then fit the labeler on sampled train bars.
    print(f"[{exp_id}] Fitting feature engineer and labeler...")
    if hasattr(fe, "fit_many"):
        fe.fit_many(minute_train_chunks[ticker] for ticker in tickers_train if ticker in minute_train_chunks)
    else:
        df_fit_minute = pd.concat(
            [minute_train_chunks[ticker] for ticker in tickers_train if ticker in minute_train_chunks],
            axis=0,
        )
        fe.fit(df_fit_minute)

    for ticker in tqdm.tqdm(tickers_train, desc="Sampling train chunks", dynamic_ncols=True):
        dft = minute_train_chunks.get(ticker)
        if dft is None or len(dft) == 0:
            continue
        dft_s = sampler.transform(dft, ticker=ticker)
        if dft_s is None or len(dft_s) == 0:
            continue
        dft_s = ensure_utc_sorted_index(dft_s)
        sampled_train_chunks.append(dft_s)

    df_fit_sampled = _concat_nonempty(sampled_train_chunks)

    if len(df_fit_sampled) == 0:
        raise RuntimeError(
            "No sampled training rows available to fit labeler. "
            "This usually means your sampler is too sparse or train data is empty."
        )

    labeler.fit(df_fit_sampled)
    feature_selector = _fit_feature_selector_on_train(
        tickers_train=tickers_train,
        minute_train_chunks=minute_train_chunks,
        sampler=sampler,
        fe=fe,
        labeler=labeler,
        lookback_L=cfg.lookback_L,
        feature_selector=feature_selector,
    )
    (exp_dir / "config.json").write_text(json.dumps(_config_payload(cfg, fe, feature_selector), indent=2, default=str))
    # --------------------------------------------------------
    # Process each ticker on full history (train+val+test)
    # --------------------------------------------------------
    valid_pos_by_ticker: Dict[str, np.ndarray] = {}
    label_metadata_by_ticker: Dict[str, list[Optional[dict]]] = {}
    label_interval_embargo = _labeler_embargo_delta(labeler)

    # global diagnostics accumulator
    density_summary_rows: list[dict] = []
    sampling_examples: list[dict] = []

    print(f"[{exp_id}] Preparing ticker artifacts...")
    for t in tqdm.tqdm(tickers_all, desc="Preparing tickers", dynamic_ncols=True):
        df_full_raw = _concat_nonempty(
            [
                ticker_dfs_train.get(t),
                ticker_dfs_val.get(t),
                ticker_dfs_test.get(t),
            ]
        )
        df_full_raw.sort_index(inplace=True)
        assert df_full_raw.index.is_monotonic_increasing == 1

        if len(df_full_raw) == 0:
            raise RuntimeError(f"Ticker {t} has no rows across train/val/test.")

        val_start, test_start = boundaries[t]

        # Raw density (before sampling)
        ts_raw = df_full_raw.index.to_numpy()
        raw_counts_by_split = _counts_by_split_for_ts(ts_raw, val_start, test_start)

        # Sampled OHLCV bars for labeling/backtesting
        df1 = sampler.transform(df_full_raw, ticker=t)
        df1 = ensure_utc_sorted_index(df1)

        # Features are computed on the full minute-resolution dataframe first,
        # then sampled at the timestamps selected by the sampler.
        X_full, feat_names = fe.transform(df_full_raw)
        if feature_selector is not None:
            X_full, feat_names = feature_selector.transform(X_full, feat_names)
        feat_df_full = pd.DataFrame(X_full, index=df_full_raw.index, columns=feat_names)
        try:
            feat_df_sampled = feat_df_full.loc[df1.index]
        except KeyError as exc:
            missing = sorted(set(df1.index).difference(set(feat_df_full.index)))
            missing_preview = [str(x) for x in missing[:5]]
            raise RuntimeError(
                f"Sampled timestamps for {t} were not found in the minute-resolution feature dataframe. "
                f"Examples: {missing_preview}"
            ) from exc

        X = feat_df_sampled.to_numpy(dtype=np.float32, copy=False)
        y, y_meta = labeler.transform(df1)
        label_metadata_by_ticker[t] = y_meta

        if len(X) != len(y):
            raise RuntimeError(f"Length mismatch for {t}: features={len(X)} labels={len(y)}")

        ts = df1.index.to_numpy()
        valid_pos = valid_target_positions(y, cfg.lookback_L)
        valid_pos_by_ticker[t] = valid_pos

        # Sampled density
        sampled_counts_by_split = _counts_by_split_for_ts(ts, val_start, test_start)

        # label distributions (overall + valid targets)
        y_counts_all = _label_counts(y)
        y_counts_valid = _label_counts(y[valid_pos] if len(valid_pos) else np.asarray([], dtype=y.dtype))

        y_counts_all_by_split = _label_counts_by_split(y, ts, val_start, test_start, only_valid_positions=None)
        y_counts_valid_by_split = _label_counts_by_split(y, ts, val_start, test_start, only_valid_positions=valid_pos)

        # Split-valid target counts after purging label intervals that cross
        # chronological boundaries and embargoing the period before the next split.
        split_safe_counts = {"train": 0, "val": 0, "test": 0}
        split_purged_counts = {"train": 0, "val": 0, "test": 0}
        for p in valid_pos:
            p = int(p)
            tt = ts[p]
            raw_split = None
            if _in_split(tt, "train", val_start, test_start):
                raw_split = "train"
            elif _in_split(tt, "val", val_start, test_start):
                raw_split = "val"
            elif _in_split(tt, "test", val_start, test_start):
                raw_split = "test"

            interval = _label_interval_from_metadata(y_meta[p], tt)
            if interval is not None and raw_split is not None:
                signal_ts, close_ts = interval
                if _label_interval_is_safe_for_split(
                    signal_ts,
                    close_ts,
                    raw_split,
                    val_start,
                    test_start,
                    embargo=label_interval_embargo,
                ):
                    split_safe_counts[raw_split] += 1
                else:
                    split_purged_counts[raw_split] += 1

        # Membership of ticker in the provided split dicts (not time membership).
        membership = []
        if t in ticker_dfs_train:
            membership.append("train")
        if t in ticker_dfs_val:
            membership.append("val")
        if t in ticker_dfs_test:
            membership.append("test")

        n_raw_full = int(len(df_full_raw))
        n_sampled_full = int(len(df1))
        retention = float(n_sampled_full / n_raw_full) if n_raw_full > 0 else 0.0

        density_row = {
            "ticker": t,
            "n_raw_full": n_raw_full,
            "n_sampled_full": n_sampled_full,
            "retention_ratio": retention,
            "bars_per_day_raw": _bars_per_day(ts_raw),
            "bars_per_day_sampled": _bars_per_day(ts),
            "raw_counts_by_split": raw_counts_by_split,
            "sampled_counts_by_split": sampled_counts_by_split,
            "sampler_ticker_meta": sampler.get_ticker_meta(t),
        }
        density_summary_rows.append(density_row)
        sampling_examples.append(
            {
                "ticker": t,
                "raw_timestamps": ts_raw.copy(),
                "raw_close": df_full_raw["close"].to_numpy(dtype=np.float64, copy=True),
                "sampled_timestamps": ts.copy(),
                "sampled_close": df1["close"].to_numpy(dtype=np.float64, copy=True),
                "val_start": val_start,
                "test_start": test_start,
                "n_raw_full": n_raw_full,
                "n_sampled_full": n_sampled_full,
                "retention_ratio": retention,
                "bars_per_day_raw": density_row["bars_per_day_raw"],
                "bars_per_day_sampled": density_row["bars_per_day_sampled"],
            }
        )

        meta = {
            "ticker": t,
            "membership": membership,
            "feature_names": feat_names,
            "selected_feature_names": None
            if feature_selector is None
            else list(getattr(feature_selector, "selected_feature_names_", feat_names)),
            "market_data_columns": list(MARKET_DATA_COLUMNS),
            "sampler_name": sampler.name,
            "sampler_global_meta": sampler_global_meta,
            "sampler_ticker_meta": sampler.get_ticker_meta(t),
            # density diagnostics
            "n_rows_raw_full": n_raw_full,
            "n_rows_sampled_full": n_sampled_full,
            "retention_ratio": retention,
            "bars_per_day_raw": density_row["bars_per_day_raw"],
            "bars_per_day_sampled": density_row["bars_per_day_sampled"],
            "raw_counts_by_split": raw_counts_by_split,
            "sampled_counts_by_split": sampled_counts_by_split,
            # labeling diagnostics
            "label_counts_all": y_counts_all,
            "label_counts_valid_targets": y_counts_valid,
            "label_counts_all_by_split": y_counts_all_by_split,
            "label_counts_valid_targets_by_split": y_counts_valid_by_split,
            "label_interval_embargo_minutes": int(label_interval_embargo / np.timedelta64(1, "m")),
            "label_interval_purged_valid_targets_by_split": split_purged_counts,
            # existing info
            "n_valid_targets_full": int(len(valid_pos)),
            "val_start_ts": None if val_start is None else str(pd.Timestamp(val_start, tz="UTC")),
            "test_start_ts": None if test_start is None else str(pd.Timestamp(test_start, tz="UTC")),
            "n_valid_train": int(split_safe_counts["train"]),
            "n_valid_val": int(split_safe_counts["val"]),
            "n_valid_test": int(split_safe_counts["test"]),
        }

        market_data = df1.loc[:, list(MARKET_DATA_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
        save_ticker_artifacts(tickers_root / t, X, y, ts, meta, market_data=market_data, label_metadata=y_meta)

    # Persist global density summary
    (exp_dir / "density_summary.json").write_text(json.dumps(density_summary_rows, indent=2, default=_json_default))

    # --------------------------------------------------------
    # Build indices for train/val/test using inferred boundaries
    # --------------------------------------------------------
    def build_index_for_tickers(tickers: list[str], split: str) -> np.ndarray:
        out = []
        for t in tqdm.tqdm(tickers, desc=f"Building {split} index", dynamic_ncols=True):
            ts = np.load(tickers_root / t / "timestamps.npy", mmap_mode="r")
            valid_pos = valid_pos_by_ticker[t]
            label_metadata = label_metadata_by_ticker[t]
            tid = ticker_id[t]
            val_start, test_start = boundaries[t]

            for p in valid_pos:
                p = int(p)
                tt = ts[p]
                interval = _label_interval_from_metadata(label_metadata[p], tt)
                if interval is None:
                    continue
                signal_ts, close_ts = interval
                if _label_interval_is_safe_for_split(
                    signal_ts,
                    close_ts,
                    split,
                    val_start,
                    test_start,
                    embargo=label_interval_embargo,
                ):
                    out.append((tid, p))

        return np.asarray(out, dtype=np.int32)

    index_train = build_index_for_tickers(tickers_train, "train")
    index_val = build_index_for_tickers(tickers_val, "val")
    index_test = build_index_for_tickers(tickers_test, "test")

    np.save(exp_dir / "index_train.npy", index_train)
    np.save(exp_dir / "index_val.npy", index_val)
    np.save(exp_dir / "index_test.npy", index_test)

    report_sampling_timeline(exp_dir, sampling_examples=sampling_examples, max_tickers=4)

    print(f"[{exp_id}] Finished preparing experiment.")
    print("Prepared indices:")
    print("  train:", len(index_train))
    print("  val:", len(index_val))
    print("  test:", len(index_test))

    return PreparedExperimentManifest(
        exp_dir=exp_dir,
        tickers_all=tickers_all,
        tickers_train=tickers_train,
        tickers_val=tickers_val,
        tickers_test=tickers_test,
    )


def prepare_single_dataset(
    dataset_split: DownloadedDatasetSplit,
    sampler,
    feature_engineer,
    labeler,
    L=64,
    feature_selector: FeatureSelector | None = None,
):
    ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data(dataset_split)

    # sampler = IdentitySampler(subsample_every=1)
    # fe = OHLCVFeatures(cols=("open", "high", "low", "close", "volume"), log1p_volume=True)
    # labeler = TripleBarrierLabeler(name="tb_w60_h2pct", width_minutes=60, height=0.02, drop_time_exit_label=True)
    # L = 200
    cfg = ExperimentConfig(
        experiment_name="exp_minimal_sep_components",
        sampler=asdict(sampler),
        feature_engineer=asdict(feature_engineer),
        feature_selector=None if feature_selector is None else asdict(feature_selector),
        labeler=asdict(labeler),
        lookback_L=L,
        label_semantics=label_semantics_payload(drop_time_exit_label=False),
    )
    from kvant.ml_prepare_data import prepared_data_root

    prepared = prepare_experiment(
        out_root=prepared_data_root,
        cfg=cfg,
        sampler=sampler,
        fe=feature_engineer,
        labeler=labeler,
        ticker_dfs_train=ticker_data_train,
        ticker_dfs_val=ticker_data_val,
        ticker_dfs_test=ticker_data_test,
        feature_selector=feature_selector,
    )
    print("Experiment prepared at:", prepared.exp_dir)
    return prepared


# ============================================================
# 6) Minimal runnable main (plug in your data loader)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare walk-forward kvant experiment artifacts.")
    parser.add_argument("--sampler", choices=("tuned_cusum", "fixed_cusum", "time_bar"), default="tuned_cusum")
    parser.add_argument("--target-bars-per-day", type=float, default=30.0)
    parser.add_argument("--time-bar-minutes", type=int, default=15, help="Aggregate to k-minute bars (for --sampler time_bar)")
    parser.add_argument("--cusum-h", type=float, default=0.01)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--barrier-width", type=int, default=180)
    parser.add_argument("--barrier-height-pct", type=float, default=1.5)
    parser.add_argument("--labeler", choices=("triple_barrier", "next_bar"), default="triple_barrier")
    parser.add_argument("--cv-manifest", type=str, default=None, help="Path to write CV manifest JSON")
    args = parser.parse_args()

    downloaded_splits = get_huggingface_top_20_normal_splits()

    TBPD = float(args.target_bars_per_day)
    L, width, height_pct = int(args.lookback), int(args.barrier_width), float(args.barrier_height_pct)
    time_bar_minutes = int(args.time_bar_minutes)

    # Sampler tag for label naming
    if args.sampler == "fixed_cusum":
        sampler_tag = f"fixedCUSUM{float(args.cusum_h):g}"
    elif args.sampler == "time_bar":
        sampler_tag = f"timebar{time_bar_minutes}m"
    else:
        sampler_tag = f"TBPD{TBPD:g}"

    # Labeler tag for label naming
    if args.labeler == "next_bar":
        labeler_tag = "nextbar"
        drop_time_exit_label = False
    else:
        labeler_tag = f"h{height_pct:g}"
        drop_time_exit_label = False

    label_suffix = "_droptexit" if drop_time_exit_label else ""
    label = f"sb_L_{L}_w{width}_{labeler_tag}_{sampler_tag}{label_suffix}"
    print(f"Writing to {label=}")

    from kvant.ml_prepare_data import prepared_data_root

    cv_rows = []
    last_prepared = None
    for fold_idx, split in enumerate(downloaded_splits):
        print(f"\nPreparing fold {fold_idx + 1}/{len(downloaded_splits)}")
        ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data(split)

        # Instantiate sampler
        if args.sampler == "fixed_cusum":
            sampler = FixedThresholdCUSUMBarSampler(h=float(args.cusum_h), aggregate_ohlcv=True)
        elif args.sampler == "time_bar":
            sampler = TimeBarSampler(time_bar_minutes=time_bar_minutes)
        else:
            sampler = TunedCUSUMBarSampler(target_bars_per_day=TBPD, aggregate_ohlcv=True)

        base_fe = IntradayTA10Features(
            volume_output="log1p",
            include_time_features=True,
            typical_bar_minutes=None,  # periods in bars (paper style)
            fillna_value=0.0,
        )
        fe = StandardizedFeatures(base=base_fe)
        feature_selector = PrimarySideFScoreSelector(top_k=16)

        # Instantiate labeler
        if args.labeler == "next_bar":
            labeler = NextBarDirectionLabeler(name=label)
        else:
            labeler = TripleBarrierLabeler(
                name=label, width_minutes=width, height=height_pct / 100, drop_time_exit_label=drop_time_exit_label
            )

        cfg = ExperimentConfig(
            experiment_name="exp_minimal_sep_components",
            sampler=asdict(sampler),
            feature_engineer=asdict(fe),
            feature_selector=asdict(feature_selector),
            labeler=asdict(labeler),
            lookback_L=L,
            label_semantics=label_semantics_payload(drop_time_exit_label=False),
        )

        fold_id = f"{label}_fold{fold_idx:02d}"
        prepared = prepare_experiment(
            out_root=prepared_data_root,
            cfg=cfg,
            sampler=sampler,
            fe=fe,
            labeler=labeler,
            ticker_dfs_train=ticker_data_train,
            ticker_dfs_val=ticker_data_val,
            ticker_dfs_test=ticker_data_test,
            experiment_id=fold_id,
            feature_selector=feature_selector,
        )
        report_sampling_density(prepared.exp_dir, bins=60, print_table=True, max_plot_tickers=4)
        print("Experiment prepared at:", prepared.exp_dir)
        last_prepared = prepared

        cv_rows.append(
            {
                "fold_idx": int(fold_idx),
                "exp_id": str(prepared.exp_dir.name),
                "exp_dir": str(prepared.exp_dir.resolve()),
                "year_quarter_train": split.split.year_quarter_train,
                "year_quarter_val": split.split.year_quarter_val,
                "year_quarter_test": split.split.year_quarter_test,
            }
        )

    # Use --cv-manifest if provided, otherwise use default naming
    if args.cv_manifest:
        manifest_path = Path(args.cv_manifest)
    else:
        manifest_path = prepared_data_root / f"{label}_cv_manifest.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"label": label, "n_folds": len(cv_rows), "folds": cv_rows}, indent=2))
    print(f"Wrote CV manifest to {manifest_path}")

    if last_prepared is not None:
        with open(prepared_data_root / "last_experiment.txt", "w") as f:
            f.write(last_prepared.exp_dir.name)
            print("Wrote name to", last_prepared.exp_dir.name)
        with open(prepared_data_root / "last_experiment_cv_manifest.txt", "w") as f:
            f.write(str(manifest_path))
            print("Wrote CV manifest pointer to", manifest_path)


if __name__ == "__main__":
    main()
