from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from kvant.labels import (
    LABEL_UP,
    class_names_from_semantics,
    event_label_from_metadata,
    event_labels_to_side_labels,
    label_ids_from_semantics,
    label_meanings_from_semantics,
    validate_label_semantics,
)


def _load_jsonl(path: Path) -> List[Optional[dict]]:
    """
    Loads a JSONL file where each line is either:
      - "null"  -> None
      - JSON object -> dict
    """
    out: List[Optional[dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _to_dt64_utc_naive(value: Any) -> np.datetime64:
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts, "ns")


class PreparedStore:
    MARKET_DATA_COLUMNS = ("open", "high", "low", "close", "volume")

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.cfg = json.loads((exp_dir / "config.json").read_text())
        self.label_spaces = dict(self.cfg.get("label_spaces") or {})
        self.label_semantics = validate_label_semantics(self.cfg, exp_dir=exp_dir)
        self.label_ids = label_ids_from_semantics(self.label_semantics)
        self.label_meanings = label_meanings_from_semantics(self.label_semantics)
        self.class_names = class_names_from_semantics(self.label_semantics)
        self.n_classes = len(self.label_ids)
        self.pipeline_stage = str(
            self.cfg.get("pipeline_stage")
            or ("event_outcome" if self.n_classes == 3 else "legacy")
        )
        feature_engineer_cfg = (self.cfg.get("feature_engineer") or {}) if isinstance(self.cfg, dict) else {}
        feature_names = feature_engineer_cfg.get("feature_names_")
        self.feature_names = tuple(str(name) for name in feature_names) if feature_names else None
        self.feature_name_to_index = (
            {name: idx for idx, name in enumerate(self.feature_names)} if self.feature_names is not None else {}
        )
        self.tickers_all = json.loads((exp_dir / "tickers_all.json").read_text())
        self.ticker_to_id = {t: i for i, t in enumerate(self.tickers_all)}

        self._features: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._timestamps: List[np.ndarray] = []
        self._label_metadata: List[List[Optional[dict]]] = []
        self._market_data: List[Optional[np.ndarray]] = []

        for t in self.tickers_all:
            tdir = exp_dir / "tickers" / t
            X = np.load(tdir / "features.npy", mmap_mode="r")
            y = np.load(tdir / "labels.npy", mmap_mode="r")
            ts = np.load(tdir / "timestamps.npy", mmap_mode="r")
            market_data_path = tdir / "market_data.npy"
            market_data = np.load(market_data_path, mmap_mode="r") if market_data_path.exists() else None

            meta_path = tdir / "label_metadata.jsonl"
            if meta_path.exists():
                md = _load_jsonl(meta_path)
                if len(md) != len(y):
                    raise RuntimeError(f"{t}: label_metadata length {len(md)} != labels length {len(y)}")
            else:
                md = [None] * int(len(y))

            self._features.append(X)
            self._labels.append(y)
            self._timestamps.append(ts)
            self._label_metadata.append(md)
            self._market_data.append(market_data)

        self.n_features = int(self._features[0].shape[1])
        self.has_market_data = all(data is not None for data in self._market_data)

    def ticker(self, tid: int) -> str:
        """Return the ticker symbol for a numeric ticker id."""
        return str(self.tickers_all[int(tid)])

    def timestamp(self, tid: int, tpos: int):
        """Return the timestamp for a ticker/position pair."""
        return self._timestamps[int(tid)][int(tpos)]

    def window_and_label(self, tid: int, tpos: int, L: int) -> Tuple[np.ndarray, int]:
        X = self._features[tid]
        y = self._labels[tid]
        x_win = X[tpos - L : tpos, :]  # (L,F)
        label = int(y[tpos])
        return x_win, label

    def event_label(self, tid: int, tpos: int) -> int:
        return event_label_from_metadata(self.metadata(tid, tpos), fallback=int(self._labels[int(tid)][int(tpos)]))

    def event_labels_for_index(self, index: np.ndarray) -> np.ndarray:
        out = np.empty(int(index.shape[0]), dtype=np.int64)
        for i in range(int(index.shape[0])):
            tid = int(index[i, 0])
            tpos = int(index[i, 1])
            out[i] = self.event_label(tid, tpos)
        return out

    def side_label(self, tid: int, tpos: int) -> int:
        return int(event_labels_to_side_labels([self.event_label(tid, tpos)])[0])

    def side_labels_for_index(self, index: np.ndarray) -> np.ndarray:
        return event_labels_to_side_labels(self.event_labels_for_index(index))

    def metadata(self, tid: int, tpos: int) -> Optional[dict]:
        return self._label_metadata[tid][tpos]

    def metadata_for_index(self, index: np.ndarray) -> List[Optional[dict]]:
        """
        index: (N,2) array of (tid, tpos). Returns list[Optional[dict]] aligned to rows.
        """
        out: List[Optional[dict]] = []
        for i in range(int(index.shape[0])):
            tid = int(index[i, 0])
            tpos = int(index[i, 1])
            out.append(self.metadata(tid, tpos))
        return out

    def market_data(self, tid: int) -> Optional[Dict[str, np.ndarray]]:
        """Return sampled raw OHLCV arrays for a ticker when available."""
        market_data = self._market_data[int(tid)]
        if market_data is None:
            return None

        return {
            "timestamp": self._timestamps[int(tid)],
            **{
                column: market_data[:, idx]
                for idx, column in enumerate(self.MARKET_DATA_COLUMNS)
            },
        }

    def require_market_data(self, tid: int) -> Dict[str, np.ndarray]:
        """Return sampled raw OHLCV arrays for a ticker or raise a clear error."""
        market_data = self.market_data(tid)
        if market_data is None:
            ticker = self.ticker(tid)
            raise RuntimeError(
                f"Prepared market data is missing for ticker {ticker}. "
                "Regenerate the prepared experiment so market_data.npy is persisted for backtesting."
            )
        return market_data

    def require_feature_names(self) -> tuple[str, ...]:
        if self.feature_names is None:
            raise RuntimeError(
                "Prepared experiment is missing persisted feature names. "
                "Regenerate the prepared data before using prepared_last:<feature_name> decision features."
            )
        return self.feature_names

    def feature_index(self, feature_name: str) -> int:
        self.require_feature_names()
        feature_name = self.resolve_feature_name(feature_name)
        if feature_name not in self.feature_name_to_index:
            raise RuntimeError(f"Unknown prepared feature name {feature_name!r}.")
        return int(self.feature_name_to_index[feature_name])

    def resolve_feature_name(self, feature_name: str) -> str:
        self.require_feature_names()
        aliases = {
            "volatility": ("ewmstd_close_20b", "ewmstd_close_10b", "ewmstd_close_15b", "ewmstd_close_50b", "bb_width"),
            "volatility_feature": ("ewmstd_close_20b", "ewmstd_close_10b", "ewmstd_close_15b", "ewmstd_close_50b", "bb_width"),
            "recent_return": ("logret_1",),
            "recent_return_feature": ("logret_1",),
        }
        key = str(feature_name).strip()
        for candidate in aliases.get(key, (key,)):
            if candidate in self.feature_name_to_index:
                return candidate
        if key in aliases:
            raise RuntimeError(
                f"Prepared feature alias {key!r} could not be resolved. "
                f"Tried {list(aliases[key])}; available features are {list(self.feature_names or ())}."
            )
        return key

    def prepared_last_feature_values(self, tids: np.ndarray, tpos: np.ndarray, feature_name: str) -> np.ndarray:
        feature_idx = self.feature_index(feature_name)
        out = np.empty(len(tids), dtype=np.float32)
        for i, (tid, pos) in enumerate(zip(tids, tpos)):
            row_idx = int(pos) - 1
            if row_idx < 0:
                raise RuntimeError(
                    f"Cannot read prepared_last:{feature_name} for ticker id {int(tid)} at position {int(pos)}."
                )
            out[i] = float(self._features[int(tid)][row_idx, feature_idx])
        return out

    def time_since_last_event_minutes(self, tids: np.ndarray, tpos: np.ndarray) -> np.ndarray:
        out = np.zeros(len(tids), dtype=np.float64)
        for i, (tid, pos) in enumerate(zip(tids, tpos)):
            tid = int(tid)
            pos = int(pos)
            if pos <= 0:
                out[i] = 0.0
                continue
            current_ts = _to_dt64_utc_naive(self._timestamps[tid][pos])
            previous_ts = _to_dt64_utc_naive(self._timestamps[tid][pos - 1])
            delta = (current_ts - previous_ts) / np.timedelta64(1, "m")
            out[i] = float(max(delta, 0.0))
        return out

    def ticker_rolling_trade_stats(
        self,
        *,
        tids: np.ndarray,
        tpos: np.ndarray,
        trade_labels: np.ndarray,
        window: int,
    ) -> dict[str, np.ndarray]:
        window = int(window)
        if window <= 0:
            raise ValueError("window must be positive.")

        rolling_win_rate = np.full(len(tids), 0.5, dtype=np.float64)
        directional_win_rate = np.full(len(tids), 0.5, dtype=np.float64)
        recent_net_return = np.zeros(len(tids), dtype=np.float64)

        for i, (tid_raw, pos_raw, trade_label_raw) in enumerate(zip(tids, tpos, trade_labels)):
            tid = int(tid_raw)
            pos = int(pos_raw)
            trade_label = int(trade_label_raw)
            current_meta = self.metadata(tid, pos)
            signal_ts = (
                _to_dt64_utc_naive(current_meta.get("signal_time"))
                if isinstance(current_meta, dict) and current_meta.get("signal_time") is not None
                else _to_dt64_utc_naive(self._timestamps[tid][pos])
            )

            long_returns: list[float] = []
            side_returns: list[float] = []
            for previous_pos in range(pos - 1, -1, -1):
                previous_meta = self.metadata(tid, previous_pos)
                if not isinstance(previous_meta, dict):
                    continue
                close_time = previous_meta.get("bar_close_time")
                pnl_fraction = previous_meta.get("pnl_fraction")
                if close_time is None or pnl_fraction is None:
                    continue
                try:
                    close_ts = _to_dt64_utc_naive(close_time)
                    pnl = float(pnl_fraction)
                except Exception:
                    continue
                if close_ts >= signal_ts:
                    continue

                long_returns.append(pnl)
                side_returns.append(pnl if trade_label == LABEL_UP else -pnl)
                if len(side_returns) >= window:
                    break

            if long_returns:
                long_arr = np.asarray(long_returns, dtype=np.float64)
                side_arr = np.asarray(side_returns, dtype=np.float64)
                rolling_win_rate[i] = float(np.mean(long_arr > 0.0))
                directional_win_rate[i] = float(np.mean(side_arr > 0.0))
                recent_net_return[i] = float(np.mean(side_arr))

        return {
            "rolling_win_rate": rolling_win_rate,
            "directional_win_rate": directional_win_rate,
            "recent_net_return": recent_net_return,
        }


class IndexWindowDataset(Dataset):
    def __init__(self, store: PreparedStore, index: np.ndarray, lookback_L: int, *, target_mode: str = "event_outcome"):
        self.store = store
        self.index = index
        self.L = int(lookback_L)
        self.target_mode = str(target_mode)
        if self.target_mode not in {"event_outcome", "primary_side"}:
            raise ValueError(f"Unsupported target_mode={self.target_mode!r}.")

    @property
    def label_ids(self) -> tuple[int, ...]:
        if self.target_mode == "primary_side":
            return (0, 1)
        return tuple(self.store.label_ids)

    def _target_label(self, tid: int, tpos: int) -> int:
        if self.target_mode == "primary_side":
            return self.store.side_label(tid, tpos)
        return self.store.event_label(tid, tpos)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        tid, tpos = int(self.index[i, 0]), int(self.index[i, 1])
        x_win, _ = self.store.window_and_label(tid, tpos, self.L)
        y = self._target_label(tid, tpos)

        x_np = np.array(x_win, dtype=np.float32, copy=True)  # (L, F)
        x_t = torch.from_numpy(x_np.T).contiguous()  # (F, L)

        y_t = torch.as_tensor(y, dtype=torch.long)

        tid_t = torch.tensor(tid, dtype=torch.int32)
        tpos_t = torch.tensor(tpos, dtype=torch.int32)
        return x_t, y_t, tid_t, tpos_t

    def get_id(self, i: int) -> tuple[int, int]:
        """i is the global index of a sample, i.e., dataset[i]."""
        tid, tpos = int(self.index[i, 0]), int(self.index[i, 1])
        return tid, tpos

    def get_info(self, i: int) -> dict:
        tid, tpos = self.get_id(i)
        return {
            "tid": tid,
            "tpos": tpos,
            "ticker": self.store.ticker(tid),
            "timestamp": self.store.timestamp(tid, tpos),
            "label_metadata": self.store.metadata(tid, tpos),
        }

    def summary(self, display: bool = True):
        if self.target_mode == "primary_side":
            from kvant.ml_prepare_data.data_loading_utils import _print_plain_summary

            label_ids = self.label_ids
            if self.index is None or int(self.index.shape[0]) == 0:
                out = {
                    "overall": {
                        "n": 0,
                        "y_counts": {label: 0 for label in label_ids},
                        "first_ts": None,
                        "last_ts": None,
                    },
                    "per_ticker": {},
                }
                if display:
                    print("(empty dataset)")
                return out

            tids = self.index[:, 0].astype(np.int64, copy=False)
            tposs = self.index[:, 1].astype(np.int64, copy=False)
            per_ticker: Dict[str, Any] = {}
            overall_counts = {label: 0 for label in label_ids}
            overall_first_ts = None
            overall_last_ts = None

            for tid in np.unique(tids):
                mask = tids == tid
                pos = tposs[mask]
                ticker = self.store.ticker(int(tid))
                side_labels = self.store.side_labels_for_index(self.index[mask])
                valid_labels = side_labels[side_labels >= 0]
                counts = {label: int(np.sum(valid_labels == label)) for label in label_ids}
                for label in label_ids:
                    overall_counts[label] += counts[label]
                ts_arr = self.store._timestamps[int(tid)]
                first_ts = ts_arr[int(pos.min())]
                last_ts = ts_arr[int(pos.max())]
                if overall_first_ts is None or first_ts < overall_first_ts:
                    overall_first_ts = first_ts
                if overall_last_ts is None or last_ts > overall_last_ts:
                    overall_last_ts = last_ts
                per_ticker[ticker] = {
                    "tid": int(tid),
                    "n": int(mask.sum()),
                    "y_counts": counts,
                    "first_ts": str(np.datetime_as_string(first_ts, unit="s")),
                    "last_ts": str(np.datetime_as_string(last_ts, unit="s")),
                }

            out = {
                "overall": {
                    "n": int(len(self.index)),
                    "y_counts": overall_counts,
                    "first_ts": None
                    if overall_first_ts is None
                    else str(np.datetime_as_string(overall_first_ts, unit="s")),
                    "last_ts": None if overall_last_ts is None else str(np.datetime_as_string(overall_last_ts, unit="s")),
                },
                "per_ticker": per_ticker,
            }
            if display:
                headers = ["ticker", "n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"]
                rows = []
                for ticker in sorted(per_ticker.keys()):
                    d = per_ticker[ticker]
                    rows.append([ticker, d["n"], *[d["y_counts"][label] for label in label_ids], d["first_ts"], d["last_ts"]])
                _print_plain_summary(headers, rows)
            return out

        from kvant.ml_prepare_data.data_loading_utils import summary

        return summary(self, display=display)


class PreparedExperiment:
    """
    Owns: config + store + split indices.
    Provides: get_datasets() and get_loaders() returning train/val/test.
    """

    @classmethod
    def does_experiment_exist(cls, exp_dir: Path) -> bool:
        return os.path.isfile(exp_dir / "config.json")

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.cfg = json.loads((exp_dir / "config.json").read_text())
        self.label_semantics = validate_label_semantics(self.cfg, exp_dir=exp_dir)
        self.label_ids = label_ids_from_semantics(self.label_semantics)
        self.label_meanings = label_meanings_from_semantics(self.label_semantics)
        self.class_names = class_names_from_semantics(self.label_semantics)
        self.n_classes = len(self.label_ids)
        self.L = int(self.cfg["lookback_L"])

        self.store = PreparedStore(exp_dir)

        self.index_train = np.asarray(np.load(exp_dir / "index_train.npy", mmap_mode="r"))
        self.index_val = np.asarray(np.load(exp_dir / "index_val.npy", mmap_mode="r"))
        self.index_test = np.asarray(np.load(exp_dir / "index_test.npy", mmap_mode="r"))

    def get_datasets(self) -> Tuple[IndexWindowDataset, IndexWindowDataset, IndexWindowDataset]:
        ds_train = IndexWindowDataset(self.store, self.index_train, self.L)
        ds_val = IndexWindowDataset(self.store, self.index_val, self.L)
        ds_test = IndexWindowDataset(self.store, self.index_test, self.L)
        return ds_train, ds_val, ds_test

    def get_primary_side_datasets(self) -> Tuple[IndexWindowDataset, IndexWindowDataset, IndexWindowDataset]:
        ds_train = IndexWindowDataset(self.store, self.index_train, self.L, target_mode="primary_side")
        ds_val = IndexWindowDataset(self.store, self.index_val, self.L, target_mode="primary_side")
        ds_test = IndexWindowDataset(self.store, self.index_test, self.L, target_mode="primary_side")
        return ds_train, ds_val, ds_test

    def get_loaders(
        self,
        train_batch_size: int = 256,
        eval_batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        ds_train, ds_val, ds_test = self.get_datasets()

        dl_train = DataLoader(
            ds_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=eval_batch_size,
            shuffle=False,  # keep this for alignment / reproducibility
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dl_train, dl_val, dl_test

    def get_primary_side_loaders(
        self,
        train_batch_size: int = 256,
        eval_batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        ds_train, ds_val, ds_test = self.get_primary_side_datasets()

        dl_train = DataLoader(
            ds_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dl_train, dl_val, dl_test

    def get_split_metadata(self) -> Tuple[List[Optional[dict]], List[Optional[dict]], List[Optional[dict]]]:
        metas_train = self.store.metadata_for_index(self.index_train)
        metas_val = self.store.metadata_for_index(self.index_val)
        metas_test = self.store.metadata_for_index(self.index_test)
        return metas_train, metas_val, metas_test
