from dataclasses import dataclass
from typing import Protocol, List, Optional
import numpy as np
import pandas as pd
from kvant.labels import LABEL_DOWN, LABEL_UP
from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index


class Labeler(Protocol):
    name: str
    def fit(self, df: pd.DataFrame) -> "Labeler": ...
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, List[Optional[dict]]]: ...


@dataclass(frozen=True)
class NextBarDirectionLabeler:
    """
    Simplest baseline labeler: next-bar direction.

    For each bar at time t with close price p_t:
    - Label = 2 if p_{t+1} > p_t (next bar's close is higher)
    - Label = 0 if p_{t+1} < p_t (next bar's close is lower)
    - Label = 2 if p_{t+1} == p_t (no change -> treated as up)

    The last bar in the series gets label = 0 (down, no next bar to compare).

    Metadata tracks:
    - signal_time: bar timestamp
    - bar_close_time: next bar timestamp
    - next_close: the close price of the next bar
    - log_return: log return from t to t+1
    - pnl_fraction: simple return from t to t+1 used by meta-labeling
    """
    name: str = "next_bar_direction"
    width_minutes: int = 15  # Backtest window width (default 15 for time_bar sampler)
    height: float = 0.025  # Backtest barrier height (2.5% default, same as triple-barrier)

    def fit(self, df: pd.DataFrame) -> "NextBarDirectionLabeler":
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[Optional[dict]]]:
        df = ensure_utc_sorted_index(df)
        # Use 0/1/2 format (down/exit/up) matching triple-barrier convention.
        # Last bar gets label 0 (down) since there's no next bar to compare
        labels = np.zeros(len(df), dtype=np.int8)
        metadata: list[Optional[dict]] = [None] * len(df)

        if "close" not in df.columns:
            return labels, metadata

        closes = df["close"].values
        timestamps = df.index.to_numpy()

        for i in range(len(df) - 1):
            current_close = closes[i]
            next_close = closes[i + 1]

            # Direction: 2 (up), 0 (down), 2 (flat -> up)
            if next_close > current_close:
                label = LABEL_UP
            elif next_close < current_close:
                label = LABEL_DOWN
            else:
                label = LABEL_UP

            labels[i] = label

            if current_close > 0:
                pnl_fraction = (next_close - current_close) / current_close
                log_return = np.log(next_close / current_close)
            else:
                pnl_fraction = 0.0
                log_return = 0.0

            # Required metadata for split-safety validation (signal_time and bar_close_time)
            metadata[i] = {
                "signal_time": timestamps[i],
                "bar_close_time": timestamps[i + 1],
                "next_close": float(next_close),
                "log_return": float(log_return),
                "pnl_fraction": float(pnl_fraction),
                "pnl_absolute": float(next_close - current_close),
                "label": int(label),
            }

        # Last bar: no next bar, label stays 0 (down)
        if len(df) > 0:
            metadata[len(df) - 1] = {
                "signal_time": timestamps[len(df) - 1],
                "bar_close_time": timestamps[len(df) - 1],
                "pnl_fraction": 0.0,
                "pnl_absolute": 0.0,
                "label": LABEL_DOWN,
            }

        return labels, metadata
