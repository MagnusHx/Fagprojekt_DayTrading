from dataclasses import dataclass
from typing import Protocol, List, Optional
import numpy as np
import pandas as pd
import tqdm
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
    - Label = +1 if p_{t+1} > p_t (next bar's close is higher)
    - Label = -1 if p_{t+1} < p_t (next bar's close is lower)
    - Label = 0 if p_{t+1} == p_t (no change, rare)

    The last bar in the series gets label = -1 (no next bar to compare).

    Metadata tracks:
    - next_close: the close price of the next bar
    - return: log return from t to t+1
    """
    name: str = "next_bar_direction"

    def fit(self, df: pd.DataFrame) -> "NextBarDirectionLabeler":
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[Optional[dict]]]:
        df = ensure_utc_sorted_index(df)
        # Use 0/1/2 format (down/exit/up) matching triple-barrier convention
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

            # Direction: 1 (up), 0 (down), 1 (flat → up)
            if next_close > current_close:
                label = 1
            elif next_close < current_close:
                label = 0
            else:
                # Flat/no change → treat as up
                label = 1

            labels[i] = label

            # Log return from current to next bar
            log_return = np.log(next_close / current_close) if current_close > 0 else 0.0

            # Required metadata for split-safety validation (signal_time and bar_close_time)
            metadata[i] = {
                "signal_time": timestamps[i],
                "bar_close_time": timestamps[i],
                "next_close": float(next_close),
                "log_return": float(log_return),
                "label": int(label),
            }

        # Last bar: no next bar, label stays 0 (down)
        if len(df) > 0:
            metadata[len(df) - 1] = {
                "signal_time": timestamps[len(df) - 1],
                "bar_close_time": timestamps[len(df) - 1],
                "label": 0,
            }

        return labels, metadata
