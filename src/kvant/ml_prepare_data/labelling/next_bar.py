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
        labels = np.full(len(df), -1, dtype=np.int8)
        metadata: list[Optional[dict]] = [None] * len(df)

        if "close" not in df.columns:
            return labels, metadata

        closes = df["close"].values

        for i in range(len(df) - 1):
            current_close = closes[i]
            next_close = closes[i + 1]

            # Direction: +1 (up), -1 (down), 0 (flat)
            if next_close > current_close:
                label = 1
            elif next_close < current_close:
                label = -1
            else:
                label = 0

            labels[i] = label

            # Log return from current to next bar
            log_return = np.log(next_close / current_close) if current_close > 0 else 0.0

            metadata[i] = {
                "next_close": float(next_close),
                "log_return": float(log_return),
                "label": int(label),
            }

        # Last bar: no next bar, label stays -1
        # metadata[len(df) - 1] = None

        return labels, metadata
