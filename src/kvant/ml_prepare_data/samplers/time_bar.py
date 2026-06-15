from dataclasses import dataclass
from typing import Dict
import pandas as pd
from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler


@dataclass
class TimeBarSampler(BaseBarSampler):
    """
    Fixed time-interval bar sampler.
    Aggregates minute-level OHLCV data into k-minute bars.

    Example: time_bar_minutes=15 produces 15-minute candles.
    For intraday trading (9:30-16:00 = 390 minutes), 15-min bars give ~26 bars/day.
    """
    name: str = "time_bar"
    time_bar_minutes: int = 15

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TimeBarSampler":
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "time_bar_minutes": int(self.time_bar_minutes),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        """
        Resample minute-level OHLCV to k-minute bars.

        OHLC aggregation:
        - Open: first value in period
        - High: max value in period
        - Low: min value in period
        - Close: last value in period
        - Volume: sum of volume in period
        """
        df = ensure_utc_sorted_index(df)

        if len(df) == 0:
            return df.copy()

        # Resample to k-minute bars
        freq_str = f"{self.time_bar_minutes}min"

        agg_dict = {}
        if "open" in df.columns:
            agg_dict["open"] = "first"
        if "high" in df.columns:
            agg_dict["high"] = "max"
        if "low" in df.columns:
            agg_dict["low"] = "min"
        if "close" in df.columns:
            agg_dict["close"] = "last"
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"

        # Add any other columns (keep first value)
        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = "first"

        resampled = df.resample(freq_str).agg(agg_dict)

        # Drop rows with NaN close (periods with no data)
        resampled = resampled.dropna(subset=["close"] if "close" in resampled.columns else [])

        return resampled
