from dataclasses import dataclass
from typing import Protocol, List, Optional

import numpy as np
import pandas as pd
import tqdm

from kvant.labelling import tripple_bar_label
from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index


class Labeler(Protocol):
    name: str
    def fit(self, df: pd.DataFrame) -> "Labeler": ...
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, List[Optional[dict]]]: ...


@dataclass(frozen=True)
class TripleBarrierLabeler:
    name: str
    height: float
    width_minutes: int | None = None
    width_periods: int | None = None
    drop_time_exit_label: bool = False  # legacy field; event-outcome artifacts always persist raw labels

    def fit(self, df: pd.DataFrame) -> "TripleBarrierLabeler":
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[Optional[dict]]]:
        df = ensure_utc_sorted_index(df)
        labels = np.full(len(df), -1, dtype=np.int8)
        metadata: list[Optional[dict]] = [None] * len(df)

        for i, t in enumerate(tqdm.tqdm(df.index, desc=f"Labeling {self.name}")):
            res = tripple_bar_label(
                df,
                time_start=t,
                height=self.height,
                width_minutes=self.width_minutes,
                width_periods=self.width_periods,
            )
            if res is None:
                continue

            # store metadata (even if we later drop a label, you can decide policy)
            metadata[i] = res.__dict__


            lab = int(res.label)
            labels[i] = lab

        return labels, metadata
