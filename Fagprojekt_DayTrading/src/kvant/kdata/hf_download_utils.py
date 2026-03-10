import pandas as pd
from huggingface_hub import hf_hub_download
from pyarrow import parquet as pq
import os

from kvant.kdata.data_vectorbt_example import REPO_ID


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def load_one_month(month_file: str) -> pd.DataFrame:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=month_file,
        token=_hf_token(),
    )
    print("Downloaded:", local_path)

    # Read whole month file (may be big). If this is too large, see note below for streaming options.
    table = pq.read_table(local_path)
    df = table.to_pandas()

    # Normalize
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"])
    return df
