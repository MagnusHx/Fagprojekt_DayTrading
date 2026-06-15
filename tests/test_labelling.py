import pandas as pd
import pytest

from kvant.labelling import tripple_bar_label
from kvant.labels import LABEL_DOWN, LABEL_UP
from kvant.ml_prepare_data.labelling.next_bar import NextBarDirectionLabeler


def test_triple_barrier_enters_on_next_sampled_bar_open() -> None:
    """A signal row should not execute at an aggregated bar's historical open."""
    idx = pd.date_range("2024-01-02 15:00:00", periods=4, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 100.0, 100.0, 100.0],
            "high": [1.0, 104.0, 106.0, 100.0],
            "low": [1.0, 99.0, 99.0, 100.0],
            "close": [1.0, 100.0, 105.0, 100.0],
        },
        index=idx,
    )

    label = tripple_bar_label(df, time_start=idx[0], width=5, height=0.05)

    assert label is not None
    assert label.signal_time == idx[0]
    assert label.bar_open_time == idx[1]
    assert label.entry_time == idx[1]
    assert label.bar_close_time == idx[2]
    assert label.label == LABEL_UP
    assert label.pnl_absolute == pytest.approx(5.0)
    assert label.pnl_fraction == pytest.approx(0.05)


def test_triple_barrier_rejects_signal_without_next_entry_bar() -> None:
    idx = pd.date_range("2024-01-02 15:00:00", periods=1, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        },
        index=idx,
    )

    assert tripple_bar_label(df, time_start=idx[0], width=5, height=0.05) is None


def test_next_bar_direction_labeler_uses_event_label_space() -> None:
    idx = pd.date_range("2024-01-02 15:00:00", periods=4, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.0, 100.0],
            "high": [101.0, 102.0, 101.0, 101.0],
            "low": [99.0, 100.0, 99.0, 99.0],
            "close": [100.0, 101.0, 100.0, 100.0],
        },
        index=idx,
    )

    labels, metadata = NextBarDirectionLabeler().transform(df)

    assert labels.tolist() == [LABEL_UP, LABEL_DOWN, LABEL_UP, LABEL_DOWN]
    assert [row["label"] for row in metadata] == [LABEL_UP, LABEL_DOWN, LABEL_UP, LABEL_DOWN]
    assert pd.Timestamp(metadata[0]["bar_close_time"]) == idx[1]
    assert metadata[0]["pnl_fraction"] == pytest.approx(0.01)
    assert metadata[1]["pnl_fraction"] == pytest.approx(-1.0 / 101.0)
    assert metadata[-1]["pnl_fraction"] == 0.0
