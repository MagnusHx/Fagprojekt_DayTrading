import numpy as np
import pandas as pd
import pytest

from kvant.ml_prepare_data.samplers import FixedThresholdCUSUMBarSampler


def _ohlcv(close: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 14:30:00", periods=len(close), freq="min", tz="UTC")
    close_arr = np.asarray(close, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr,
            "low": close_arr,
            "close": close_arr,
            "volume": np.ones(len(close_arr), dtype=np.float64),
        },
        index=idx,
    )


def test_fixed_threshold_cusum_uses_same_h_for_all_tickers() -> None:
    sampler = FixedThresholdCUSUMBarSampler(h=0.01, aggregate_ohlcv=False)
    sampler.fit({"AAA": _ohlcv([100.0, 101.2]), "BBB": _ohlcv([200.0, 202.4])})

    assert sampler.get_global_meta()["h"] == pytest.approx(0.01)
    assert sampler.get_global_meta()["tuning"] == "fixed_threshold"
    assert sampler.get_ticker_meta("AAA") == {"h": 0.01, "tuned": False}
    assert sampler.get_ticker_meta("BBB") == {"h": 0.01, "tuned": False}


def test_fixed_threshold_cusum_samples_threshold_events_without_tuning() -> None:
    df = _ohlcv([100.0, 100.4, 101.2, 101.0, 99.8])
    sampler = FixedThresholdCUSUMBarSampler(h=0.01, aggregate_ohlcv=False)

    sampled = sampler.transform(df, ticker="AAA")

    assert list(sampled.index) == [df.index[2], df.index[4]]
    np.testing.assert_allclose(sampled["close"].to_numpy(), np.asarray([101.2, 99.8]))


def test_fixed_threshold_cusum_can_aggregate_ohlcv_segments() -> None:
    df = _ohlcv([100.0, 100.4, 101.2, 101.0, 99.8])
    sampler = FixedThresholdCUSUMBarSampler(h=0.01, aggregate_ohlcv=True)

    sampled = sampler.transform(df, ticker="AAA")

    assert list(sampled.index) == [df.index[2], df.index[4]]
    np.testing.assert_allclose(sampled["open"].to_numpy(), np.asarray([100.0, 101.0]))
    np.testing.assert_allclose(sampled["close"].to_numpy(), np.asarray([101.2, 99.8]))
    np.testing.assert_allclose(sampled["volume"].to_numpy(), np.asarray([3.0, 2.0]))


def test_fixed_threshold_cusum_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        FixedThresholdCUSUMBarSampler(h=0.0)
