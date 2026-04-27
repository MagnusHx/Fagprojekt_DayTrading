import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import pytest

from kvant.labels import (
    label_ids_from_semantics,
    label_semantics_payload,
    model_labels_to_trade_labels,
    validate_label_semantics,
)
from kvant.ml_framework.run_validation import RunValidationError, validate_cv_manifest, validate_prepared_experiment
from kvant.ml_framework.scripts.smoke_prepared_experiment import _smoke_one
from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler


def _write_ticker_fixture(exp_dir, ticker: str, *, labels: list[int], market_rows: int | None = None) -> None:
    tdir = exp_dir / "tickers" / ticker
    tdir.mkdir(parents=True)
    n = len(labels)
    features = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    timestamps = np.asarray(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(i, "m") for i in range(n)],
        dtype="datetime64[ns]",
    )
    label_meta = [{"label": int(y)} for y in labels]
    np.save(tdir / "features.npy", features)
    np.save(tdir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(tdir / "timestamps.npy", timestamps)
    if market_rows is not None:
        market = np.ones((market_rows, 5), dtype=np.float32)
        np.save(tdir / "market_data.npy", market)
    (tdir / "label_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in label_meta))


def _write_prepared_fixture(tmp_path, *, binary: bool = True, with_market_data: bool = True):
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "tickers").mkdir()
    config = {
        "lookback_L": 2,
        "feature_engineer": {
            "feature_names_": ["f0", "f1", "f2"],
            "mean_": [0.0, 0.0, 0.0],
            "std_": [1.0, 1.0, 1.0],
        },
        "labeler": {"width_minutes": 60, "height": 0.01, "drop_time_exit_label": binary},
        "label_semantics": label_semantics_payload(drop_time_exit_label=binary),
    }
    (exp_dir / "config.json").write_text(json.dumps(config))
    (exp_dir / "tickers_all.json").write_text(json.dumps(["AAA"]))
    labels = [0, 1, 1, 0] if binary else [0, 1, 2, 0]
    _write_ticker_fixture(exp_dir, "AAA", labels=labels, market_rows=(len(labels) if with_market_data else None))
    np.save(exp_dir / "index_train.npy", np.asarray([[0, 2]], dtype=np.int64))
    np.save(exp_dir / "index_val.npy", np.asarray([[0, 3]], dtype=np.int64))
    np.save(exp_dir / "index_test.npy", np.asarray([[0, 1]], dtype=np.int64))
    return exp_dir


def test_validate_label_semantics_matches_current_mapping() -> None:
    """The current runtime label semantics should validate cleanly."""
    validate_label_semantics({"label_semantics": label_semantics_payload()})


def test_prepared_experiment_rejects_missing_label_semantics(tmp_path) -> None:
    """Prepared experiments without semantics metadata should fail fast."""
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(json.dumps({"lookback_L": 12}))

    with pytest.raises(RuntimeError, match="label semantics"):
        PreparedExperiment(exp_dir)


def test_prepared_experiment_rejects_mismatched_label_semantics(tmp_path) -> None:
    """Prepared experiments with stale semantics should fail fast."""
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "lookback_L": 12,
                "label_semantics": {
                    "version": 1,
                    "labels": {"0": "up", "1": "exit", "2": "down"},
                },
            }
        )
    )

    with pytest.raises(RuntimeError, match="Regenerate the prepared data"):
        PreparedExperiment(exp_dir)


def test_validate_label_semantics_accepts_binary_directional_mapping() -> None:
    """Directional-only prepared artifacts should validate cleanly."""
    semantics = validate_label_semantics({"label_semantics": label_semantics_payload(drop_time_exit_label=True)})
    assert label_ids_from_semantics(semantics) == (0, 1)


def test_model_labels_to_trade_labels_maps_binary_up_to_canonical_up() -> None:
    """Binary model labels should map into the canonical trade label space."""
    y = np.asarray([0, 1, 1, 0], dtype=np.int64)
    out = model_labels_to_trade_labels(y, label_semantics_payload(drop_time_exit_label=True))
    np.testing.assert_array_equal(out, np.asarray([0, 2, 2, 0], dtype=np.int64))


def test_validate_prepared_experiment_returns_diagnostics_for_binary_fixture(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=True)

    diagnostics = validate_prepared_experiment(exp_dir, require_market_data=True)

    assert diagnostics.label_regime == "binary"
    assert diagnostics.n_classes == 2
    assert diagnostics.has_market_data is True
    assert diagnostics.split_summaries["train"]["n"] == 1


def test_validate_prepared_experiment_infers_event_outcome_for_three_class_fixture(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=False, with_market_data=True)

    diagnostics = validate_prepared_experiment(exp_dir, require_market_data=True)

    assert diagnostics.label_regime == "event_outcome"
    assert diagnostics.n_classes == 3


def test_validate_prepared_experiment_fails_when_market_data_is_required(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=False)

    with pytest.raises(RunValidationError, match="market_data integrity failed"):
        validate_prepared_experiment(exp_dir, require_market_data=True)


def test_smoke_one_materializes_batch_and_forward_pass(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=False, with_market_data=True)

    report = _smoke_one(exp_dir, model_name="conv1d", batch_size=2, require_market_data=True)

    assert report["n_classes"] == 3
    assert report["batch_shape"][0] == 1
    assert report["logits_shape"] == [1, 3]


def test_primary_side_datasets_keep_exit_rows_ignored(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=False, with_market_data=True)
    exp = PreparedExperiment(exp_dir)

    ds_train, ds_val, ds_test = exp.get_primary_side_datasets()

    assert ds_train[0][1].item() in (0, 1)
    assert ds_val[0][1].item() in (0, 1)
    assert ds_test[0][1].item() == -1


class _EveryOtherSampler(BaseBarSampler):
    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        return df.iloc[1::2].copy()


@dataclass
class _LaggedDiffFeatureEngineer:
    name: str = "lagged_diff"

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        close = df["close"].astype(float)
        feat = pd.DataFrame({"close_diff_prev_minute": close.diff().fillna(0.0)}, index=df.index)
        return feat.to_numpy(dtype=np.float32), list(feat.columns)

    def get_meta(self) -> dict:
        return {"name": self.name}


@dataclass
class _ConstantUpLabeler:
    name: str = "constant_up"

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        labels = np.full(len(df), 2, dtype=np.int8)
        metadata = [
            {
                "label": 2,
                "bar_open_time": str(ts),
                "bar_close_time": str(ts),
                "pnl_fraction": 0.01,
                "pnl_absolute": 1.0,
            }
            for ts in df.index
        ]
        return labels, metadata


@dataclass
class _RecordingFeatureSelector:
    name: str = "recording_selector"

    fit_rows: int | None = field(default=None, init=False)
    fit_targets: list[int] = field(default_factory=list, init=False)
    feature_names_: list[str] | None = field(default=None, init=False)
    selected_indices_: list[int] | None = field(default=None, init=False)
    selected_feature_names_: list[str] | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names: list[str]):
        self.fit_rows = int(len(X))
        self.fit_targets = [int(v) for v in y]
        self.feature_names_ = [str(name) for name in feature_names]
        self.selected_indices_ = [0]
        self.selected_feature_names_ = [self.feature_names_[0]]
        return self

    def transform(self, X: np.ndarray, feature_names: list[str]):
        if self.selected_indices_ is None or self.feature_names_ is None:
            raise RuntimeError("selector not fit")
        names = [str(name) for name in feature_names]
        if names != self.feature_names_:
            raise RuntimeError("feature names drifted")
        return np.asarray(X)[:, self.selected_indices_], list(self.selected_feature_names_)

    def get_meta(self) -> dict:
        return {
            "name": self.name,
            "fit_rows": self.fit_rows,
            "fit_targets": list(self.fit_targets),
            "selected_indices": None if self.selected_indices_ is None else list(self.selected_indices_),
            "selected_feature_names": None
            if self.selected_feature_names_ is None
            else list(self.selected_feature_names_),
        }


def test_prepare_experiment_computes_features_before_sampling(tmp_path) -> None:
    from kvant.ml_prepare_data.prepare_experiment import ExperimentConfig, prepare_experiment

    idx = pd.date_range("2024-01-01 09:30:00", periods=6, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14, 15],
            "high": [10, 11, 12, 13, 14, 15],
            "low": [10, 11, 12, 13, 14, 15],
            "close": [10, 11, 12, 13, 14, 15],
            "volume": [100, 101, 102, 103, 104, 105],
        },
        index=idx,
    )
    cfg = ExperimentConfig(
        experiment_name="minute_before_sampling",
        sampler={"name": "every_other"},
        feature_engineer={"name": "lagged_diff"},
        labeler={"name": "constant_up"},
        lookback_L=1,
    )

    prepared = prepare_experiment(
        out_root=tmp_path,
        cfg=cfg,
        sampler=_EveryOtherSampler(name="every_other"),
        fe=_LaggedDiffFeatureEngineer(),
        labeler=_ConstantUpLabeler(),
        ticker_dfs_train={"AAA": df.iloc[:4]},
        ticker_dfs_val={"AAA": df.iloc[4:5]},
        ticker_dfs_test={"AAA": df.iloc[5:]},
        experiment_id="minute_before_sampling",
    )

    saved = np.load(prepared.exp_dir / "tickers" / "AAA" / "features.npy")

    np.testing.assert_allclose(saved[:, 0], np.asarray([1.0, 1.0, 1.0], dtype=np.float32))


def test_prepare_experiment_feature_selector_is_fit_on_train_only_and_rewrites_metadata(tmp_path) -> None:
    from kvant.ml_prepare_data.prepare_experiment import ExperimentConfig, prepare_experiment

    idx = pd.date_range("2024-01-01 09:30:00", periods=6, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14, 15],
            "high": [10, 11, 12, 13, 14, 15],
            "low": [10, 11, 12, 13, 14, 15],
            "close": [10, 11, 12, 13, 14, 15],
            "volume": [100, 101, 102, 103, 104, 105],
        },
        index=idx,
    )
    selector = _RecordingFeatureSelector()
    cfg = ExperimentConfig(
        experiment_name="train_only_feature_selection",
        sampler={"name": "every_other"},
        feature_engineer={"name": "lagged_diff"},
        labeler={"name": "constant_up"},
        lookback_L=1,
        feature_selector={"name": selector.name},
    )

    prepared = prepare_experiment(
        out_root=tmp_path,
        cfg=cfg,
        sampler=_EveryOtherSampler(name="every_other"),
        fe=_LaggedDiffFeatureEngineer(),
        labeler=_ConstantUpLabeler(),
        ticker_dfs_train={"AAA": df.iloc[:4]},
        ticker_dfs_val={"AAA": df.iloc[4:5]},
        ticker_dfs_test={"AAA": df.iloc[5:]},
        experiment_id="train_only_feature_selection",
        feature_selector=selector,
    )

    saved = np.load(prepared.exp_dir / "tickers" / "AAA" / "features.npy")
    saved_cfg = json.loads((prepared.exp_dir / "config.json").read_text())

    assert selector.fit_rows == 1
    assert selector.fit_targets == [1]
    assert saved.shape == (3, 1)
    assert saved_cfg["feature_engineer"]["feature_names_"] == ["close_diff_prev_minute"]
    assert saved_cfg["feature_selector"]["selected_feature_names"] == ["close_diff_prev_minute"]


def test_label_interval_split_safety_purges_boundary_crossing_and_embargoes() -> None:
    from kvant.ml_prepare_data.prepare_experiment import _label_interval_is_safe_for_split

    val_start = np.datetime64("2024-01-02T10:00:00")
    test_start = np.datetime64("2024-01-02T11:00:00")
    embargo = np.timedelta64(15, "m")

    assert not _label_interval_is_safe_for_split(
        np.datetime64("2024-01-02T09:59:00"),
        np.datetime64("2024-01-02T10:01:00"),
        "train",
        val_start,
        test_start,
        embargo,
    )
    assert not _label_interval_is_safe_for_split(
        np.datetime64("2024-01-02T09:50:00"),
        np.datetime64("2024-01-02T09:55:00"),
        "train",
        val_start,
        test_start,
        embargo,
    )
    assert _label_interval_is_safe_for_split(
        np.datetime64("2024-01-02T09:40:00"),
        np.datetime64("2024-01-02T09:45:00"),
        "train",
        val_start,
        test_start,
        embargo,
    )
    assert not _label_interval_is_safe_for_split(
        np.datetime64("2024-01-02T10:55:00"),
        np.datetime64("2024-01-02T10:58:00"),
        "val",
        val_start,
        test_start,
        embargo,
    )
    assert _label_interval_is_safe_for_split(
        np.datetime64("2024-01-02T11:00:00"),
        np.datetime64("2024-01-02T11:05:00"),
        "test",
        val_start,
        test_start,
        embargo,
    )


def test_validate_cv_manifest_accepts_pointer_txt(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=True)
    manifest_json = tmp_path / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold_idx": 0,
                        "exp_id": exp_dir.name,
                        "exp_dir": str(exp_dir),
                    }
                ]
            }
        )
    )
    manifest_ptr = tmp_path / "manifest.txt"
    manifest_ptr.write_text(str(manifest_json))

    out = validate_cv_manifest(manifest_ptr, require_market_data=True)

    assert out["n_folds"] == 1
    assert out["folds"][0]["exp_id"] == exp_dir.name
